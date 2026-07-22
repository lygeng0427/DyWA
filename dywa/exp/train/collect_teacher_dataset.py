#!/usr/bin/env python3
"""
Collect a teacher-rollout dataset for offline TTT meta-training.

DyWA analog of `collect_dataset_bars.py`. Instead of recording expert demos, we
roll out the **privileged teacher** in Isaac Gym and log, per step per env, the
transition needed for TTT:

    partial_cloud_t   [512,3]   (object-segmented, unnormalized — invariant #3)
    goal_cloud_t      [512,3]
    <state keys>                (abs_goal/rel_goal, hand_state, robot_state, previous_action)
    teacher_action    [2,20]    (stacked mu, log_std)  ← imitation target
    partial_cloud_next[512,3]   (object cloud at t+1, the dynamics "outcome")

Only episodes that end in **success** (`info['success']==1`) are kept (mirrors the
bars collector, which only returns successful episodes). The teacher is driven
deterministically by its own `teacher_action` mean, reproducing a deterministic
teacher rollout.

MUST run inside the DyWA Docker container (Isaac Gym).

Example:
    PYTORCH_JIT=0 python3 collect_teacher_dataset.py \
        +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
        ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
        ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
        ++env.single_object_scene.filter_file=/input/DGN/bottle_train.json \
        ++env.num_env=256
"""

import isaacgym  # noqa: F401  (must be imported before torch)

import os
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch as th
from tqdm.auto import tqdm

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from util.torch_util import dcn
from env.util import set_seed

from train_ppo_arm import (
    setup as setup_logging,
    load_agent,
    load_env,
    AddTensorboardWriter,
)
from rma_env import setup_rma_env_v2
from train_rma import Config as RMAConfig, update_net_cfg, get_config_path


# Extra collector knobs are read from the environment to avoid touching the
# shared Hydra Config schema.
def _envint(name, default):
    return int(os.environ.get(name, default))


def _envstr(name, default):
    return os.environ.get(name, default)


NUM_EPISODES = _envint('TTT_NUM_EPISODES', 2000)   # successful episodes to keep
NUM_STEPS = _envint('TTT_MAX_STEPS', 20000)        # hard cap on env steps
MIN_EP_LEN = _envint('TTT_MIN_EP_LEN', 4)          # drop trivially-short episodes
OUT_PATH = _envstr('TTT_OUT', './dataset_ttt_teacher.pkl')
CLOUD_DTYPE = np.float16 if _envstr('TTT_F16', '1') == '1' else np.float32


def _episode_from_steps(steps, obj_id):
    """Stack a list of per-step dicts into arrays keyed `[T, ...]`."""
    ep = {}
    for k in steps[0].keys():
        arr = np.stack([s[k] for s in steps], axis=0)
        if k in ('partial_cloud', 'goal_cloud', 'partial_cloud_next'):
            arr = arr.astype(CLOUD_DTYPE)
        else:
            arr = arr.astype(np.float32)
        ep[k] = arr
    ep['obj_id'] = obj_id
    ep['length'] = len(steps)
    return ep


@hydra_cli(config_path=get_config_path(), config_name='train_rl')
def main(cfg: RMAConfig):
    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})

    # Teacher rollout (not a student run); dagger=True so AddTeacherAction injects
    # `teacher_action` into the obs (see rma_env.setup_rma_env_v2:598).
    cfg = replace(cfg, train_student_policy=False, dagger=True)

    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    set_seed(cfg.env.seed)

    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(None)

    cfg = replace(cfg, net=update_net_cfg(cfg.net, env, cfg.state_net_blocklist))
    teacher_agent = load_agent(cfg, env, None, None)
    teacher_agent.eval()

    env = setup_rma_env_v2(cfg, env, teacher_agent,
                           state_size=128, is_student=False, dagger=True)

    device = env.device
    E = env.num_env

    # Keys of the current-step obs snapshot we log for the query forward.
    state_keys = list(cfg.student.state_keys or [])
    store_keys = state_keys + ['partial_cloud', 'goal_cloud']

    def scene_names():
        try:
            return list(env.scene.cur_names)
        except Exception:
            return [None] * E

    per_env = [[] for _ in range(E)]
    dataset = []
    obs = env.reset()

    pbar = tqdm(total=NUM_EPISODES, desc='collect(success eps)')
    for step in range(NUM_STEPS):
        if 'teacher_action' not in obs:
            raise KeyError("obs has no 'teacher_action' — is dagger/AddTeacherAction active?")
        ta = obs['teacher_action']                     # [E, 2, 20]
        action = ta[..., 0, :]                          # deterministic teacher mu

        snap = {k: dcn(obs[k]) for k in store_keys if k in obs}
        snap['teacher_action'] = dcn(ta)
        # RAW (unnormalized) object world pose (pos ‖ quat_xyzw) straight from the
        # sim root tensor. NOTE: obs['object_state'] is *standardized* by the env
        # normalizer (it is not in the identity `constlist`), so it is NOT a valid
        # SE(3) pose and must not be used for the rigid-flow target.
        obj_ids = env.scene.cur_ids.long()
        snap['object_pose'] = dcn(env.tensors['root'][obj_ids, :7])

        obs, rew, done, info = env.step(action)

        nxt = dcn(obs['partial_cloud'])                 # object cloud at t+1
        done_np = dcn(done).astype(bool).reshape(-1)
        if 'success' in info:
            succ_np = dcn(info['success']).astype(bool).reshape(-1)
        else:
            succ_np = np.zeros(E, dtype=bool)
        names = scene_names()

        for e in range(E):
            rec = {k: snap[k][e] for k in snap}
            rec['partial_cloud_next'] = nxt[e]
            per_env[e].append(rec)

            if done_np[e]:
                if succ_np[e] and len(per_env[e]) >= MIN_EP_LEN:
                    dataset.append(_episode_from_steps(per_env[e], names[e]))
                    pbar.update(1)
                per_env[e] = []

        if len(dataset) >= NUM_EPISODES:
            break
        if step % 500 == 0 and len(dataset) > 0:
            with open(OUT_PATH, 'wb') as f:
                pickle.dump(dataset, f)

    pbar.close()
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(dataset, f)

    n = len(dataset)
    sz = os.path.getsize(OUT_PATH) / 1e6 if os.path.exists(OUT_PATH) else 0.0
    lens = [d['length'] for d in dataset] or [0]
    print(f"\nSaved {n} successful episodes → {OUT_PATH}  ({sz:.1f} MB)")
    print(f"  episode length: mean={np.mean(lens):.1f} min={np.min(lens)} max={np.max(lens)}")


if __name__ == '__main__':
    main()
