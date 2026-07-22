#!/usr/bin/env python3
"""
Evaluate a TTT-adapted DyWA student in Isaac Gym.

DyWA analog of `ttt_with_dyn_bars_rollout.py`. Rolls out `StudentAgentTTT` in the
vectorized env; before acting, each env adapts its belief `q0 -> q` by TTT on its
own recent transition history (per-env, `create_graph=False`, shared-frame
normalized). Per-object success is tallied by `CountCategoricalSuccess`.

Sweep the number of inner steps to compare against the no-TTT baseline:
    ++student.ttt.k_inner_steps=0   # baseline (q == q0, no adaptation)
    ++student.ttt.k_inner_steps=1   # one TTT step
    ++student.ttt.k_inner_steps=2

MUST run inside the DyWA Docker container.

Example:
    PYTORCH_JIT=0 python3 eval_ttt.py \
        +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
        ++student.norm=ln ++student.ttt.k_inner_steps=1 \
        ++load_student=./checkpoints/ttt_dywa/best.ckpt \
        ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
        ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
        ++env.single_object_scene.filter_file=/input/DGN/bottle_test.json \
        ++env.single_object_scene.mode=valid ++env.num_env=60
"""

import isaacgym  # noqa: F401

import os
from dataclasses import replace

# Rollout length (env steps). `train_rma.Config` (used here) has no `test_step`
# field — that lives on test_rma.py's Config — so read it from the environment
# instead of cfg. Default matches test_rma.py's test_step=4000.
TEST_STEP = int(os.environ.get('TTT_TEST_STEP', 4000))
# Max envs adapted at once inside masked_adapt (bounds point-attention memory).
ADAPT_ENV_CHUNK = int(os.environ.get('TTT_ADAPT_ENV_CHUNK', 16))

import torch as th
from tqdm.auto import tqdm

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from env.util import set_seed

from train_ppo_arm import (
    setup as setup_logging,
    load_env,
    AddTensorboardWriter,
)
from rma_env import setup_rma_env_v2
from train_rma import Config as RMAConfig, get_config_path
from distill import StudentAgentTTT
from models.ttt import frame_of, to_frame, to_frame_vec, rigid_flow, ee_pose_delta
from envs.cube_env_wrappers import CountCategoricalSuccess
from env.env.wrap.nvdr_record_episode import NvdrRecordEpisode


def _assert_ckpt_covers_policy(student, path):
    """Fail loudly if the checkpoint does not supply the belief/policy weights.

    `student.load(..., strict=False)` is required (the env/teacher pieces live
    elsewhere), but it silently tolerates a checkpoint that has none of the
    modules we are about to evaluate. The dangerous case is a `policy_trunk`
    mismatch: loading a `film_transformer` ckpt into a `dywa`-built student (or
    vice versa) leaves the ENTIRE policy randomly initialised, and the eval then
    reads as "TTT fails" for completely the wrong reason.
    """
    import torch as _th
    from train.ckpt import last_ckpt as _last_ckpt

    raw = _th.load(_last_ckpt(path), map_location='cpu')
    ck = raw.get('self', raw)
    have = set(ck.keys())
    need = set(student.state_dict().keys())

    trunk = student.cfg.ttt.policy_trunk
    critical = ('film_trunk.' if trunk == 'film_transformer'
                else 'decoder.', 'q0', 'dyn.', 'outcome_enc.')
    missing = sorted(k for k in need - have
                     if any(k == c or k.startswith(c) for c in critical))
    if missing:
        raise RuntimeError(
            f"checkpoint {path} is missing {len(missing)} weights the "
            f"policy_trunk={trunk!r} student needs, e.g. {missing[:5]}.\n"
            f"  ckpt has these top-level groups: "
            f"{sorted({k.split('.')[0] for k in have})}\n"
            f"  This is almost always a policy_trunk mismatch — pass "
            f"++student.ttt.policy_trunk=<the trunk the ckpt was trained with>.")
    extra = sorted({k.split('.')[0] for k in have - need})
    if extra:
        print(f"[eval_ttt] note: ckpt has unused groups {extra} "
              f"(expected when switching policy_trunk)")


def masked_adapt(student, q0E, hc, hmacro, hflow, valid, k, alpha):
    """Per-env TTT adaptation with a validity mask over macro-history slots.

    q0E    : [E, D]          (broadcast belief prior)
    hc     : [E, H, N, 3]    before-clouds `cloud_s` (shared-frame normalized)
    hmacro : [E, H, A_macro] macro-actions over each K-window
    hflow  : [E, H, N, 3]    rigid flow over each window (shared-frame normalized)
    valid  : [E, H] bool
    Returns adapted belief [E, D]. Target = OutcomeEncoder(cloud_s, flow).
    """
    E, H = valid.shape
    # Chunk over envs to bound peak memory: OutcomeEncoder/dyn run point-attention
    # over E*H clouds of N points at once, and the N×N attention over all E*H=500
    # windows OOMs at E=50. Each env adapts independently (per-env masked mean +
    # per-env grad), so splitting E is exact, not an approximation.
    if E > ADAPT_ENV_CHUNK:
        outs = []
        for s in range(0, E, ADAPT_ENV_CHUNK):
            sl = slice(s, min(s + ADAPT_ENV_CHUNK, E))
            outs.append(masked_adapt(student, q0E[sl], hc[sl], hmacro[sl],
                                     hflow[sl], valid[sl], k, alpha))
        return th.cat(outs, dim=0)
    D = q0E.shape[-1]
    A = hmacro.shape[-1]
    N = hc.shape[-2]
    vf = valid.float()
    denom = vf.sum(dim=1).clamp_min(1.0)              # [E]
    q = q0E.detach()
    for _ in range(k):
        q_in = q.detach().requires_grad_(True)
        q_exp = q_in.unsqueeze(1).expand(E, H, D).reshape(E * H, D)
        cl = hc.reshape(E * H, N, 3)
        zp = student.dyn(q_exp, hmacro.reshape(E * H, A), cl).reshape(E, H, D)
        za = student.outcome_enc(cl, hflow.reshape(E * H, N, 3)).reshape(E, H, D)
        err = ((zp - za) ** 2).mean(dim=-1)          # [E, H]
        per = (err * vf).sum(dim=1) / denom          # [E]  (masked mean)
        inner = per.sum()
        g = th.autograd.grad(inner, q_in)[0]
        g = th.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
        nrm = g.norm(dim=-1, keepdim=True) + 1e-8
        g = th.where(nrm > 1.0, g / nrm, g)
        q = q - alpha * g
    return th.nan_to_num(q.detach(), nan=0.0)


@hydra_cli(config_path=get_config_path(), config_name='train_rl')
def main(cfg: RMAConfig):
    cfg = recursive_replace_map(cfg, {'finalize': True})
    cfg = replace(cfg, train_student_policy=False, dagger=False)

    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    set_seed(cfg.env.seed)

    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(None)

    # Optional per-episode video recording (succ/fail into separate subdirs when
    # nvdr_record_episode.episode_type=both). Wraps the base env so it can read
    # info['success']; the success accounting and cloud wrappers sit above it.
    if cfg.use_nvdr_record_episode:
        env = NvdrRecordEpisode(cfg.nvdr_record_episode, env, hide_arm=False)

    # Per-object success accounting sits below the cloud wrappers.
    env = CountCategoricalSuccess(env)
    env = setup_rma_env_v2(cfg, env, None,
                           state_size=128, is_student=False, dagger=False)

    device = env.device
    E = env.num_env

    student = StudentAgentTTT(replace(cfg.student, batch_size=E),
                              writer=None, device=device).to(device)
    if cfg.load_student is None:
        raise ValueError("pass ++load_student=<meta-trained ckpt>")
    student.load(cfg.load_student, strict=False)
    _assert_ckpt_covers_policy(student, cfg.load_student)
    student.eval()

    if cfg.load_environment is not None:
        env.load(cfg.load_environment, strict=False)

    tp = student.cfg.ttt
    Kin = int(tp.k_inner_steps)         # number of inner TTT steps (0 = no-TTT baseline)
    Kw = int(tp.window_k)               # macro-transition stride (outcome window)
    H = int(tp.history_len)             # max macro-transitions kept
    A = int(student.cfg.action_size)
    D = int(tp.latent_dim)
    n_gains = A - 6
    is_seq = (tp.macro_action == 'seq')

    obs = env.reset()
    if 'hand_state' not in obs:
        raise KeyError("obs has no 'hand_state' — required for the macro-action")
    # RAW object world pose (pos ‖ quat_xyzw) from the sim root tensor; obs
    # ['object_state'] is standardized and cannot be used for the rigid flow.
    def raw_obj_pose():
        return env.tensors['root'][env.scene.cur_ids.long(), :7]
    N = obs['partial_cloud'].shape[-2]

    # Macro-transition ring buffer (world frame; normalized per-step at adapt time).
    hist_c = th.zeros(E, H, N, 3, device=device)      # cloud_s
    hist_flow = th.zeros(E, H, N, 3, device=device)   # rigid flow over window
    hist_ee = th.zeros(E, H, 6, device=device)        # net EE motion (Δpos ‖ rotvec)
    hist_mg = th.zeros(E, H, n_gains, device=device)  # window-mean gains
    hist_seq = th.zeros(E, H, Kw, A, device=device) if is_seq else None
    valid = th.zeros(E, H, dtype=th.bool, device=device)
    ptr = 0

    # Current-block accumulators (one K-window in progress).
    q0E = student.q0.unsqueeze(0).expand(E, D)
    q_cur = q0E
    act_block = th.zeros(E, Kw, A, device=device)
    blk_cloud = th.zeros(E, N, 3, device=device)
    blk_hand = th.zeros(E, 9, device=device)
    blk_pose = th.zeros(E, 7, device=device)
    blk_ok = th.zeros(E, dtype=th.bool, device=device)

    def _build_macro(hee, hmg, hseq):
        return student.build_macro(hee, hmg, hseq)

    def _adapt_now(cloud_t):
        if Kin <= 0 or not bool(valid.any()):
            return q0E
        center, scale = frame_of(cloud_t)                    # [E,1,3],[E,1,1]
        hc = to_frame(hist_c, center, scale)
        hflow = to_frame_vec(hist_flow, scale)
        hee = hist_ee.clone()
        hee[..., :3] = to_frame_vec(hist_ee[..., :3], scale)
        macro = _build_macro(hee, hist_mg, hist_seq)
        with th.enable_grad():
            return masked_adapt(student, q0E, hc, macro, hflow, valid, Kin, tp.ttt_alpha)

    mode = f'TTT-{Kin}step' if Kin > 0 else 'No-TTT'
    print(f"[eval_ttt] mode={mode}  E={E} H={H} N={N} Kw={Kw} macro={tp.macro_action} alpha={tp.ttt_alpha}")

    for step in tqdm(range(TEST_STEP), desc=f'eval[{mode}]'):
        cloud_t = obs['partial_cloud']
        kc = step % Kw

        if kc == 0:
            # open a new K-window; adapt the belief from the current buffer.
            blk_cloud = cloud_t.clone()
            blk_hand = obs['hand_state'].clone()
            blk_pose = raw_obj_pose().clone()
            blk_ok = th.ones(E, dtype=th.bool, device=device)
            act_block.zero_()
            q_cur = _adapt_now(cloud_t)

        with th.no_grad():
            out = student.belief_forward(obs, q_cur)          # [E, 2, A]
            action = out[..., 0, :]                           # deterministic mu
        act_block[:, kc] = action.detach()

        obs, rew, done, info = env.step(action)

        if kc == Kw - 1:
            # close the window: (cloud_s, macro, rigid flow) → ring buffer.
            pose_sK = raw_obj_pose()
            flow = rigid_flow(blk_cloud, blk_pose, pose_sK)   # [E,N,3]
            hist_c[:, ptr] = blk_cloud
            hist_flow[:, ptr] = flow
            hist_ee[:, ptr] = ee_pose_delta(blk_hand, obs['hand_state'])   # [E,6]
            hist_mg[:, ptr] = act_block[..., 6:].mean(dim=1)
            if is_seq:
                hist_seq[:, ptr] = act_block
            valid[:, ptr] = blk_ok
            ptr = (ptr + 1) % H

        # a reset invalidates all macro-history for that env (windows can't cross it).
        if done.any():
            valid[done] = False
            blk_ok[done] = False

    env.unwrap(target=CountCategoricalSuccess).save()
    print(f"[eval_ttt] done ({mode}); per-object results saved by CountCategoricalSuccess.")


if __name__ == '__main__':
    main()
