#!/usr/bin/env python3
"""CLOSED-LOOP rollout: success rate, not imitation error.

    ROLL_AGENT=teacher bash dywa/exp/scripts/eval_rollout.sh 0     # validation
    ROLL_AGENT=policy  ROLL_CKPT=... bash ... 0                    # the real test

WHY THIS EXISTS. Everything measured in this program so far is offline action
MSE -- how well a policy predicts the teacher's commanded action at a random
diffusion noise level. That is not the same as whether the robot pushes the
bottle to the goal, and the two can come apart badly: an imitator can score well
and still fail in closed loop from compounding error, and the dims where the
mass signal lives (joint KP/KD) set contact FORCE rather than the path. Success
rate is the metric the task is defined by; this measures it.

PHASE 1 (ROLL_AGENT=teacher) IS A HARNESS TEST, NOT A RESULT. It drives the
same teacher the corpus was collected with, through this loop, and must
reproduce the success rate measured from the corpus itself:

    in  teacher DR (mass <= 0.50):  0.837
    out of DR      (mass >  0.50):  0.718

If it does not, the loop is wrong and no number it later produces for our own
policy can be trusted. Run it before believing anything in phase 2.

The env, camera, DINO and the rolling elapsed-window track buffer are IMPORTED
from `collect_gap_tracks` rather than reimplemented -- a divergent copy of the
reset-staleness bookkeeping is exactly the kind of drift that produces a
plausible number for the wrong reason. The only thing phase 2 replaces is the
one line that reads the teacher's action.

A CAVEAT TO CARRY INTO ANY RESULT. Test-time adaptation needs ground-truth
tracks of ELAPSED windows, and in simulation those come from recorded rigid-body
poses -- privileged information. It is exact and free here, but it means the
number is "TTT given oracle observation of past object motion", not a
deployable system; on hardware a point tracker would be substituted and its
error inherited.
"""
# MUST precede torch: isaacgym raises "PyTorch was imported before isaacgym
# modules" otherwise. Same ordering as collect_gap_tracks.py:49.
from isaacgym import gymtorch  # noqa: F401
from isaacgym import gymapi    # noqa: F401

import os
import sys
import math
from dataclasses import replace

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import numpy as np
import torch as th

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from util.torch_util import dcn
from env.util import set_seed
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper
from train_ppo_arm import (
    AddTensorboardWriter, setup as setup_logging, load_env, load_agent,
)
from rma_env import setup_rma_env_v2
from train_rma import Config as RMAConfig, update_net_cfg, get_config_path

import collect_gap_tracks as G
from env.env.wrap.nvdr_camera_wrapper import unproject

AGENT = os.environ.get('ROLL_AGENT', 'teacher')
N_EPISODES = int(os.environ.get('ROLL_EPISODES', 320))
MAX_STEPS = int(os.environ.get('ROLL_MAX_STEPS', 20000))
# The swept grid, baked per env exactly as collect_fric_tracks does, so the
# success rates are comparable to the corpus episode-for-episode.
LO = float(os.environ.get('ROLL_LO', 0.1))
HI = float(os.environ.get('ROLL_HI', 1.2))
SPACING = os.environ.get('ROLL_SPACING', 'log')
OUT = os.environ.get('ROLL_OUT', '')
# phase 2
CKPT = os.environ.get('ROLL_CKPT', '')          # a train_meta_policy checkpoint
MODE = os.environ.get('ROLL_MODE', 'blind')     # blind | ttt | oracle | none
N_ACT = int(os.environ.get('ROLL_N_ACTION', 8))  # steps executed per chunk
DINO_K = int(os.environ.get('ROLL_DINO_K', 64))
SPLIT_SEED = int(os.environ.get('ROLL_SPLIT_SEED', 0))
EVERY = int(os.environ.get('ROLL_EVERY', 16))    # belief refit cadence
M_MAX = int(os.environ.get('ROLL_M_MAX', 4))
K_INNER = int(os.environ.get('ROLL_K', 1))


def dino_proj(device):
    """The SAME fixed 1024->k projection the corpus was written with.

    Regenerated from the seed exactly as `train_fric_track.ProjectedDino` does
    (`randn(1024, k, generator=manual_seed(seed)) / sqrt(1024)`). A different
    matrix would put the live features in a different subspace from the one the
    policy was trained on -- a silently wrong model, never an error.
    """
    gen = th.Generator().manual_seed(SPLIT_SEED)
    return (th.randn(1024, DINO_K, generator=gen) / math.sqrt(1024.0)).to(device)


def build_policy(device):
    """-> (inner, policy, phys_embed, cfg) from a train_meta_policy ckpt."""
    sys.path.insert(0, '/home/user/DyWA/dywa/src')
    from models.gap.common import FieldNorm
    from models.fric.latent_model import LatentTrackModel
    from models.policy.joint_dit import JointPolicyDiT
    import torch.nn as nn
    ck = th.load(CKPT, map_location='cpu')
    c = ck['cfg']
    rms = float(ck['track_rms'])
    pol = JointPolicyDiT(
        dino_dim=c['dino_dim'], d_model=c['d_model'], dropout=0.0,
        obj_weight=c['obj_weight'], track_loss_weight=c['lam'],
        cond='latent', cond_dim=c['z_dim'],
        film_init_scale=c.get('film_init_scale', 1.0),
        phys_lo=c['phys_lo'], phys_hi=c['phys_hi'], phys_log=c['phys_log'])
    pol.normalizer.add('gt_3d_track', FieldNorm(3).fit_isotropic_rms(rms))
    pol.normalizer.add('action', FieldNorm(pol.action_dim))
    pol.load_state_dict(ck['policy'], strict=True)
    inner = LatentTrackModel(
        trunk='transformer', d_model=c['d_model'], action_mode=c['action_mode'],
        obj_weight=c['obj_weight'], dropout=0.0, dino_dim=c['dino_dim'],
        z_dim=c['z_dim'], direct_film=c.get('direct_film', False),
        film_init_scale=c.get('film_init_scale', 1.0),
        q0_init_std=c['q0_init_std'], alpha_init=c['alpha_init'],
        phys_lo=c['phys_lo'], phys_hi=c['phys_hi'], phys_log=c['phys_log'])
    inner.normalizer.add('gt_3d_track', FieldNorm(3).fit_isotropic_rms(rms))
    inner.load_state_dict(ck['inner'], strict=True)
    pe = None
    if 'phys_embed' in ck:
        pe = nn.Sequential(nn.Linear(1, c['z_dim']), nn.SiLU(),
                           nn.Linear(c['z_dim'], c['z_dim']))
        pe.load_state_dict(ck['phys_embed'])
        pe = pe.to(device).eval()
    print('[roll] policy ckpt %s (step %s) cond_dim=%d alpha=%.3f'
          % (CKPT, ck.get('gstep'), c['z_dim'], float(inner.alpha())))
    return inner.to(device).eval(), pol.to(device).eval(), pe, c


def bake_mass(env):
    """Per-env mass grid, written pre-`prepare_sim`.

    Copied in CALL SHAPE from `collect_fric_tracks._bake_channel` and kept to
    mass only. `recomputeInertia=True` (the 4th arg) is load-bearing: without it
    inertia stays at the config mass while mass changes, so the object
    translates as if heavy and rotates as if light -- and the readback still
    looks perfect.
    """
    gym, scene = env.gym, env.scene
    n_env = env.num_env
    n_obj = int(scene.obj_handles.shape[1])
    grid = (np.geomspace(LO, HI, n_env) if SPACING == 'log'
            else np.linspace(LO, HI, n_env))
    for i in range(n_env):
        for j in range(n_obj):
            oh = int(scene.obj_handles[i, j])
            pr = gym.get_actor_rigid_body_properties(env.envs[i], oh)
            assert len(pr) == 1, (
                'object actor has %d bodies -- a per-body mass write would not '
                'sum to the labelled value' % len(pr))
            for p in pr:
                p.mass = float(grid[i])
            if not gym.set_actor_rigid_body_properties(env.envs[i], oh, pr, True):
                raise RuntimeError('mass write failed (env %d, obj %d)' % (i, j))
    env._roll_grid = grid
    return grid


@hydra_cli(config_path=get_config_path(), config_name='train_rl')
def main(cfg: RMAConfig):
    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})
    cfg = replace(cfg, train_student_policy=False, dagger=True)
    cfg = replace(cfg, add_phys_params=True)
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    set_seed(cfg.env.seed)

    # Bake the grid inside PushEnv.setup, which runs immediately before
    # prepare_sim -- a mid-sim mass write CORRUPTS the simulation (the object
    # explodes), unlike a mistimed friction write which is merely inert.
    from env.push_env import PushEnv
    orig_setup = PushEnv.setup
    def setup(self, *a, **k):
        out = orig_setup(self, *a, **k)
        bake_mass(self)
        return out
    PushEnv.setup = setup

    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(None)
    cfg = replace(cfg, net=update_net_cfg(cfg.net, env, cfg.state_net_blocklist))
    teacher = load_agent(cfg, env, None, None)
    teacher.eval()
    env = setup_rma_env_v2(cfg, env, teacher, state_size=128,
                           is_student=False, dagger=True)
    img_env = NvdrCameraWrapper(env, G.make_cam_cfg())

    E = env.num_env
    grid = np.asarray(env.unwrap(target=PushEnv)._roll_grid
                      if hasattr(env.unwrap(target=PushEnv), '_roll_grid')
                      else (np.geomspace(LO, HI, E) if SPACING == 'log'
                            else np.linspace(LO, HI, E)))
    print('[roll] agent=%s envs=%d  mass grid [%.3f, %.3f] %s'
          % (AGENT, E, grid.min(), grid.max(), SPACING))
    print('[roll] target: %d episodes' % N_EPISODES)
    assert AGENT in ('teacher', 'policy'), AGENT

    # Read the mass back from the simulator rather than trusting the write.
    raw = G.read_raw_phys(env, E, env.device)
    live = dcn(raw[:, 0]).reshape(-1)
    err = float(np.abs(live - grid).max())
    print('[roll] mass readback max|err| = %.3e  (0 = the write took)' % err)
    assert err < 1e-5, 'the baked mass is not what the simulator holds'

    # ---- phase 2 setup ---------------------------------------------------
    inner = pol = phys_embed = pcfg = None
    dino = mean = std = P = None
    rows = cols = None
    if AGENT == 'policy':
        assert CKPT, 'ROLL_CKPT is required with ROLL_AGENT=policy'
        assert MODE in ('blind', 'ttt', 'oracle'), MODE
        inner, pol, phys_embed, pcfg = build_policy(env.device)
        assert pcfg['dino_dim'] == DINO_K, (pcfg['dino_dim'], DINO_K)
        dino = G.load_dino(env.device)
        mean = th.tensor(G.IMAGENET_MEAN, device=env.device).view(1, 3, 1, 1)
        std = th.tensor(G.IMAGENET_STD, device=env.device).view(1, 3, 1, 1)
        P = dino_proj(env.device)
        rows = th.arange(G.GRID_H, device=env.device) * G.PATCH + G.PATCH // 2
        cols = th.arange(G.GRID_W, device=env.device) * G.PATCH + G.PATCH // 2
        if MODE == 'oracle':
            assert phys_embed is not None, \
                'the checkpoint carries no phys_embed, so oracle is undefined'
        H = pol.H
        q0 = inner.q0.detach().reshape(1, -1).expand(E, -1).contiguous()
        q_live = q0.clone()
        if MODE == 'oracle':
            lo, hi = pcfg['phys_lo'], pcfg['phys_hi']
            mm = th.as_tensor(grid, dtype=th.float32,
                              device=env.device).reshape(E, 1)
            x = mm.clamp(min=1e-6).log() if pcfg['phys_log'] else mm
            a_, b_ = ((math.log(lo), math.log(hi)) if pcfg['phys_log']
                      else (lo, hi))
            with th.no_grad():
                q_live = phys_embed(2.0 * (x - a_) / (b_ - a_) - 1.0)
        chunk = None            # (E, H, 20) current action chunk
        chunk_age = N_ACT       # force a sample on the first step
        hist = __import__('collections').deque(maxlen=H + 1)
        wins = [[] for _ in range(E)]      # completed elapsed windows, per env
        step_in_ep = np.zeros(E, np.int64)
        tip = env.robot.tip_body_indices.long()
        n_body = env.gym.get_env_rigid_body_count(env.envs[0])

    n_done = np.zeros(E, np.int64)
    succ_n = np.zeros(E, np.int64)
    ep_len = [[] for _ in range(E)]
    cur_len = np.zeros(E, np.int64)
    total = 0
    n_refit = 0

    obs = img_env.reset()
    for step in range(MAX_STEPS):
        if AGENT == 'teacher':
            action = obs['teacher_action'][..., 0, :]
        else:
            # ---- perception, identical to the collector's -------------------
            depth, label, color = (obs['gap_depth'], obs['gap_label'],
                                   obs['gap_color'])
            pts = unproject(depth, G.make_cam_cfg())
            anc = pts[:, :, rows][:, :, :, cols]
            anchors = anc.reshape(E, 3, G.N_PATCH).permute(0, 2, 1)
            lab_p = label[:, 0][:, rows][:, :, cols].reshape(E, G.N_PATCH)
            dep_p = depth[:, 0][:, rows][:, :, cols].reshape(E, G.N_PATCH)
            lab_r = th.round(lab_p)
            valid = ((lab_p - lab_r).abs() < 1e-3) & (dep_p > 0) & (lab_r > 0)
            owner = th.where(valid, lab_r.long() - 1,
                             th.full_like(lab_r, -1, dtype=th.long))
            owner_idx = owner.clamp(min=0)
            with th.no_grad():
                feat = G.dino_features(dino, color, mean, std).float() @ P
            eef = env.tensors['body'][tip, :7].reshape(E, 7)
            body = env.tensors['body'][:, :7].reshape(E, n_body, 7)

            # ---- resample the action chunk ---------------------------------
            if chunk_age >= N_ACT:
                b = {'dino_features': feat, 'patch_centers': anchors,
                     'agent_pos': eef, 'q': q_live}
                with th.no_grad():
                    a_chunk, _ = pol.sample(b, noise_seed=1000 + step)
                chunk = a_chunk
                chunk_age = 0
            action = chunk[:, chunk_age, :].clamp(-1.0, 1.0)
            chunk_age += 1

            # ---- the elapsed-window buffer ---------------------------------
            hist.append({'anchor': anchors, 'owner': owner,
                         'owner_idx': owner_idx, 'valid': valid,
                         'pose': body, 'eef': eef, 'feat': feat,
                         'cmd': action.clone(),
                         'step': step_in_ep.copy()})
            if MODE == 'ttt' and len(hist) == H + 1:
                rec = hist[0]
                fut = th.stack([h['pose'] for h in list(hist)[1:]], 1)
                trk, msk = G.compute_tracks(rec, fut, owner_clamp=n_body)
                cmds = th.stack([h['cmd'] for h in list(hist)[1:]], 1)
                for k in range(E):
                    # A window may not span a reset: step_in_ep would have
                    # gone backwards, and the track would then replay a
                    # teleport. Same rule the collector enforces with ep_start.
                    if int(hist[-1]['step'][k]) - int(rec['step'][k]) != H:
                        continue
                    wins[k].append({
                        'dino_features': rec['feat'][k], 'patch_centers':
                        rec['anchor'][k], 'agent_pos': rec['eef'][k],
                        'action_cmd_seq': cmds[k], 'action_seq':
                        th.stack([h['eef'][k] for h in list(hist)[1:]]),
                        'owner': rec['owner'][k], 'gt_3d_track': trk[k],
                        'track_valid_mask': msk[k],
                        'step': int(rec['step'][k])})
                    if len(wins[k]) > 64:
                        wins[k].pop(0)

            # ---- refit the belief ------------------------------------------
            if MODE == 'ttt' and (step_in_ep % EVERY == 0).any():
                import meta_ttt as MT
                due = np.nonzero(step_in_ep % EVERY == 0)[0]
                for k in due:
                    w = wins[k]
                    if not w:
                        q_live[k] = q0[k]
                        continue
                    sel = [w[t] for t in np.unique(np.linspace(
                        0, len(w) - 1, min(M_MAX, len(w))).astype(int))]
                    d = {kk: th.stack([x[kk] for x in sel])
                         for kk in ('dino_features', 'patch_centers',
                                    'agent_pos', 'action_seq',
                                    'action_cmd_seq', 'owner', 'gt_3d_track',
                                    'track_valid_mask')}
                    sup = MT.pack_support([d], len(sel))
                    with th.enable_grad():
                        qk, _ = MT.adapt(inner, sup,
                                         q0[k:k + 1].clone(), K_INNER,
                                         create_graph=False)
                    q_live[k] = qk.reshape(-1).detach()
                    n_refit += 1

        obs, rew, done, info = img_env.step(action)
        done = done.reshape(-1).bool()
        cur_len += 1
        if AGENT == 'policy':
            step_in_ep += 1
        if 'success' in info:
            s = dcn(info['success']).astype(bool).reshape(-1)
        else:
            s = np.zeros(E, bool)
        d = dcn(done).astype(bool).reshape(-1)
        if d.any():
            for k in np.nonzero(d)[0]:
                n_done[k] += 1
                succ_n[k] += int(s[k])
                ep_len[k].append(int(cur_len[k]))
                cur_len[k] = 0
                total += 1
                if AGENT == 'policy':
                    # A new episode starts from the PRIOR with no evidence --
                    # the same rule meta-training uses, and the reason `blind`
                    # and `ttt` agree exactly on an episode's first frames.
                    step_in_ep[k] = 0
                    wins[k] = []
                    if MODE != 'oracle':
                        q_live[k] = q0[k]
            if AGENT == 'policy':
                chunk_age = N_ACT        # resample; the chunk is stale
            if total >= N_EPISODES:
                break
        if step % 500 == 0 and step:
            print('  .. step %d, %d episodes, %d refits'
                  % (step, total, n_refit), flush=True)

    print('\n=== closed-loop rollout: %s%s, %d episodes ==='
          % (AGENT, ('/' + MODE) if AGENT == 'policy' else '', total))
    if AGENT == 'policy':
        print('    n_action_steps=%d  belief refits=%d  (mode %s)'
              % (N_ACT, n_refit, MODE))
    m = grid
    keep = n_done > 0
    sr = np.where(keep, succ_n / np.maximum(n_done, 1), np.nan)
    ed = np.exp(np.linspace(math.log(LO), math.log(HI), 9))
    print('%16s %7s %9s %9s %8s' % ('mass bucket (kg)', 'n', 'success',
                                    'ep len', 'in DR?'))
    for i in range(8):
        sel = keep & (m >= ed[i]) & ((m <= ed[i + 1]) if i == 7
                                     else (m < ed[i + 1]))
        if not sel.any():
            continue
        nn = n_done[sel].sum()
        ss = succ_n[sel].sum() / max(nn, 1)
        ll = np.mean([x for k in np.nonzero(sel)[0] for x in ep_len[k]] or [0])
        tag = 'yes' if ed[i + 1] <= 0.5 else ('part' if ed[i] < 0.5 else 'NO')
        print('%7.3f - %-7.3f %7d %9.3f %9.1f %8s'
              % (ed[i], ed[i + 1], nn, ss, ll, tag))
    for lab, sel in (('IN  DR (<=0.50)', keep & (m <= 0.5)),
                     ('OUT DR (> 0.50)', keep & (m > 0.5))):
        if not sel.any():
            continue
        nn = n_done[sel].sum()
        print('%-18s n=%5d  success %.3f' % (lab, nn, succ_n[sel].sum() / nn))
    print('\nCORPUS REFERENCE (same teacher, measured from the collected data):'
          '\n  in DR 0.837   out of DR 0.718'
          '\nA harness that does not reproduce these is wrong.')
    if OUT:
        np.savez(OUT, mass=m, n=n_done, succ=succ_n)
        print('[saved] %s' % OUT)


if __name__ == '__main__':
    main()
