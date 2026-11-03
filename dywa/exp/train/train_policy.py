#!/usr/bin/env python3
"""Train `JointPolicyDiT` on streamed shards, with NO inner loop.

    POLICY_COND=none bash dywa/exp/scripts/train_policy.sh 0
    POLICY_COND=mass bash dywa/exp/scripts/train_policy.sh 1
    POLICY_COND=mass POLICY_PHYS_SHUFFLE=1 bash dywa/exp/scripts/train_policy.sh 0

WHAT THIS IS FOR. It is the headroom gate for the whole belief program, and it
must run before the meta-trainer. The belief can only ever recover information
the TRUE value carries, so if `cond=mass` does not beat `cond=none` on the
ACTION loss by more than the seed spread, there is nothing for a latent to
infer and the second-order machinery is wasted compute. `shuffled` is the
negative control: a model handed a meaningless input should learn to ignore it
and land back on `none`.

This is the same discipline as `_probe_fric_track_headroom.py` and the
open-loop gate -- with one correction those probes earned the hard way: a
PROXY that nulls does not license skipping the real A/B. This is not a proxy.
It is the same model, the same data and the same loss as the meta run, minus
the inner loop.

READ THE ACTION TERM, NOT THE TOTAL. `L_out` is the action MSE plus lambda
times a track term that is ~87% arm, and the arm's track is the action's own
consequence. This project has inverted a conclusion twice by reading an
aggregate that was ~90% arm. `val/action` is the gate; `val/loss` is context.

THE VALIDATION NOISE IS PINNED. The loss is stochastic in `(t, eps)` and the
effects here are small, so an unpinned val draw contributes its own difference
on top of the one being measured. Every val batch uses `noise_seed = VAL_SEED +
batch_index`, so two checkpoints are scored on the identical noise.

The shard window, its admission guards and its eviction policy are
`train_stream_tracks.ShardWindow`, imported rather than reimplemented, so the
collector contract stays the one `_probe_stream_equivalence.py` verified. Its
knobs are therefore the `STREAM_*` ones (`STREAM_DIR`, `STREAM_WINDOW_SHARDS`,
`STREAM_MIN_SHARDS`, `STREAM_DINO_DIM`); everything owned by this file is
`POLICY_*`.
"""
import os

# Must precede any torch import (meta_train_ttt.py:30-33).
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import gc
import math
import sys
import time

import h5py
import numpy as np
import torch as th
import wandb

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.fric.dataset import FricTrackDataset, read_grid, assert_phys_grid
from models.policy.joint_dit import JointPolicyDiT

import train_stream_tracks as S
from train_fric_track import ShuffledPhys, fit_track_norm, PINNED_ALL, PHYS_NAMES

COND = os.environ.get('POLICY_COND', 'none')
COND_DIM = int(os.environ.get('POLICY_COND_DIM', 32))
SHUFFLE = os.environ.get('POLICY_PHYS_SHUFFLE', '0') == '1'
# Train on SUCCESSFUL episodes only. Worth a flag because it is the natural
# response to "the teacher degrades out of its DR range" -- but note what it
# conditions on: mass strongly affects success, so keeping only successes is a
# COLLIDER on an effect of the swept variable, and it deletes precisely the
# episodes where mass mattered most. Measured on the unfiltered gate, the
# mass-conditioning benefit is +2.83% on failed episodes and -0.18% on
# successful ones, so this is expected to REMOVE the effect, not clean it up.
# Val is left unfiltered so the number stays comparable to the unfiltered arms.
SUCCESS_ONLY = os.environ.get('POLICY_SUCCESS_ONLY', '0') == '1'
# Restrict the swept variable to a sub-range of the corpus grid -- in practice
# the TEACHER's own DR ([0.1, 0.5] kg; see icra_base.yaml), outside which it
# degrades and the measured "mass helps" is largely failure-detection.
# Narrowing also removes the success collider: in-DR the teacher's success is
# FLAT in mass (0.814 -> 0.855), so failed episodes can be kept.
# 0 -> use the corpus grid endpoints.
MASS_LO = float(os.environ.get('POLICY_MASS_LO', 0.0))
MASS_HI = float(os.environ.get('POLICY_MASS_HI', 0.0))
EVAL_DATA = os.environ.get('POLICY_EVAL_DATA', '')
VAL_IDS_FROM = os.environ.get('POLICY_VAL_IDS_FROM', '')
SAVE_DIR = os.environ.get('POLICY_SAVE_DIR',
                          '/home/user/DyWA/output/mass/ckpt_policy')
LAMBDA = float(os.environ.get('POLICY_LAMBDA', 1.0))
OBJ_WEIGHT = float(os.environ.get('POLICY_OBJ_WEIGHT', 9.0))
D_MODEL = int(os.environ.get('POLICY_D_MODEL', 512))
DROPOUT = float(os.environ.get('POLICY_DROPOUT', 0.1))
BATCH = int(os.environ.get('POLICY_BATCH', 16))
LR = float(os.environ.get('POLICY_LR', 1e-4))
WD = float(os.environ.get('POLICY_WD', 1e-6))
WARMUP = int(os.environ.get('POLICY_WARMUP', 4000))
TOTAL_STEPS = int(os.environ.get('POLICY_TOTAL_STEPS', 200000))
BLOCK_STEPS = int(os.environ.get('POLICY_BLOCK_STEPS', 10000))
VAL_FRAMES = int(os.environ.get('POLICY_VAL_FRAMES', 3000))
VAL_SEED = int(os.environ.get('POLICY_VAL_SEED', 9000))
SEED = int(os.environ.get('POLICY_SEED', 0))
DEVICE = os.environ.get('POLICY_DEVICE', 'cuda:0')
LOG_EVERY = int(os.environ.get('POLICY_LOG_EVERY', 100))
TAG = os.environ.get('POLICY_TAG', '')
WANDB_PROJECT = os.environ.get('POLICY_WANDB_PROJECT', 'dywa-joint-policy')
WANDB_MODE = os.environ.get('POLICY_WANDB_MODE', 'online')


class PolicyShardWindow(S.ShardWindow):
    """`ShardWindow` that asserts the stored DINO width instead of projecting.

    Identical in intent to `train_meta_stream.MetaShardWindow`: every shard we
    stream is written at 64-d already, and `ProjectedDino` would both re-project
    an already-narrow file and hide `.episodes`, which `ShuffledPhys` needs.
    """

    def _build(self, path):
        g = read_grid([path] + self.eval_paths)
        assert g['col'] == self.col
        assert_phys_grid([path], self.expect_pinned, self.col, self.grid)
        eps = None
        if SUCCESS_ONLY or MASS_LO or MASS_HI:
            with h5py.File(path, 'r', locking=False) as f:
                keep = np.ones(len(f['episode_ends']), bool)
                if SUCCESS_ONLY:
                    keep &= f['episode_success'][:].astype(bool)
                if MASS_LO or MASS_HI:
                    ends = f['episode_ends'][:]
                    st = np.concatenate([[0], ends[:-1]])
                    pm = f['phys_params'][:, self.col][st]
                    if MASS_LO:
                        keep &= pm >= MASS_LO
                    if MASS_HI:
                        keep &= pm <= MASS_HI
            eps = [int(x) for x in np.nonzero(keep)[0]]
        ds = FricTrackDataset([path], phys_col=self.col, episodes=eps)
        w = ds[0]['dino_features'].shape[-1]
        assert w == S.DINO_DIM, (
            '%s stores %d-d features but STREAM_DINO_DIM=%d' % (path, w, S.DINO_DIM))
        return ShuffledPhys(ds, self.grid, SEED + 1) if SHUFFLE else ds


def fit_action_norm(ds, max_frames=2000):
    """Zero mean / UNIT STD per command channel.

    Per-channel (not the track's single isotropic scale) because the 20
    commanded dims are not one geometric quantity: 3 dpos, 3 dori and 14 joint
    KP/KD, so one scale would leave the gains either saturated or inert.

    Standardisation rather than GAP's [min, max] -> [-1, 1], because the action
    is a DIFFUSION target and its std is what competes with the unit-variance
    gaussian noise. The commanded action is an unsquashed policy mean with
    heavy tails -- 36% of raw values fall outside [-1, 1] -- so limits scaling
    was set by outliers and compressed the bulk to std 0.265-0.386, roughly 3x
    below the track branch. Measured consequence: SNR > 1 on only the first 19
    of 100 cosine timesteps (track: 49), so 81% of action gradient steps saw a
    target buried in noise and the denoiser collapsed to the conditional mean.
    """
    idx = np.linspace(0, len(ds) - 1, min(max_frames, len(ds))).astype(int)
    a = th.stack([ds[int(i)]['action_cmd_seq'] for i in idx])
    f = FieldNorm(a.shape[-1]).fit_standard(a)
    an = f.normalize(a)
    print('[norm] action scale %.4f..%.4f  offset %.4f..%.4f over %d frames'
          % (f.scale.min(), f.scale.max(), f.offset.min(), f.offset.max(),
             len(idx)))
    print('[norm] normalised action std %.4f  max|.| %.3f  (noise std is 1.0)'
          % (an.std(), an.abs().max()))
    return f


@th.no_grad()
def validate(model, ds, dev):
    """Mean action MSE / track term / total on a FIXED frame set and FIXED noise."""
    model.eval()
    dl = th.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False,
                                  num_workers=4, pin_memory=True)
    acc = {'loss': 0.0, 'action': 0.0, 'track': 0.0}
    n = 0
    for bi, b in enumerate(dl):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        if model.cond == 'latent':
            b['q'] = model_q0(model, b['patch_centers'].shape[0], dev)
        _, info = model.compute_loss(b, noise_seed=VAL_SEED + bi)
        for k in acc:
            acc[k] += info[k]
        n += 1
    model.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def model_q0(model, b, dev):
    """Zero belief -- `cond=latent` is not trained by THIS script, but the arm
    must still be runnable so the gate and the meta run share one code path."""
    return th.zeros(b, model.cond_dim, device=dev)


def lr_at(step):
    """Warmup then constant. Not OneCycle: a stream run must be resumable, and
    OneCycle needs `total_steps` up front and raises past it."""
    return LR * min(1.0, float(step + 1) / max(WARMUP, 1))


def main():
    eval_paths = [p for p in EVAL_DATA.split(',') if p and os.path.exists(p)]
    assert eval_paths, 'POLICY_EVAL_DATA must name the frozen corpus'
    assert VAL_IDS_FROM and os.path.exists(VAL_IDS_FROM), \
        'POLICY_VAL_IDS_FROM must name a checkpoint whose val_ids to reuse'
    assert COND in ('none', 'mass', 'latent'), COND
    assert not (SHUFFLE and COND != 'mass'), 'shuffle is only a control for mass'
    th.manual_seed(SEED)
    np.random.seed(SEED)
    dev = th.device(DEVICE)

    g = read_grid(eval_paths)
    col, grid = g['col'], g['grid']
    expect_pinned = {k: v for k, v in PINNED_ALL.items() if k != PHYS_NAMES[col]}
    print('[phys] swept=%s col=%d spacing=%s [%.3f, %.3f]'
          % (g['channel'], col, g['spacing'], g['lo'], g['hi']))

    val_ids = [int(x) for x in th.load(VAL_IDS_FROM, map_location='cpu')['val_ids']]
    if MASS_LO or MASS_HI:
        # The val set must live in the SAME range, or the arms are scored on
        # episodes they were never trained for.
        off, pm_all = 0, {}
        for q in eval_paths:
            with h5py.File(q, 'r', locking=False) as f:
                e_ = f['episode_ends'][:]
                s_ = np.concatenate([[0], e_[:-1]])
                v_ = f['phys_params'][:, col][s_]
            for i_, m_ in enumerate(v_):
                pm_all[off + i_] = float(m_)
            off += len(v_)
        n0 = len(val_ids)
        val_ids = [e for e in val_ids
                   if (not MASS_LO or pm_all[e] >= MASS_LO)
                   and (not MASS_HI or pm_all[e] <= MASS_HI)]
        print('[range] swept variable restricted to [%.3f, %.3f]: val %d -> %d '
              'episodes' % (MASS_LO or g['lo'], MASS_HI or g['hi'], n0,
                            len(val_ids)))
    ds_va_full = FricTrackDataset(eval_paths, phys_col=col, episodes=val_ids)
    stride = max(1, len(ds_va_full) // VAL_FRAMES)
    ds_va = th.utils.data.Subset(ds_va_full, list(range(0, len(ds_va_full), stride)))
    print('[val] %d episodes, %d of %d frames (stride %d)'
          % (len(val_ids), len(ds_va), len(ds_va_full), stride))

    # Both norms are fitted from the SAME fixed draw of the val corpus in every
    # arm, so `none`/`mass`/`shuffled` differ in the conditioning and in nothing
    # else. FieldNorm's scale/offset are persistent buffers and ride inside
    # state_dict, so a per-arm fit would be an invisible second difference.
    trk_norm, rms = fit_track_norm(ds_va_full)
    act_norm = fit_action_norm(ds_va_full)

    model = JointPolicyDiT(
        dino_dim=S.DINO_DIM, d_model=D_MODEL, dropout=DROPOUT,
        obj_weight=OBJ_WEIGHT, track_loss_weight=LAMBDA,
        cond=COND, cond_dim=COND_DIM,
        # NOT the corpus endpoints when a sub-range is in force: leaving them
        # wide would compress the used range into a fraction of the
        # conditioning input's dynamic range, silently.
        phys_lo=(MASS_LO or g['lo']), phys_hi=(MASS_HI or g['hi']),
        phys_log=(g['spacing'] == 'log'))
    model.normalizer.add('gt_3d_track', trk_norm)
    model.normalizer.add('action', act_norm)
    model = model.to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    name = '%s%s%s%s%s' % (COND, '_shuf' if SHUFFLE else '',
                           '_succ' if SUCCESS_ONLY else '',
                           ('_m%g-%g' % (MASS_LO, MASS_HI)) if (MASS_LO or MASS_HI) else '',
                           ('_' + TAG) if TAG else '')
    print('[model] cond=%s cond_dim=%d lambda=%.2f params=%d -> %s'
          % (COND, model.cond_dim, LAMBDA, n_par, name))

    out_dir = os.path.join(SAVE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=name, mode=WANDB_MODE,
               config=dict(cond=COND, cond_dim=COND_DIM, shuffle=SHUFFLE,
                           lam=LAMBDA, d_model=D_MODEL, batch=BATCH, lr=LR,
                           seed=SEED, params=n_par, track_rms=rms,
                           total_steps=TOTAL_STEPS))

    opt = th.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    win = PolicyShardWindow(S.STREAM_DIR, S.WINDOW_SHARDS, col, grid,
                            expect_pinned, eval_paths)
    win.wait_for_min()

    step, best = 0, float('inf')
    while step < TOTAL_STEPS:
        added, evicted = win.refresh()
        dl = th.utils.data.DataLoader(
            win.dataset, batch_size=BATCH, shuffle=True, num_workers=S.WORKERS,
            pin_memory=True, drop_last=True, persistent_workers=False)
        print('[stream] step %d: %d shards (+%d/-%d), %d frames'
              % (step, len(win.resident), added, evicted, win.n_frames),
              flush=True)

        run, nacc, t0 = {'loss': 0.0, 'action': 0.0, 'track': 0.0}, 0, time.time()
        model.train()
        for b in dl:
            if step >= TOTAL_STEPS or nacc >= BLOCK_STEPS:
                break
            for gp in opt.param_groups:
                gp['lr'] = lr_at(step)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            if COND == 'latent':
                b['q'] = model_q0(model, b['patch_centers'].shape[0], dev)
            loss, info = model.compute_loss(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in run:
                run[k] += info[k]
            nacc += 1
            step += 1
            if nacc % LOG_EVERY == 0:
                wandb.log({'train/%s' % k: v / nacc for k, v in run.items()},
                          step=step)
                wandb.log({'train/lr': lr_at(step),
                           'window/frames': win.n_frames,
                           'window/shards': len(win.resident)}, step=step)

        del dl
        gc.collect()
        va = validate(model, ds_va, dev)
        print('  step %d | train a %.5f t %.5f | VAL a %.5f t %.5f L %.5f '
              '| %.0f f/s'
              % (step, run['action'] / max(nacc, 1), run['track'] / max(nacc, 1),
                 va['action'], va['track'], va['loss'],
                 nacc * BATCH / max(time.time() - t0, 1e-6)), flush=True)
        wandb.log({'val/%s' % k: v for k, v in va.items()}, step=step)

        ck = {'model': model.state_dict(), 'cfg': dict(
                  act_norm='standard',
                  cond=COND, cond_dim=COND_DIM, shuffle=SHUFFLE,
                  success_only=SUCCESS_ONLY, mass_lo=MASS_LO, mass_hi=MASS_HI,
                  lam=LAMBDA,
                  d_model=D_MODEL, dino_dim=S.DINO_DIM, obj_weight=OBJ_WEIGHT,
                  phys_lo=g['lo'], phys_hi=g['hi'],
                  phys_log=(g['spacing'] == 'log'), phys_col=col, seed=SEED),
              'track_rms': rms, 'val_ids': val_ids, 'paths': eval_paths,
              'gstep': step, 'val': va}
        th.save(ck, os.path.join(out_dir, 'last.ckpt'))
        if va['action'] < best:
            best = va['action']
            th.save(ck, os.path.join(out_dir, 'best.ckpt'))
            print('  [*] best val action %.5f' % best)

    wandb.finish()


if __name__ == '__main__':
    sys.exit(main())
