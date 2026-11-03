#!/usr/bin/env python3
"""Train a track model on friction-swept data, with or without conditioning.

The experiment this exists for is one A/B: does handing the model the true
friction recover error that varying friction costs it? Everything except
`FRIC_PHYS_COND` is held at the configuration the fixed-physics model was tuned
at (`action_mode=realized_gain`, `obj_weight=9`, `d_model=512`), so the two runs
differ in the conditioning path and nothing else.

TWO CONTROLS, not one. Val episodes come from the same 32 frictions as train, so
a friction scalar doubles as a 32-way episode ID and a "win" could be pure
memorisation:

  FRIC_PHYS_SHUFFLE=1   permutes mu across episodes. Same input distribution,
                        same parameter count, no physical meaning. If this wins
                        as much as the real labels, the gain is leakage.
  FRIC_HELDOUT_MU=8     holds out 8 whole friction VALUES. Memorising buckets
                        cannot help; only mu -> dynamics can.

No simulator here: reads the HDF5 only, so it runs alongside a collection.

    FRIC_TRUNK=transformer FRIC_PHYS_COND=both \
    FRIC_DATA=/path/a.hdf5,/path/b.hdf5 python3 train_fric_track.py
"""
import os
import sys
import time

import h5py
import numpy as np
import torch as th
import wandb

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.gap.dataset import PHYS_NAMES
from models.gap.dataset import count_episodes
from models.fric.dataset import (FricTrackDataset, read_grid, episode_mu,
                                 episode_success,
                                 bucketize, balanced_episode_split,
                                 heldout_mu_split, assert_phys_grid)
from models.fric.track_model import FricTrackModel

TRUNK = os.environ.get('FRIC_TRUNK', 'transformer')
PHYS_COND = os.environ.get('FRIC_PHYS_COND', 'both')
SHUFFLE = os.environ.get('FRIC_PHYS_SHUFFLE', '0') == '1'
HELDOUT = int(os.environ.get('FRIC_HELDOUT_MU', 0))
ACTION_MODE = os.environ.get('FRIC_ACTION_MODE', 'realized_gain')
OBJ_WEIGHT = float(os.environ.get('FRIC_OBJ_WEIGHT', 9.0))
DATA = os.environ.get('FRIC_DATA',
                      '/home/user/DyWA/output/fric/tracks_tf_a.hdf5,'
                      '/home/user/DyWA/output/fric/tracks_tf_b.hdf5')
SAVE = os.environ.get('FRIC_SAVE_DIR', '/home/user/DyWA/output/fric/ckpt')
EPOCHS = int(os.environ.get('FRIC_EPOCHS', 10))
BATCH = int(os.environ.get('FRIC_BATCH', 16))
LR = float(os.environ.get('FRIC_LR', 1e-4))
D_MODEL = int(os.environ.get('FRIC_D_MODEL', 512))
VAL_RATIO = float(os.environ.get('FRIC_VAL_RATIO', 0.2))
PER_BUCKET = int(os.environ.get('FRIC_PER_BUCKET', 0)) or None
# Trim every bucket to the smallest episode count? Only correct when the raw
# FRAME counts are skewed -- see balanced_episode_split's docstring. Off by
# default now: it is a no-op on a frame-balanced dataset except for throwing
# away half the data, and on the mass dataset it actively creates a 2.6x frame
# imbalance. `_probe_fric_census.py` prints the raw balance; decide from that.
TRIM = os.environ.get('FRIC_TRIM', '0') == '1'
# Draw VAL only from the first N episodes, so appended files enlarge TRAIN and
# the val set stays bit-identical across dataset sizes. Required for any
# data-scaling comparison; 0 = off.
VAL_POOL = int(os.environ.get('FRIC_VAL_POOL', 0))
# Reuse ANOTHER checkpoint's exact val episode ids. This is the only way to make
# a data-scaling comparison honest.
#
# FRIC_VAL_POOL alone is NOT sufficient and I got this wrong once: it restricts
# val to ids < N, but the per-bucket permutation runs over ALL ids including the
# appended ones, so the chosen SUBSET still moves. Measured: adding two files
# left only 94 of 419 val episodes in common (22%). Pointing at a reference
# ckpt fixes the set by construction, and the assertion below proves it rather
# than trusting it.
VAL_IDS_FROM = os.environ.get('FRIC_VAL_IDS_FROM', '')
# Regularisation knobs. Defaults reproduce every run so far, so an unset sweep
# variable cannot silently change an old result.
WD = float(os.environ.get('FRIC_WD', 1e-6))
DROPOUT = float(os.environ.get('FRIC_DROPOUT', 0.1))
# DINO input width. 1024 = the stored features untouched. k>0 projects them to k
# dims with a FIXED random matrix; 0 zeroes them entirely (geometry only).
# The point: 391x1024 ~ 400k numbers per frame describe a scene holding ONE known
# bottle, whose physics needs a dozen numbers. That is the memorisation surface
# to test, and it is also 93% of the bytes on disk.
DINO_DIM = int(os.environ.get('FRIC_DINO_DIM', 1024))
SEED = int(os.environ.get('FRIC_SEED', 0))
# Separate from FRIC_SEED on purpose: varying the init must not also change the
# val episodes, or two runs are not comparable frame for frame.
SPLIT_SEED = int(os.environ.get('FRIC_SPLIT_SEED', 0))
WORKERS = int(os.environ.get('FRIC_WORKERS', 8))
DEVICE = os.environ.get('FRIC_DEVICE', 'cuda:0')
WANDB_PROJECT = os.environ.get('FRIC_WANDB_PROJECT', 'dywa-fric-track')
WANDB_MODE = os.environ.get('FRIC_WANDB_MODE', 'online')
# Train on successful episodes only? Off by default: task success is an effect
# of the swept variable, so conditioning on it is a COLLIDER that preferentially
# deletes the episodes where the variable mattered most. Kept as a knob for
# comparability with the friction arm, whose dataset is success-only.
SUCCESS_ONLY = os.environ.get('FRIC_SUCCESS_ONLY', '0') == '1'

# What collect_fric_tracks.sh pins, for EVERY channel. The swept column is
# dropped below -- listing only four of the five here is how the mass run used
# to fail: 'mass' became a dead entry (assert_phys_grid skips the free column)
# while 'table_fric' went unchecked and then tripped the constancy test.
PINNED_ALL = {'mass': 0.3, 'table_fric': 0.55, 'obj_fric': 0.6,
              'hand_fric': 1.25, 'restitution': 0.0}


def collate(items):
    return {k: th.stack([it[k] for it in items]) for k in items[0]}


def fit_track_norm(ds, max_frames=800):
    """Isotropic RMS over VALID displacement components only.

    Not per-axis min/max: that would distort displacement DIRECTIONS, and a
    nonzero offset would stop a zero displacement mapping to zero.
    """
    idx = np.linspace(0, len(ds) - 1, min(max_frames, len(ds))).astype(int)
    sq, n = 0.0, 0
    for i in idx:
        it = ds[int(i)]
        t, m = it['gt_3d_track'].numpy(), it['track_valid_mask'].numpy()
        if m.any():
            v = t[m]
            sq += float((v ** 2).sum()); n += v.size
    rms = float(np.sqrt(sq / max(n, 1)))
    print('[norm] track rms = %.6f m over %d valid components' % (rms, n))
    return FieldNorm(3).fit_isotropic_rms(rms), rms


class ProjectedDino(th.utils.data.Dataset):
    """Narrow (or remove) the DINO features before the model sees them.

    DINO_DIM=0 zeroes them, leaving only geometry (`patch_centers`) and the
    action -- the strongest form of the test. k>0 applies a FIXED random
    projection 1024 -> k, shared by train and val and regenerated identically
    from the seed, so the two splits cannot see different features.

    A projection rather than a learned bottleneck on purpose: a learned layer
    would be trained on the same data and could itself memorise, which is the
    thing under test.
    """

    def __init__(self, ds, k, seed=0):
        self.ds, self.k = ds, int(k)
        if self.k > 0:
            gen = th.Generator().manual_seed(seed)
            self.P = th.randn(1024, self.k, generator=gen) / np.sqrt(1024.0)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        d = it['dino_features']
        if self.k == 0:
            it['dino_features'] = th.zeros_like(d[..., :1])
        elif d.shape[-1] != self.k:
            it['dino_features'] = d @ self.P
        # else: the file already stores exactly k dims (written projected by
        # collect_fric_tracks.py). Projecting again would apply the matrix twice
        # and silently destroy the features.
        return it


class ShuffledPhys(th.utils.data.Dataset):
    """Wraps a dataset and replaces `phys` with another episode's value.

    A per-EPISODE permutation, not per-frame noise: mu is constant within an
    episode in the real data, so per-frame randomisation would be a strictly
    easier thing to ignore and would flatter the true-label run.
    """

    def __init__(self, ds, grid, seed):
        self.ds = ds
        rng = np.random.default_rng(seed)
        ep = np.asarray(ds.episodes)
        self.map = dict(zip(ep.tolist(),
                            grid[rng.integers(0, len(grid), len(ep))].tolist()))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        it = self.ds[i]
        it['phys'] = th.tensor([self.map[int(it['episode'])]], dtype=th.float32)
        return it


def main():
    paths = [p for p in DATA.split(',') if p and os.path.exists(p)]
    assert paths, 'no data files found in FRIC_DATA=%s' % DATA
    th.manual_seed(SEED); np.random.seed(SEED)
    dev = th.device(DEVICE)

    g = read_grid(paths)
    col, grid = g['col'], g['grid']
    expect_pinned = {k: v for k, v in PINNED_ALL.items()
                     if k != PHYS_NAMES[col]}
    pinned, n_ep_seen, n_hit = assert_phys_grid(paths, expect_pinned, col, grid)
    print('[phys] swept=%s (col %d, %s) over %d/%d grid values in [%.4f, %.4f]'
          % (g['channel'], col, g['spacing'], n_hit, len(grid),
             g['lo'], g['hi']))
    print('[phys] pinned: %s' % '  '.join(
        '%s=%.4g' % (k, v) for k, v in pinned.items() if v is not None))

    if VAL_IDS_FROM:
        _ref = th.load(VAL_IDS_FROM, map_location='cpu')
        _va = np.sort(np.asarray([int(x) for x in _ref['val_ids']], np.int64))
        n_ep_all = count_episodes(paths)
        assert _va.max() < n_ep_all, (
            'reference val id %d exceeds this dataset (%d episodes) -- the '
            'reference must be a PREFIX of these paths, in the same order'
            % (_va.max(), n_ep_all))
        # The reference's files must be the leading entries here, or id k means a
        # different episode in the two runs and the "identical val set" is a lie.
        assert [os.path.realpath(x) for x in _ref['paths']] == \
               [os.path.realpath(x) for x in paths[:len(_ref['paths'])]], (
            'reference ckpt paths %s are not a prefix of FRIC_DATA %s'
            % (_ref['paths'], paths))
        tr_ids = np.setdiff1d(np.arange(n_ep_all, dtype=np.int64), _va)
        va_ids, n_ep = _va, n_ep_all
        counts = np.bincount(bucketize(episode_mu(paths, col), grid),
                             minlength=len(grid))
        print('[data] val ids REUSED from %s: %d episodes (identical to that '
              'run); all %d remaining episodes -> train'
              % (VAL_IDS_FROM, len(va_ids), len(tr_ids)))
    else:
        tr_ids = va_ids = None

    keep = None
    if SUCCESS_ONLY:
        keep = episode_success(paths)
        print('[data] SUCCESS-ONLY: %d/%d episodes kept'
              % (int(keep.sum()), len(keep)))

    if VAL_IDS_FROM:
        pass                             # already fixed above
    elif HELDOUT:
        tr_ids, va_ids, n_ep, held = heldout_mu_split(paths, col, grid,
                                                      HELDOUT, SPLIT_SEED,
                                                      keep=keep)
        print('[data] HELD-OUT-MU split: %d values never seen in training: %s'
              % (len(held), np.round(held, 3).tolist()))
        counts = np.bincount(bucketize(episode_mu(paths, col), grid),
                             minlength=len(grid))
    else:
        tr_ids, va_ids, n_ep, counts, n_keep = balanced_episode_split(
            paths, col, grid, VAL_RATIO, SPLIT_SEED, PER_BUCKET, keep=keep,
            trim=TRIM, val_pool=VAL_POOL)
        print('[data] per-bucket split (trim=%s): raw episode counts %d..%d%s'
              % (TRIM, counts[counts > 0].min(), counts.max(),
                 ', trimmed to %d each' % n_keep if TRIM or PER_BUCKET else ''))
    print('[data] %d files | %d episodes -> %d train / %d val'
          % (len(paths), n_ep, len(tr_ids), len(va_ids)))

    ds_tr = FricTrackDataset(paths, phys_col=col, episodes=tr_ids)
    ds_va = FricTrackDataset(paths, phys_col=col, episodes=va_ids)
    print('[data] frames: train %d  val %d' % (len(ds_tr), len(ds_va)))
    assert len(ds_tr) and len(ds_va), 'empty split'

    if DINO_DIM != 1024:
        # Files may be MIXED: some written with raw 1024-d features, some already
        # projected by the collector. The wrapper projects only what needs it, so
        # every sample reaches the model at DINO_DIM either way. The seed must
        # match whatever the collector used (`dino_proj_seed` in the attrs), or
        # the two sources land in different subspaces -- a mismatch that shows up
        # as a bad model, never as an error.
        widths, seeds = set(), set()
        for pth in paths:
            with h5py.File(pth, 'r', locking=False) as _f:
                widths.add(int(_f['dino_features'].shape[-1]))
                if 'dino_proj_seed' in _f.attrs:
                    seeds.add(int(_f.attrs['dino_proj_seed']))
        assert len(seeds) <= 1, (
            'files were projected with different seeds %s -- they cannot be '
            'pooled' % sorted(seeds))
        if seeds:
            assert int(list(seeds)[0]) == SPLIT_SEED, (
                'files were projected with seed %d but FRIC_SPLIT_SEED=%d is '
                'what the wrapper would use on the raw files; set '
                'FRIC_SPLIT_SEED=%d' % (list(seeds)[0], SPLIT_SEED,
                                        list(seeds)[0]))
        ds_tr = ProjectedDino(ds_tr, DINO_DIM, SPLIT_SEED)
        ds_va = ProjectedDino(ds_va, DINO_DIM, SPLIT_SEED)
        print('[data] DINO -> %s | stored widths in files: %s'
              % ('ZEROED (geometry only)' if DINO_DIM == 0
                 else 'dim %d' % DINO_DIM, sorted(widths)))

    norm, rms = fit_track_norm(ds_tr)
    if SHUFFLE:
        # Both sides, or val would score the model against labels drawn from a
        # different distribution than it trained on.
        ds_tr = ShuffledPhys(ds_tr, grid, SPLIT_SEED)
        ds_va = ShuffledPhys(ds_va, grid, SPLIT_SEED + 1)
        print('[control] mu labels SHUFFLED across episodes -- any gain here is '
              'leakage, not physics')

    model = FricTrackModel(trunk=TRUNK, d_model=D_MODEL,
                           action_mode=ACTION_MODE, obj_weight=OBJ_WEIGHT,
                           phys_cond=PHYS_COND, dropout=DROPOUT,
                           dino_dim=max(DINO_DIM, 1),
                           phys_lo=g['lo'], phys_hi=g['hi'],
                           phys_log=(g['spacing'] == 'log')).to(dev)
    model.normalizer.add('gt_3d_track', norm.to(dev))
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print('[model] trunk=%s phys_cond=%s action=%s obj_w=%g n_prefix=%d '
          'wd=%g dropout=%g dino=%s %.1fM params'
          % (TRUNK, PHYS_COND, ACTION_MODE, OBJ_WEIGHT, model.n_prefix,
             WD, DROPOUT, ('ablated' if DINO_DIM == 0 else DINO_DIM), n_par))

    dl_tr = th.utils.data.DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                                     num_workers=WORKERS, collate_fn=collate,
                                     drop_last=True, pin_memory=True)
    dl_va = th.utils.data.DataLoader(ds_va, batch_size=BATCH, shuffle=False,
                                     num_workers=max(1, WORKERS // 2),
                                     collate_fn=collate, pin_memory=True)

    opt = th.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = th.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=max(1, EPOCHS * len(dl_tr)), pct_start=0.05)

    # EVERY knob that changes what the model is must appear here, or two runs
    # share a directory and the second silently overwrites the first. action_mode
    # and success_only were missing: a `command` run and a `realized_gain` run
    # both landed on `transformer_none`, and only the log said which was which.
    tag = ''
    if WD != 1e-6:
        tag += '_wd%g' % WD
    if DROPOUT != 0.1:
        tag += '_do%g' % DROPOUT
    if DINO_DIM != 1024:
        tag += '_dino%d' % DINO_DIM
    run = '%s_%s%s%s%s%s%s' % (TRUNK, PHYS_COND,
                               '' if ACTION_MODE == 'realized_gain'
                               else '_' + ACTION_MODE,
                               '_succ' if SUCCESS_ONLY else '',
                               '_shuf' if SHUFFLE else '',
                               '_ho%d' % HELDOUT if HELDOUT else '',
                               ('' if SEED == 0 else '_s%d' % SEED) + tag)
    out_dir = os.path.join(SAVE, run)
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=run, mode=WANDB_MODE, dir=out_dir,
               config={'trunk': TRUNK, 'phys_cond': PHYS_COND,
                       'shuffle': SHUFFLE, 'heldout_mu': HELDOUT,
                       'action_mode': ACTION_MODE, 'obj_weight': OBJ_WEIGHT,
                       'd_model': D_MODEL, 'epochs': EPOCHS, 'batch': BATCH,
                       'lr': LR, 'seed': SEED, 'split_seed': SPLIT_SEED,
                       'track_rms': rms, 'fric_channel': g['channel'],
                       'fric_lo': g['lo'], 'fric_hi': g['hi'],
                       'fric_spacing': g['spacing'], 'wd': WD,
                       'dropout': DROPOUT, 'dino_dim': DINO_DIM,
                       'val_pool': VAL_POOL, 'val_ids_from': VAL_IDS_FROM,
                       'success_only': SUCCESS_ONLY, 'trim': TRIM,
                       'n_episodes': n_ep, 'n_train_ep': len(tr_ids),
                       'n_val_ep': len(va_ids), 'train_frames': len(ds_tr),
                       'val_frames': len(ds_va), 'paths': paths})
    print('[run] %s' % out_dir)
    if WANDB_MODE != 'disabled':
        print('[wandb] %s' % wandb.run.get_url(), flush=True)

    best = float('inf')
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = k = 0.0
        for batch in dl_tr:
            batch = {kk: v.to(dev, non_blocking=True) for kk, v in batch.items()}
            loss, info = model.compute_loss(batch)
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += info['loss']; k += 1
        tr_loss = tot / max(k, 1)

        model.eval(); vt = vk = 0.0
        with th.no_grad():
            for batch in dl_va:
                batch = {kk: v.to(dev, non_blocking=True)
                         for kk, v in batch.items()}
                # Pin the DiT's (t, noise) on val so the curve tracks the model
                # improving, not a different noise draw each epoch.
                _, info = model.compute_loss(batch, noise_seed=1000 + vk)
                vt += info['loss']; vk += 1
        va_loss = vt / max(vk, 1)

        print('[ep %3d] train %.5f  val %.5f  %.0fs'
              % (ep, tr_loss, va_loss, time.time() - t0), flush=True)
        wandb.log({'epoch': ep, 'train/loss': tr_loss, 'val/loss': va_loss,
                   'train/lr': sched.get_last_lr()[0],
                   'val/gap': va_loss - tr_loss}, step=ep)

        ck = {'model': model.state_dict(), 'trunk': TRUNK, 'epoch': ep,
              'track_rms': rms, 'val_loss': va_loss,
              'cfg': {'d_model': D_MODEL, 'action_mode': ACTION_MODE,
                      'obj_weight': OBJ_WEIGHT, 'phys_cond': PHYS_COND,
                      'phys_lo': g['lo'], 'phys_hi': g['hi'],
                      'phys_log': (g['spacing'] == 'log'),
                      'phys_channel': g['channel'],
                      'dropout': DROPOUT, 'dino_dim': DINO_DIM, 'wd': WD,
                      'split_seed': SPLIT_SEED, 'val_pool': VAL_POOL,
                      'val_ids_from': VAL_IDS_FROM,
                      'phys_col': col, 'shuffle': SHUFFLE,
                      'success_only': SUCCESS_ONLY,
                      'heldout_mu': HELDOUT},
              'train_ids': tr_ids, 'val_ids': va_ids, 'paths': paths}
        th.save(ck, os.path.join(out_dir, 'last.ckpt'))
        if va_loss < best:
            best = va_loss
            th.save(ck, os.path.join(out_dir, 'best.ckpt'))
    print('[done] %d epochs, best val %.5f -> %s' % (EPOCHS, best, out_dir))
    wandb.finish()


if __name__ == '__main__':
    main()
