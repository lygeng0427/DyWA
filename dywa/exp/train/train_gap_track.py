#!/usr/bin/env python3
"""Train a GAP-style track model (transformer | dit) on single-friction data.

No simulator: this reads the HDF5 written by collect_gap_tracks.py, so it does
NOT need Isaac Gym and can run alongside a collection.

Single-friction setting -- every episode shares one pinned dynamics config, so
the split is by EPISODE and "held out" means a different object pose / goal /
seed, not different physics. `assert_single_physics` refuses to train on data
where that is not true, or where the pinned value is not the one that was asked
for (constant-but-wrong has happened here before).

    GAP_TRUNK=transformer GAP_DATA=/path/a.hdf5,/path/b.hdf5 \
    python3 train_gap_track.py
"""
import os
import sys
import time

import numpy as np
import torch as th
import wandb

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.dataset import (GapTrackDataset, split_by_episode,
                                assert_single_physics, PHYS_NAMES)
from models.gap.common import FieldNorm
from models.gap.track_model import TrackModel

TRUNK = os.environ.get('GAP_TRUNK', 'transformer')
ACTION_MODE = os.environ.get('GAP_ACTION_MODE', 'realized')
OBJ_WEIGHT = float(os.environ.get('GAP_OBJ_WEIGHT', 1.0))
HIST_K = int(os.environ.get('GAP_HIST_K', 0))
HEAD = os.environ.get('GAP_HEAD', 'free')
DATA = os.environ.get('GAP_DATA', '/home/user/DyWA/output/gap/tracks_act_a.hdf5')
SAVE = os.environ.get('GAP_SAVE_DIR', '/home/user/DyWA/output/gap/ckpt')
EPOCHS = int(os.environ.get('GAP_EPOCHS', 40))
BATCH = int(os.environ.get('GAP_BATCH', 16))
LR = float(os.environ.get('GAP_LR', 1e-4))
D_MODEL = int(os.environ.get('GAP_D_MODEL', 512))
VAL_RATIO = float(os.environ.get('GAP_VAL_RATIO', 0.2))
SEED = int(os.environ.get('GAP_SEED', 0))
# Deliberately separate: varying GAP_SEED alone re-inits the network while
# holding the train/val split fixed, so two runs stay comparable frame for
# frame. Tying them would make every seed repeat also change the val set.
SPLIT_SEED = int(os.environ.get('GAP_SPLIT_SEED', 0))
# Draw val only from the first N episodes; extra files go to train.
VAL_POOL = int(os.environ.get('GAP_VAL_POOL', 0)) or None
WORKERS = int(os.environ.get('GAP_WORKERS', 8))
DEVICE = os.environ.get('GAP_DEVICE', 'cuda:0')
WANDB_PROJECT = os.environ.get('GAP_WANDB_PROJECT', 'dywa-gap-track')
WANDB_ENTITY = os.environ.get('GAP_WANDB_ENTITY', '')
WANDB_MODE = os.environ.get('GAP_WANDB_MODE', 'online')   # 'disabled' to skip
# What collect_gap_tracks.sh was asked to pin. Checked against the file.
EXPECT = {'mass': 0.3, 'table_fric': 0.55, 'obj_fric': 0.6,
          'hand_fric': 1.25, 'restitution': 0.0}


def collate(items):
    return {k: th.stack([it[k] for it in items]) for k in items[0]}


def fit_track_norm(ds, max_frames=800):
    """Isotropic RMS over VALID displacement components only.

    Deliberately not per-axis min/max: that would distort displacement
    DIRECTIONS, and a nonzero offset would stop a zero displacement mapping to
    zero. Both matter for tracks.
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
    print('[norm] track rms = %.6f m  over %d valid components' % (rms, n))
    return FieldNorm(3).fit_isotropic_rms(rms), rms


def main():
    paths = [p for p in DATA.split(',') if p]
    th.manual_seed(SEED); np.random.seed(SEED)
    dev = th.device(DEVICE)

    phys, n_ep_seen = assert_single_physics(paths, expect=EXPECT)
    print('[phys] one config over %d episodes: %s' % (
        n_ep_seen, '  '.join('%s=%.4g' % (k, phys[k]) for k in PHYS_NAMES)))

    tr_ids, va_ids, n_ep = split_by_episode(paths, VAL_RATIO, seed=SPLIT_SEED,
                                           val_pool=VAL_POOL)
    print('[data] %d files | %d episodes -> %d train / %d val '
          '(same physics; differing object pose / goal / seed)'
          % (len(paths), n_ep, len(tr_ids), len(va_ids)))

    ds_tr = GapTrackDataset(paths, episodes=tr_ids, hist_k=HIST_K)
    ds_va = GapTrackDataset(paths, episodes=va_ids, hist_k=HIST_K)
    print('[data] frames: train %d  val %d' % (len(ds_tr), len(ds_va)))
    if len(ds_tr) == 0 or len(ds_va) == 0:
        raise SystemExit('empty split -- collect more episodes first')

    norm, rms = fit_track_norm(ds_tr)

    model = TrackModel(trunk=TRUNK, d_model=D_MODEL,
                       action_mode=ACTION_MODE,
                       obj_weight=OBJ_WEIGHT, hist_k=HIST_K,
                       head=HEAD).to(dev)
    model.normalizer.add('gt_3d_track', norm.to(dev))
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print('[model] trunk=%s d_model=%d action=%s obj_w=%g hist_k=%d head=%s'
          '  %.1fM params'
          % (TRUNK, D_MODEL, ACTION_MODE, OBJ_WEIGHT, HIST_K, HEAD, n_par))

    dl_tr = th.utils.data.DataLoader(ds_tr, batch_size=BATCH, shuffle=True,
                                     num_workers=WORKERS, collate_fn=collate,
                                     drop_last=True, pin_memory=True)
    dl_va = th.utils.data.DataLoader(ds_va, batch_size=BATCH, shuffle=False,
                                     num_workers=max(1, WORKERS // 2),
                                     collate_fn=collate, pin_memory=True)

    opt = th.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    steps = max(1, EPOCHS * len(dl_tr))
    sched = th.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps, pct_start=0.05)

    run = '%s_d%d_%s%s%s' % (TRUNK, D_MODEL, ACTION_MODE,
                             ('' if OBJ_WEIGHT == 1.0 else '_ow%g' % OBJ_WEIGHT)
                             + ('' if HIST_K == 0 else '_h%d' % HIST_K)
                             + ('' if HEAD == 'free' else '_' + HEAD),
                             '' if SEED == 0 else '_s%d' % SEED)
    out_dir = os.path.join(SAVE, run)
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=run, entity=(WANDB_ENTITY or None),
               mode=WANDB_MODE, dir=out_dir,
               config={'trunk': TRUNK, 'd_model': D_MODEL, 'epochs': EPOCHS,
                       'action_mode': ACTION_MODE,
                       'obj_weight': OBJ_WEIGHT, 'hist_k': HIST_K,
                       'head': HEAD,
                       'batch': BATCH, 'lr': LR, 'seed': SEED,
                       'val_ratio': VAL_RATIO, 'track_rms': rms,
                       'split_seed': SPLIT_SEED,
                       'n_episodes': n_ep, 'n_train_ep': len(tr_ids),
                       'n_val_ep': len(va_ids), 'train_frames': len(ds_tr),
                       'val_frames': len(ds_va), 'paths': paths, **phys})
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
                batch = {kk: v.to(dev, non_blocking=True) for kk, v in batch.items()}
                # Pin the DiT's (t, noise) on val so the curve reflects the
                # model improving, not a different noise draw each epoch.
                _, info = model.compute_loss(batch, noise_seed=1000 + vk)
                vt += info['loss']; vk += 1
        va_loss = vt / max(vk, 1)

        print('[ep %3d] train %.5f  val %.5f  %.0fs'
              % (ep, tr_loss, va_loss, time.time() - t0), flush=True)
        wandb.log({'epoch': ep, 'train/loss': tr_loss, 'val/loss': va_loss,
                   'train/lr': sched.get_last_lr()[0],
                   'val/gap': va_loss - tr_loss}, step=ep)

        ck = {'model': model.state_dict(), 'trunk': TRUNK, 'epoch': ep,
              'track_rms': rms, 'val_loss': va_loss, 'phys': phys,
              'cfg': {'d_model': D_MODEL, 'action_mode': ACTION_MODE,
                      'obj_weight': OBJ_WEIGHT, 'hist_k': HIST_K,
                      'head': HEAD},
              'train_ids': tr_ids, 'val_ids': va_ids, 'paths': paths}
        th.save(ck, os.path.join(out_dir, 'last.ckpt'))
        # best.ckpt is meaningful again now the latent is gone: val loss is a
        # plain held-out number with nothing unfitted in it. (It was disabled
        # while the belief existed, because val episodes' latent rows never
        # received a gradient and the metric punished learning to use them.)
        if va_loss < best:
            best = va_loss
            th.save(ck, os.path.join(out_dir, 'best.ckpt'))
    print('[done] %d epochs, best val %.5f -> %s' % (EPOCHS, best, out_dir))
    wandb.finish()


if __name__ == '__main__':
    main()
