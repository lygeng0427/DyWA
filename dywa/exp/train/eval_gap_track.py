#!/usr/bin/env python3
"""Held-out tracking error for a GAP track model, in mm.

Ports the metric from third_party/GAP/scripts/analyze_track_loss.py (mean
per-step L2 in mm) with ONE correction that decides whether the number means
anything.

  GAP's "move" group is a VALIDITY split, not a movement split -- its mask is
  `depth_valid & is_object`. In DyWA that is worse than in GAP: the TABLE is a
  rigid body like any other, so table patches are `track_valid_mask_gt=True`
  with a ground-truth track of exactly 0. An mm number averaged over "valid"
  patches is therefore dominated by patches whose GT is zero, and A MODEL THAT
  PREDICTS ALL ZEROS SCORES WELL ON IT.

So these subsets are reported --

    all       every patch
    valid     track_valid_mask_gt, i.e. GAP's "move" (includes the static table)
    moving    ||GT endpoint displacement|| > tau, the patches that actually moved
    mv:object moving patches on body 1, the pushed object
    mv:arm    moving patches on the franka links (bodies >= 7)

The last two exist because the model is given the arm's future EEF path. That
makes the ARM's own tracks nearly determined -- close to a kinematics lookup --
while the OBJECT's still depend on contact and friction. They are different
problems and a single number hides which one improved. `mv:object` is the one
that measures scene dynamics.

-- each against a CONSTANT-ZERO-PREDICTION baseline. The zero row costs nothing
and is the whole point: beating it decisively on `moving` is what "the model
works" means. Doing well on `all` is not evidence of anything.

There is no latent to adapt: every episode shares one pinned friction config,
so the model is a plain conditional regressor and EVERY val frame is scored.
(While a per-episode latent existed this had to fit it on a disjoint half of
each episode first, because the fit objective was the GT track itself.)
"""
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.dataset import GapTrackDataset, PHYS_NAMES
from models.gap.common import FieldNorm
from models.gap.track_model import TrackModel

CKPT = os.environ.get('GAP_CKPT', '')
DEVICE = os.environ.get('GAP_DEVICE', 'cuda:0')
BATCH = int(os.environ.get('GAP_BATCH', 16))
MAX_FRAMES = int(os.environ.get('GAP_EVAL_FRAMES', 100000))
TAU = float(os.environ.get('GAP_TAU', 0.01))        # 1 cm, as the loss uses
# DDIM steps at eval (dit only). GAP.yaml uses 10; the chain measurably degrades
# its own first estimate here, so 1 is the setting that represents the trunk.
INFER_STEPS = int(os.environ.get('GAP_INFER_STEPS', 0))


def collate(items):
    return {k: th.stack([it[k] for it in items]) for k in items[0]}


def build(ck, dev):
    m = TrackModel(trunk=ck['trunk'], d_model=ck['cfg']['d_model'],
                   action_mode=ck['cfg'].get('action_mode', 'realized'),
                   hist_k=ck['cfg'].get('hist_k', 0),
                   head=ck['cfg'].get('head', 'free'))
    m.normalizer.add('gt_3d_track', FieldNorm(3).fit_isotropic_rms(ck['track_rms']))
    m.load_state_dict(ck['model'])
    return m.to(dev).eval()


SUBSETS = ('all', 'valid', 'moving', 'mv:object', 'mv:arm')


def subsets(batch, tau=TAU):
    """-> dict of [B, N] bool masks. `moving` selects patches whose GT actually
    moved, which `valid` does not: the table is valid and perfectly static."""
    gt = batch['gt_3d_track']
    valid = batch['track_valid_mask'].any(dim=-1)
    disp = gt[:, :, -1, :].norm(dim=-1)
    moving = valid & (disp > tau)
    own = batch['owner']
    return {'all': th.ones_like(valid),
            'valid': valid,
            'moving': moving,
            'mv:object': moving & (own == 1),
            'mv:arm': moving & (own >= 7)}


@th.no_grad()
def track_error(model, dl, dev, max_frames=MAX_FRAMES):
    """Mean per-step L2 in mm for the model and for a constant-zero prediction.

    Both are accumulated in the SAME pass over the SAME patches, so the
    comparison cannot drift through a differing subset or frame count. Also
    accumulates predicted vs GT displacement MAGNITUDE per owner body, which is
    how a shrinkage bias shows up -- it is invisible in the error alone, and on
    single frames the eye reads it backwards.
    """
    acc = {k: [0.0, 0.0, 0] for k in SUBSETS}      # pred, zero, n
    mag = {}
    seen = 0
    for bi, batch in enumerate(dl):
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        pred = model.predict(batch, noise_seed=2000 + bi)   # [B,N,H,3] metres
        gt = batch['gt_3d_track']
        e_pred = (pred - gt).norm(dim=-1)               # [B,N,H]
        e_zero = gt.norm(dim=-1)                        # constant-zero baseline
        sub = subsets(batch)
        for name, m in sub.items():
            mm = m.unsqueeze(-1).expand_as(e_pred)
            acc[name][0] += float(e_pred[mm].sum())
            acc[name][1] += float(e_zero[mm].sum())
            acc[name][2] += int(mm.sum())
        for name in ('moving', 'mv:object', 'mv:arm'):
            mv = sub[name]
            if mv.any():
                pn = pred[:, :, -1, :].norm(dim=-1)[mv]
                gn = gt[:, :, -1, :].norm(dim=-1)[mv]
                a = mag.setdefault(name, [0.0, 0.0, 0])
                a[0] += float(pn.sum()); a[1] += float(gn.sum())
                a[2] += int(mv.sum())
        seen += gt.shape[0]
        if seen >= max_frames:
            break
    out = {k: dict(pred=1000.0 * v[0] / max(v[2], 1),
                   zero=1000.0 * v[1] / max(v[2], 1), n=v[2])
           for k, v in acc.items()}
    return out, seen, mag


def main():
    assert CKPT, 'set GAP_CKPT'
    dev = th.device(DEVICE)
    ck = th.load(CKPT, map_location='cpu')
    model = build(ck, dev)
    if ck['trunk'] == 'dit':
        model.num_inference_steps = INFER_STEPS or 1
    paths, va_ids = ck['paths'], ck['val_ids']
    print('[eval] %s  trunk=%s action=%s epoch=%d%s'
          % (CKPT, ck['trunk'], ck['cfg'].get('action_mode', 'realized'),
             ck['epoch'],
             ('  ddim_steps=%d' % model.num_inference_steps
              if ck['trunk'] == 'dit' else '')))
    if 'phys' in ck:
        print('[eval] physics (one config): %s' % '  '.join(
            '%s=%.4g' % (k, ck['phys'][k]) for k in PHYS_NAMES))

    ds_va = GapTrackDataset(paths, episodes=va_ids,
                            hist_k=ck['cfg'].get('hist_k', 0))
    dl_va = th.utils.data.DataLoader(ds_va, batch_size=BATCH, shuffle=False,
                                     num_workers=4, collate_fn=collate)
    print('[eval] %d val episodes / %d frames (held-out pose+goal, SAME physics)'
          % (len(va_ids), len(ds_va)))

    res, n_fr, mag = track_error(model, dl_va, dev)
    print('\n=== held-out track error (mean per-step L2, mm; %d frames) ===' % n_fr)
    print('  %-9s %10s %10s %8s   %s' % ('subset', 'model', 'zero-pred',
                                          'ratio', 'n patch-steps'))
    for k in SUBSETS:
        r = res[k]
        print('  %-9s %9.2f  %9.2f  %7.2fx   %d'
              % (k, r['pred'], r['zero'], r['zero'] / max(r['pred'], 1e-9), r['n']))
    print('  NOTE: `valid` is GAP\'s "move" group -- a VALIDITY mask that '
          'includes the\n        static table, whose GT track is exactly 0. '
          '`moving` (>%.0f cm\n        displacement) is the row that says '
          'whether the model works.' % (100 * TAU))

    if mag:
        print('\n=== displacement magnitude (endpoint) ===')
        for name in ('moving', 'mv:object', 'mv:arm'):
            if name not in mag:
                continue
            p, g, n = mag[name]
            print('  %-9s mean |pred| %.4f m   mean |gt| %.4f m   ratio %.2f'
                  % (name, p / n, g / n, (p / n) / max(g / n, 1e-9)))
        print('  <1 means the model UNDERSHOOTS how far things move (MSE '
              'shrinkage).\n  Do not judge this from the overlay: on single '
              'frames the predicted\n  polyline often looks longer, which is '
              'the opposite of the aggregate.')


if __name__ == '__main__':
    main()
