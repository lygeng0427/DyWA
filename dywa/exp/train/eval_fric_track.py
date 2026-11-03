#!/usr/bin/env python3
"""Held-out tracking error for a friction-conditioned track model, in mm.

Same metric as eval_gap_track.py -- mean per-step L2, with the subset breakdown
that matters here:

    all       every patch
    valid     track_valid_mask_gt (GAP's "move" group; INCLUDES the static
              table, whose GT track is exactly 0, so a model predicting all
              zeros scores well on it)
    moving    ||GT endpoint displacement|| > tau
    mv:object moving patches on body 1, the pushed object   <-- THE HEADLINE
    mv:arm    moving patches on the franka links

`mv:object` is the only row this experiment can be read from. The model is given
the arm's future EEF path, which makes the arm's own tracks nearly a kinematics
lookup; ~90% of moving patches are arm, so an aggregate is ~90% a measure of
something friction cannot possibly affect. That aggregate has inverted the
conclusion twice in this codebase.

TWO THINGS THIS ADDS over the gap evaluator, both specific to the friction A/B:

  1. **Per-friction-bucket error.** If conditioning works, the gain must be
     largest where friction is most extreme. A uniform gain across mu is a
     capacity or regularisation effect wearing the costume of a physics one.
  2. **mu-substitution.** Re-scores every frame with the friction input replaced
     by the grid mean and by a shuffled value. If neither changes the error, the
     trunk is IGNORING the conditioning -- and a "no difference" result between
     the conditioned and unconditioned runs would then be a plumbing finding,
     not a physics one. Rising loss under substitution is the only evidence that
     the input is actually used.

    FRIC_CKPT=output/fric/ckpt/transformer_both/best.ckpt python3 eval_fric_track.py
"""
import os
import sys

import numpy as np
import torch as th

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.fric.dataset import FricTrackDataset, read_grid, bucketize
from models.fric.track_model import FricTrackModel

CKPT = os.environ.get('FRIC_CKPT', '')
DEVICE = os.environ.get('FRIC_DEVICE', 'cuda:0')
BATCH = int(os.environ.get('FRIC_BATCH', 16))
MAX_FRAMES = int(os.environ.get('FRIC_EVAL_FRAMES', 100000))
TAU = float(os.environ.get('FRIC_TAU', 0.01))
# DDIM steps at eval (dit only). GAP.yaml uses 10; the chain measurably degrades
# its own first estimate on this data, so 1 is what represents the trunk.
INFER_STEPS = int(os.environ.get('FRIC_INFER_STEPS', 0))
NBUCKET = int(os.environ.get('FRIC_NBUCKET', 4))

SUBSETS = ('all', 'valid', 'moving', 'mv:object', 'mv:arm')


def collate(items):
    return {k: th.stack([it[k] for it in items]) for k in items[0]}


def build(ck, dev):
    c = ck['cfg']
    m = FricTrackModel(trunk=ck['trunk'], d_model=c['d_model'],
                       action_mode=c.get('action_mode', 'realized_gain'),
                       obj_weight=c.get('obj_weight', 1.0),
                       phys_cond=c.get('phys_cond', 'none'),
                       # No defaults: a friction-valued 0.1/1.5 fallback silently
                       # squashes a mass grid's conditioning input into a third of
                       # its range. Every trainer since this file existed writes
                       # these, so a KeyError here means a genuinely broken ckpt.
                       # Passed unconditionally -- an unconditioned ckpt still
                       # records them, and withholding them here only re-created
                       # the missing-value problem from the other direction.
                       phys_lo=c['phys_lo'], phys_hi=c['phys_hi'],
                       phys_log=c.get('phys_log', False),
                       # A ckpt trained with narrowed DINO carries a
                       # dino_proj sized to that width; building at 1024 fails
                       # the strict load outright (which is the desired
                       # behaviour -- silently rebuilding it would evaluate a
                       # partly-random model).
                       dino_dim=max(int(c.get('dino_dim', 1024)), 1),
                       dropout=c.get('dropout', 0.1))
    m.normalizer.add('gt_3d_track',
                     FieldNorm(3).fit_isotropic_rms(ck['track_rms']))
    # strict: a silent key mismatch would leave part of the net random and the
    # A/B would compare a trained model against a half-initialised one.
    m.load_state_dict(ck['model'], strict=True)
    return m.to(dev).eval()


class _ProjectedDino(th.utils.data.Dataset):
    """Reproduce training's DINO projection at eval time.

    MUST use the same seed as training (`split_seed`), or the model is scored
    against features drawn through a different random matrix -- a silent
    mismatch that looks like a terrible model rather than a bug.
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
        # else: the file already stores exactly k dims, written pre-projected by
        # collect_fric_tracks.py. Projecting again would apply the matrix twice
        # and silently destroy the features -- and with P shaped [1024, k] it
        # cannot even be done, so before this guard existed a pre-projected file
        # raised `mat1 and mat2 shapes cannot be multiplied (391x64 and
        # 1024x64)`. That made every `_dino64` checkpoint un-evaluatable once
        # the mass files were narrowed on disk. This mirrors the guard
        # train_fric_track.py's ProjectedDino has always had.
        return it


def subsets(batch, tau=TAU):
    gt = batch['gt_3d_track']
    valid = batch['track_valid_mask'].any(dim=-1)
    disp = gt[:, :, -1, :].norm(dim=-1)
    moving = valid & (disp > tau)
    own = batch['owner']
    return {'all': th.ones_like(valid), 'valid': valid, 'moving': moving,
            'mv:object': moving & (own == 1), 'mv:arm': moving & (own >= 7)}


@th.no_grad()
def track_error(model, dl, dev, grid, mu_mode='true', max_frames=MAX_FRAMES):
    """Mean per-step L2 in mm, plus the zero-prediction baseline, in one pass.

    `mu_mode` swaps the conditioning input: 'true' | 'mean' (grid mean, i.e. no
    information) | 'shuffle' (a wrong value). Both are scored on the SAME frames
    in the same order, so the comparison cannot drift through frame selection.
    """
    acc = {k: [0.0, 0.0, 0] for k in SUBSETS}
    per_mu = {}
    mag = {}
    seen = 0
    # The "no information" value is the CENTRE OF THE GRID, and where that sits
    # depends on how the grid was laid out. On a geometric mass grid the
    # arithmetic mean is pulled towards the heavy end (0.05..1.0 -> 0.33 vs a
    # geometric centre of 0.22), so substituting it would not be a neutral
    # replacement -- it would be a light-biased one, and the per-bucket U-shape
    # would be measured about the wrong centre.
    _geo = bool(getattr(model, 'phys_log', False))
    gmean = float(np.exp(np.mean(np.log(grid))) if _geo else np.mean(grid))
    rng = np.random.default_rng(0)
    for bi, batch in enumerate(dl):
        batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
        mu_true = batch['phys'].clone()
        if mu_mode == 'mean':
            batch['phys'] = th.full_like(mu_true, gmean)
        elif mu_mode == 'shuffle':
            batch['phys'] = th.as_tensor(
                grid[rng.integers(0, len(grid), mu_true.shape[0])],
                dtype=mu_true.dtype, device=mu_true.device).reshape(-1, 1)

        pred = model.predict(batch, noise_seed=2000 + bi)
        gt = batch['gt_3d_track']
        e_pred = (pred - gt).norm(dim=-1)
        e_zero = gt.norm(dim=-1)
        sub = subsets(batch)
        for name, m in sub.items():
            mm = m.unsqueeze(-1).expand_as(e_pred)
            acc[name][0] += float(e_pred[mm].sum())
            acc[name][1] += float(e_zero[mm].sum())
            acc[name][2] += int(mm.sum())
        # per-friction breakdown, on the headline subset only
        ob = sub['mv:object']
        if ob.any():
            b = bucketize(mu_true.reshape(-1).cpu().numpy(), grid)
            for j in range(gt.shape[0]):
                m = ob[j]
                if not m.any():
                    continue
                a = per_mu.setdefault(int(b[j]), [0.0, 0.0, 0])
                a[0] += float(e_pred[j][m].sum())
                a[1] += float(e_zero[j][m].sum())
                a[2] += int(m.sum()) * gt.shape[2]
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
    return out, seen, mag, per_mu


def main():
    assert CKPT, 'set FRIC_CKPT'
    dev = th.device(DEVICE)
    ck = th.load(CKPT, map_location='cpu')
    c = ck['cfg']
    model = build(ck, dev)
    if ck['trunk'] == 'dit':
        model.num_inference_steps = INFER_STEPS or 1
    paths, va_ids = ck['paths'], ck['val_ids']
    g = read_grid(paths)
    grid = g['grid']

    print('[eval] %s' % CKPT)
    print('[eval] trunk=%s phys_cond=%s action=%s epoch=%d%s%s%s'
          % (ck['trunk'], c.get('phys_cond'), c.get('action_mode'), ck['epoch'],
             '  SHUFFLED-LABEL CONTROL' if c.get('shuffle') else '',
             '  HELD-OUT-MU=%d' % c['heldout_mu'] if c.get('heldout_mu') else '',
             '  ddim=%d' % model.num_inference_steps if ck['trunk'] == 'dit' else ''))
    print('[eval] swept %s over [%.3f, %.3f], %d grid values'
          % (g['channel'], g['lo'], g['hi'], len(grid)))

    ds_va = FricTrackDataset(paths, phys_col=g['col'], episodes=va_ids)
    _dd = int(c.get('dino_dim', 1024))
    if _dd != 1024:
        # split_seed, matching train_fric_track.py's ProjectedDino call.
        ds_va = _ProjectedDino(ds_va, _dd, int(c.get('split_seed', 0)))
        print('[eval] DINO features -> %s'
              % ('ZEROED (geometry only)' if _dd == 0
                 else 'random-projected 1024 -> %d' % _dd))
    dl_va = th.utils.data.DataLoader(ds_va, batch_size=BATCH, shuffle=False,
                                     num_workers=4, collate_fn=collate)
    print('[eval] %d val episodes / %d frames' % (len(va_ids), len(ds_va)))

    res, n_fr, mag, per_mu = track_error(model, dl_va, dev, grid, 'true')
    print('\n=== held-out track error (mean per-step L2, mm; %d frames) ===' % n_fr)
    print('  %-9s %10s %10s %8s   %s'
          % ('subset', 'model', 'zero-pred', 'ratio', 'n patch-steps'))
    for k in SUBSETS:
        r = res[k]
        print('  %-9s %9.2f  %9.2f  %7.2fx   %d'
              % (k, r['pred'], r['zero'], r['zero'] / max(r['pred'], 1e-9), r['n']))
    print('  READ `mv:object`. ~90%% of moving patches are ARM, whose future is '
          'handed to\n  the model as input, so the aggregate rows mostly measure '
          'something friction\n  cannot affect.')

    if mag:
        print('\n=== displacement magnitude (endpoint) ===')
        for name in ('moving', 'mv:object', 'mv:arm'):
            if name in mag:
                p, gm, n = mag[name]
                print('  %-9s mean |pred| %.4f m  mean |gt| %.4f m  ratio %.2f'
                      % (name, p / n, gm / n, (p / n) / max(gm / n, 1e-9)))
        print('  <1 means the model UNDERSHOOTS how far things move (MSE '
              'shrinkage).')

    if per_mu:
        print('\n=== mv:object error by friction bucket ===')
        ks = sorted(per_mu)
        edges = np.array_split(np.asarray(ks), min(NBUCKET, len(ks)))
        print('  %-16s %10s %10s %8s   %s'
              % ('mu range', 'model', 'zero-pred', 'ratio', 'n'))
        for grp in edges:
            p = sum(per_mu[k][0] for k in grp)
            z = sum(per_mu[k][1] for k in grp)
            n = sum(per_mu[k][2] for k in grp)
            print('  [%.3f, %.3f]  %9.2f  %9.2f  %7.2fx   %d'
                  % (grid[grp[0]], grid[grp[-1]], 1000 * p / max(n, 1),
                     1000 * z / max(n, 1), z / max(p, 1e-9), n))
        print('  If conditioning works, the gain must be LARGEST at the extremes '
              'of mu.\n  A flat gain is a capacity effect, not a physics one.')

    if c.get('phys_cond', 'none') != 'none':
        print('\n=== mu-substitution: is the conditioning actually used? ===')
        base = res['mv:object']['pred']
        print('  %-24s %10s %10s' % ('conditioning input', 'mv:object', 'delta'))
        print('  %-24s %9.2f  %9s' % ('true mu', base, '-'))
        sub_mu = {}
        _lab_mean = ('grid geo-mean (no info)' if getattr(model, 'phys_log', False)
                     else 'grid mean (no info)')
        for mode, lab in (('mean', _lab_mean), ('shuffle', 'shuffled value')):
            r2, _, _, pm = track_error(model, dl_va, dev, grid, mode)
            e = r2['mv:object']['pred']
            sub_mu[mode] = pm
            print('  %-24s %9.2f  %+9.2f' % (lab, e, e - base))

        # Per-bucket, and this is the version to trust for the SHAPE question.
        # Comparing a conditioned model against a separately-trained
        # unconditioned one confounds the friction effect with init noise --
        # measured here as 0.05-0.23 mm per bucket, which is the same size as
        # the effect. Substitution instead compares ONE model against ITSELF on
        # the SAME frames, so there is no seed term at all: the only thing that
        # changed is the friction number handed in.
        if per_mu and sub_mu:
            print('\n  --- cost of removing mu, per friction bucket (same model,'
                  ' same frames) ---')
            ks = sorted(per_mu)
            groups = np.array_split(np.asarray(ks), min(NBUCKET, len(ks)))
            print('  %-16s %10s %10s %10s %10s'
                  % ('value range', 'true', 'grid centre', 'shuffled', 'n'))
            for grp in groups:
                n = sum(per_mu[k][2] for k in grp)
                t = 1000 * sum(per_mu[k][0] for k in grp) / max(n, 1)
                row = [t]
                for mode in ('mean', 'shuffle'):
                    pm = sub_mu[mode]
                    nn = sum(pm[k][2] for k in grp if k in pm)
                    row.append(1000 * sum(pm[k][0] for k in grp if k in pm)
                               / max(nn, 1))
                print('  [%.3f, %.3f]  %9.2f  %9.2f %+5.2f  %9.2f %+5.2f  %9d'
                      % (grid[grp[0]], grid[grp[-1]], row[0],
                         row[1], row[1] - row[0], row[2], row[2] - row[0], n))
            print('  A physical effect should cost MOST where the variable is '
                  'most extreme.\n  Flat across the range = the model uses it as '
                  'a generic nuisance code, not as physics.\n  Read `n` too: '
                  'equal-EPISODE buckets carry unequal FRAMES whenever the swept '
                  'variable\n  changes episode length, and the profile is '
                  'weighted by n.')
        print('  ~0 delta means the trunk IGNORES the friction input. Then a '
              'null A/B is a\n  statement about the plumbing, not about the '
              'physics -- check this BEFORE\n  concluding anything from the '
              'conditioned-vs-unconditioned comparison.')


if __name__ == '__main__':
    main()
