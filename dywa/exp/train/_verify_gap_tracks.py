#!/usr/bin/env python3
"""Independently check that generated GAP tracks are actually right.

"It ran" is not evidence. Three checks, all from stored data only -- none of
them re-uses the generator's own math:

  1. STATIC BODIES. The table never moves, so every track on a table patch
     must be ~0. A nonzero table track means the owner->body mapping or the
     pose lookup is wrong.

  2. MOVING BODIES. Arm links move a lot under a pushing policy; if their
     tracks are ~0 too, the replay is not doing anything.

  3. CROSS-FRAME AGREEMENT (the real test). A track says "this surface point
     will be HERE in h steps". Frame t+h is also in the file, with its own
     independently unprojected anchors. So for each predicted point, measure
     the distance to the nearest anchor of the SAME body at t+h, and compare
     against the same measure with the track zeroed out ("assume nothing
     moves"). For a moving body, tracked must beat untracked decisively; if it
     does not, the tracks carry no real motion.

Usage:
    python3 _verify_gap_tracks.py <tracks.hdf5> [--h 15] [--max-frames 40]
"""
import argparse
import os
import sys

import h5py
import numpy as np

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')


def nn_dist(pred, ref):
    """median nearest-neighbour distance from pred (P,3) to ref (R,3)."""
    if len(pred) == 0 or len(ref) == 0:
        return np.nan
    d = np.linalg.norm(pred[:, None, :] - ref[None, :, :], axis=-1)
    return float(np.median(d.min(axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--h', type=int, default=None, help='horizon step to test')
    ap.add_argument('--max-frames', type=int, default=40)
    args = ap.parse_args()

    f = h5py.File(args.path, 'r', locking=False)
    print(f'\nkeys: {sorted(f.keys())}')
    for k in sorted(f.keys()):
        print(f'  {k:34s} {f[k].shape} {f[k].dtype}')
    ends = f['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    H = f['gt_3d_track'].shape[2]
    h = args.h if args.h is not None else H - 1
    print(f'\nepisodes={len(ends)}  frames={ends[-1]}  horizon={H}  testing h={h}')

    track = f['gt_3d_track']
    mask = f['track_valid_mask_gt']
    anch = f['image_patch_centre_3d_position']
    owner = f['patch_owner_body']

    # ── 1/2. per-body displacement magnitude ────────────────────────────────
    n = min(args.max_frames, int(ends[-1]))
    mags = {}
    for i in range(n):
        o = owner[i]
        m = mask[i][:, h]
        d = np.linalg.norm(track[i][:, h, :], axis=-1)
        for b in np.unique(o[m]):
            mags.setdefault(int(b), []).append(d[m & (o == b)])

    print('\n--- per-body track magnitude at h={} (metres) ---'.format(h))
    print(f'  {"body":>5} {"patches":>8} {"mean":>9} {"p95":>9} {"max":>9}')
    for b in sorted(mags):
        v = np.concatenate(mags[b])
        print(f'  {b:>5} {len(v):>8} {v.mean():>9.5f} '
              f'{np.percentile(v, 95):>9.5f} {v.max():>9.5f}')

    static = [b for b in mags if b == 0]          # body 0 = table
    ok_static = True
    for b in static:
        v = np.concatenate(mags[b])
        if v.max() > 5e-3:
            print(f'  !! body {b} (table) moves up to {v.max():.4f} m -- '
                  f'owner mapping or pose lookup is wrong')
            ok_static = False
    if ok_static and static:
        print('  [ok] table tracks are ~0 as expected')

    moving = [b for b in mags if b >= 2]          # arm links
    if moving:
        mx = max(np.concatenate(mags[b]).max() for b in moving)
        print(f'  [{"ok" if mx > 1e-3 else "!!"}] max arm-link displacement '
              f'{mx:.5f} m')

    # ── 3. cross-frame agreement ────────────────────────────────────────────
    print(f'\n--- cross-frame agreement at h={h} '
          f'(median NN distance to frame t+{h+1} anchors) ---')
    print(f'  {"body":>5} {"n":>5} {"tracked":>10} {"untracked":>10} {"ratio":>8}')
    agg = {}
    for ep in range(len(ends)):
        s, e = int(starts[ep]), int(ends[ep])
        for t in range(s, min(e - (h + 1), s + args.max_frames)):
            tt = t + h + 1
            if tt >= e:
                break
            a_t, o_t, m_t = anch[t], owner[t], mask[t][:, h]
            tr = track[t][:, h, :]
            a_f, o_f = anch[tt], owner[tt]
            for b in np.unique(o_t[m_t]):
                sel = m_t & (o_t == b)
                ref = a_f[o_f == b]
                if sel.sum() < 5 or len(ref) < 5:
                    continue
                d_tr = nn_dist(a_t[sel] + tr[sel], ref)
                d_un = nn_dist(a_t[sel], ref)
                if np.isnan(d_tr) or np.isnan(d_un):
                    continue
                agg.setdefault(int(b), []).append((d_tr, d_un))

    # A ratio computed from a handful of frames swings wildly; only bodies with
    # a real sample get to fail the run. Low-n rows are printed but not judged.
    MIN_N = 20
    verdict_ok = True
    for b in sorted(agg):
        v = np.array(agg[b])
        tr_m, un_m = float(np.median(v[:, 0])), float(np.median(v[:, 1]))
        ratio = un_m / max(tr_m, 1e-9)
        flag = ''
        if b >= 1 and un_m > 5e-3:      # only meaningful if the body moved
            if len(v) < MIN_N:
                flag = f'  (n<{MIN_N}, not judged)'
            elif ratio < 1.5:
                flag = '  <-- tracks add nothing'
                verdict_ok = False
        print(f'  {b:>5} {len(v):>5} {tr_m:>10.5f} {un_m:>10.5f} '
              f'{ratio:>8.2f}x{flag}')

    print('\n' + '=' * 62)
    if verdict_ok and ok_static:
        print('VERDICT: tracks look correct '
              '(static bodies still, moving bodies predicted better than '
              'assuming no motion)')
    else:
        print('VERDICT: PROBLEM -- see flags above')
    print('=' * 62 + '\n')
    return 0 if (verdict_ok and ok_static) else 1


if __name__ == '__main__':
    sys.exit(main())
