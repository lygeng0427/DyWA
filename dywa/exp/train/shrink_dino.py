#!/usr/bin/env python3
"""Narrow a track dataset's `dino_features` from 1024 -> k in place of a copy.

Why this exists: `dino_features` is 72% of a 1024-d track file (782 of 1091
KB/frame). Training never sees the raw 1024 dims anyway -- `ProjectedDino` in
train_fric_track.py applies a fixed random projection at load time -- so storing
the wide version buys nothing and costs ~90 GB per 1250-episode file.

The projection MUST be bit-compatible with the one used by:
  * collect_fric_tracks.py `_install_dino_projection`  (GAP_DINO_PROJ_SEED)
  * train_fric_track.py    `ProjectedDino`             (FRIC_SPLIT_SEED)
both of which build `randn(1024,k, manual_seed(seed)) / sqrt(1024)`. Mixing two
different P across files in one FRIC_DATA list would put the features in two
incompatible spaces and would NOT raise -- so the output is stamped with
`dino_dim`/`dino_proj_seed`, which is what the trainer's seed guard reads.

Everything else is copied verbatim, including `image`: the report's overlay
videos are rendered from those frames and only m_a/m_b carry them. Episode
bookkeeping (`episode_ends`/`_success`/`_length`) is copied unchanged so that
episode INDICES are preserved -- FRIC_VAL_IDS_FROM reuses ids across files, and
a reordering here would silently move the val split.

Numerical note: the source features were already rounded to float16 on write, so
this projects f16 -> f32 -> f16 and carries one extra rounding relative to a file
collected with GAP_DINO_PROJ=64. Over a 1024-term inner product that is far below
the f16 quantum; it is not bit-identical to a fresh collect and does not need to be.

  python3 shrink_dino.py IN.hdf5 OUT.hdf5 [--k 64] [--seed 0] [--limit N]
"""
import argparse
import os
import sys
import time

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=256)
    ap.add_argument('--limit', type=int, default=0,
                    help='frames to convert; 0 = all. For a cheap self-test.')
    a = ap.parse_args()
    if os.path.exists(a.dst):
        sys.exit('refusing to overwrite existing %s' % a.dst)

    import torch as th
    gen = th.Generator().manual_seed(a.seed)
    P = th.randn(1024, a.k, generator=gen) / np.sqrt(1024.0)
    P_np = P.numpy().astype(np.float32)

    t0 = time.time()
    with h5py.File(a.src, 'r') as f, h5py.File(a.dst, 'w') as o:
        src = f['dino_features']
        assert src.shape[-1] == 1024, 'source is already %d-d' % src.shape[-1]
        N = src.shape[0] if not a.limit else min(a.limit, src.shape[0])

        for key in f.attrs:
            o.attrs[key] = f.attrs[key]
        o.attrs['dino_dim'] = a.k
        o.attrs['dino_proj_seed'] = a.seed

        # Verbatim copies first, so a crash mid-projection leaves an obviously
        # incomplete file rather than one missing only its labels.
        for key in sorted(f.keys()):
            if key == 'dino_features':
                continue
            d = f[key]
            n = min(N, d.shape[0]) if d.shape[0] == src.shape[0] else d.shape[0]
            # Slice-copy, never `data=d[:n]`: that materialises the whole
            # dataset, and `image` alone is 28 GB at 130k frames.
            od = o.create_dataset(key, shape=(n,) + d.shape[1:], dtype=d.dtype,
                                  compression=d.compression, chunks=d.chunks)
            for i in range(0, n, a.chunk):
                j = min(i + a.chunk, n)
                od[i:j] = d[i:j]
            print('  copied %-32s %s' % (key, od.shape), flush=True)

        out = o.create_dataset(
            'dino_features', shape=(N,) + src.shape[1:-1] + (a.k,),
            dtype=src.dtype, compression=src.compression,
            chunks=(1,) + src.chunks[1:-1] + (a.k,))
        for i in range(0, N, a.chunk):
            j = min(i + a.chunk, N)
            blk = np.asarray(src[i:j], dtype=np.float32)
            flat = blk.reshape(-1, 1024) @ P_np
            out[i:j] = flat.reshape(blk.shape[:-1] + (a.k,)).astype(src.dtype)
            if (i // a.chunk) % 40 == 0:
                el = time.time() - t0
                print('  dino %7d/%d  %5.1f%%  %.0fs elapsed, ~%.0fs left'
                      % (j, N, 100.0 * j / N, el, el * (N - j) / max(j, 1)),
                      flush=True)

        o.attrs['conversion_complete'] = True
        o.attrs['conversion_frames'] = int(N)

    sz = lambda p: os.path.getsize(p) / 2 ** 30
    print('[done] %s %.1f GB -> %s %.1f GB  (%.2fx)  in %.0fs'
          % (a.src, sz(a.src), a.dst, sz(a.dst), sz(a.src) / sz(a.dst),
             time.time() - t0))


if __name__ == '__main__':
    main()
