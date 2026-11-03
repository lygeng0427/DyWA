#!/usr/bin/env python3
"""Dataset for the friction-swept HDF5 written by collect_fric_tracks.py.

Same frames as `models.gap.GapTrackDataset`, plus the one thing that differs:
`phys`, the scalar the model may condition on. Everything else is inherited so
that a `phys_cond='none'` run and a `phys_cond='both'` run see byte-identical
batches apart from that key.

THE SPLIT IS THE EXPERIMENT'S WEAK POINT, so there are two of them.

`balanced_episode_split` holds out whole episodes drawn evenly from every
friction bucket. Val episodes therefore come from the SAME 32 frictions as
train, which means a friction scalar can act as a 32-way episode ID: a model
could win by memorising "bucket 17 behaves like this" without learning any
physics. That is what the shuffled-label control in the trainer is for.

`heldout_mu_split` removes that route entirely by holding out whole FRICTION
VALUES. A model that only memorised buckets scores at chance on it; one that
learned mu -> dynamics interpolates. Slower to satisfy but the stronger claim.

Balancing matters because friction is pinned per env for the whole run and the
collector keeps only successful episodes: if low-mu envs succeed less often, the
episode counts skew along the very axis under study. Balancing at split time
rather than by a collector quota keeps the raw file complete and makes the
balance reproducible from the file alone.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch as th

from models.gap.dataset import GapTrackDataset, PHYS_NAMES, count_episodes


class FricTrackDataset(GapTrackDataset):
    """One item = one frame. Adds `phys` [1]: the swept physics scalar."""

    def __init__(self, paths, phys_col: int, **kwargs):
        super().__init__(paths, **kwargs)
        self.phys_col = int(phys_col)

    def __getitem__(self, i):
        out = super().__getitem__(i)
        _, fi, t, _, _ = self.index[i]
        out['phys'] = th.tensor(
            [float(self._f(fi)['phys_params'][t, self.phys_col])],
            dtype=th.float32)
        return out


# ── file-level metadata ────────────────────────────────────────────────────
def read_grid(paths) -> Dict:
    """The friction grid the collector baked, read from the file attrs.

    Raises rather than falling back to dataset statistics: a run that silently
    normalised mu differently from the run it is compared against would produce
    a difference that looks like the experimental effect.
    """
    if isinstance(paths, str):
        paths = [paths]
    got = None
    for p in paths:
        with h5py.File(p, 'r', locking=False) as f:
            missing = [k for k in ('fric_grid', 'fric_col', 'fric_lo', 'fric_hi')
                       if k not in f.attrs]
            assert not missing, (
                '%s lacks %s -- it was not written by collect_fric_tracks.py'
                % (p, missing))
            cur = dict(grid=np.asarray(f.attrs['fric_grid'], np.float64),
                       col=int(f.attrs['fric_col']),
                       lo=float(f.attrs['fric_lo']),
                       hi=float(f.attrs['fric_hi']),
                       channel=str(f.attrs.get('fric_channel', '?')),
                       # Files written before the mass channel existed carry no
                       # spacing attr and were all linear grids.
                       spacing=str(f.attrs.get('fric_spacing', 'linear')))
        if got is None:
            got = cur
        else:
            assert (cur['col'] == got['col']
                    and cur['spacing'] == got['spacing']
                    and np.allclose(cur['grid'], got['grid'])), (
                'files disagree on the swept grid; they cannot be pooled')
    return got


def episode_mu(paths, col: int) -> np.ndarray:
    """The swept value for every episode, in the dataset's episode order."""
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        with h5py.File(p, 'r', locking=False) as f:
            ends = f['episode_ends'][:]
            starts = np.concatenate([[0], ends[:-1]])
            pp = f['phys_params']
            out.extend(float(pp[int(s), col]) for s in starts)
    return np.asarray(out, np.float64)


def episode_success(paths) -> np.ndarray:
    """Per-episode task outcome, in the dataset's episode order.

    Only meaningful on a file collected with GAP_SUCCESS_ONLY=0; a success-only
    file returns all True, and filtering on it is then a no-op rather than a
    silent error.
    """
    if isinstance(paths, str):
        paths = [paths]
    out = []
    for p in paths:
        with h5py.File(p, 'r', locking=False) as f:
            out.append(np.asarray(f['episode_success'][:], bool))
    return np.concatenate(out)


def bucketize(mu: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Snap each episode's mu to its grid index.

    By nearest value, not by equality: the label and the attr are both float32
    but reach here through different casts, so `==` would test the rounding path
    rather than the data.
    """
    d = np.abs(mu[:, None] - grid[None, :])
    idx = d.argmin(1)
    err = d.min(1).max()
    assert err < 1e-4, (
        'episode friction values are not on the recorded grid (max err %.3e)'
        % err)
    return idx


# ── splits ─────────────────────────────────────────────────────────────────
def balanced_episode_split(paths, col, grid, val_ratio=0.2, seed=0,
                           per_bucket: Optional[int] = None,
                           keep: Optional[np.ndarray] = None,
                           trim: bool = True,
                           val_pool: Optional[int] = None):
    """Per-bucket train/val split, optionally trimmed to an equal episode count.

    Splitting WITHIN each bucket, rather than globally, guarantees every value
    of the swept variable appears on BOTH sides. A global permutation would
    leave some buckets val-only by chance, and the unconditioned baseline would
    then be penalised for values it never trained on -- inflating the apparent
    benefit of conditioning. That part is unconditional.

    `trim` is the part that is NOT always right, and the default is wrong for at
    least one real dataset here. Trimming equalises EPISODES per bucket. The loss
    is per FRAME. Those coincide only when episode length does not vary with the
    swept variable.

    Measured on the mass dataset (all outcomes kept, so every env contributes the
    same number of simulation steps): raw frames per bucket span 1.06x while raw
    episodes span 2.28x, because heavy objects produce fewer, longer episodes.
    Trimming to the smallest bucket turns a 1.06x frame imbalance into 2.62x --
    it CREATES the skew it is meant to remove -- and discards 43% of the frames.

    So: trim when the raw FRAME counts are skewed (the friction dataset, where
    success-only collection dropped whole failed episodes), and do not when they
    are already flat. `_probe_fric_census.py` prints both; check it, do not guess.
    """
    mu = episode_mu(paths, col)
    b = bucketize(mu, grid)
    rng = np.random.default_rng(seed)
    # `keep` restricts the CANDIDATE POOL, before balancing rather than after.
    # Filtering afterwards would re-skew the buckets along the swept axis -- the
    # exact imbalance the balancing exists to remove.
    b = np.where(keep, b, -1) if keep is not None else b
    counts = np.bincount(b[b >= 0], minlength=len(grid))
    # `val_pool` restricts which episodes may become VAL: only ids < val_pool.
    # Files are concatenated in `paths` order, so pointing it at the original
    # episode count makes appended files enlarge TRAIN only, leaving the val set
    # bit-identical. Without it, adding data repermutes the split and a
    # data-scaling comparison is measured against different episodes -- the exact
    # trap that GAP_VAL_POOL exists for on the gap side.
    vp = int(val_pool) if val_pool else None
    nz = counts[counts > 0]
    n_keep = (int(per_bucket) if per_bucket
              else (int(nz.min()) if trim else int(counts.max())))
    # With one episode per bucket every bucket goes entirely to val and train
    # comes back EMPTY -- the DataLoader then yields nothing and the run reports
    # a loss of 0.0 rather than failing. Refuse instead.
    assert int(nz.min()) >= 2, (
        'only %d episode(s) in the smallest bucket: a per-bucket split needs '
        '>=2. Collect more episodes, or pass per_bucket.' % int(nz.min()))
    tr, va = [], []
    for k in range(len(grid)):
        ids = np.nonzero(b == k)[0]
        if len(ids) == 0:
            continue
        ids = rng.permutation(ids)
        if trim or per_bucket:
            ids = ids[:n_keep]
        if vp is None:
            n_val = max(1, int(round(len(ids) * val_ratio)))
            va.extend(ids[:n_val].tolist())
            tr.extend(ids[n_val:].tolist())
        else:
            elig = ids[ids < vp]
            rest = ids[ids >= vp]
            # val_ratio is applied to the ELIGIBLE pool, so the val set matches
            # what a run over only the original files would have drawn.
            n_val = max(1, int(round(len(elig) * val_ratio)))
            va.extend(elig[:n_val].tolist())
            tr.extend(elig[n_val:].tolist())
            tr.extend(rest.tolist())
    tr = np.sort(np.asarray(tr, np.int64))
    va = np.sort(np.asarray(va, np.int64))
    assert len(tr) and len(va), 'balanced split produced an empty side'
    return tr, va, len(mu), counts, n_keep


def heldout_mu_split(paths, col, grid, n_holdout=8, seed=0,
                     keep: Optional[np.ndarray] = None):
    """Hold out whole FRICTION VALUES, not episodes.

    Interior values only: the endpoints of the grid would make this an
    extrapolation test as well as a generalisation test, and a model failing to
    extrapolate past the training range says nothing about whether it learned
    mu -> dynamics in between.
    """
    mu = episode_mu(paths, col)
    b = bucketize(mu, grid)
    interior = np.arange(1, len(grid) - 1)
    rng = np.random.default_rng(seed)
    hold = np.sort(rng.permutation(interior)[:n_holdout])
    ok = np.ones(len(b), bool) if keep is None else np.asarray(keep, bool)
    va = np.nonzero(np.isin(b, hold) & ok)[0]
    tr = np.nonzero(~np.isin(b, hold) & ok)[0]
    return np.sort(tr), np.sort(va), len(mu), grid[hold]


def assert_phys_grid(paths, expect_pinned: Dict[str, float],
                     free_col: int, grid: np.ndarray, tol: float = 1e-5):
    """Pinned columns constant AND correct; the free column on the grid.

    Two failure modes, not one. Reset-time domain randomisation has been
    silently inert on this GPU pipeline before, and the previous varying-friction
    collection shipped hand_friction=1.5 on every episode while the script asked
    for 1.25. "All episodes agree" is necessary but not sufficient --
    constant-but-WRONG passes a uniqueness check and is exactly what has bitten
    this project.
    """
    if isinstance(paths, str):
        paths = [paths]
    rows = []
    for p in paths:
        with h5py.File(p, 'r', locking=False) as f:
            ends = f['episode_ends'][:]
            starts = np.concatenate([[0], ends[:-1]])
            pp = f['phys_params']
            rows.extend(pp[int(s)] for s in starts)
    rows = np.asarray(rows, np.float64)

    bad = []
    for c, name in enumerate(PHYS_NAMES):
        if c == free_col:
            continue
        u = np.unique(np.round(rows[:, c], 6))
        if len(u) != 1:
            bad.append('%s varies: %d unique in [%.4f, %.4f]'
                       % (name, len(u), u.min(), u.max()))
        elif name in expect_pinned and abs(u[0] - expect_pinned[name]) > tol:
            bad.append('%s is constant at %.6f but %.6f was requested'
                       % (name, u[0], expect_pinned[name]))
    d = np.abs(rows[:, free_col][:, None] - grid[None, :]).min(1).max()
    if d > 1e-4:
        bad.append('%s is off the recorded grid by up to %.3e'
                   % (PHYS_NAMES[free_col], d))
    assert not bad, ('phys_params check failed over %d episodes:\n  %s'
                     % (len(rows), '\n  '.join(bad)))
    n_hit = len(np.unique(np.abs(rows[:, free_col][:, None]
                                 - grid[None, :]).argmin(1)))
    return {n: (float(rows[0, c]) if c != free_col else None)
            for c, n in enumerate(PHYS_NAMES)}, len(rows), n_hit
