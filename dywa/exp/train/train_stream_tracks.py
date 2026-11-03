#!/usr/bin/env python3
"""Train a track model on a STREAM of freshly collected shards.

Answers the question the offline pipeline cannot: the 12K-episode mass model
scores `mv:object` 11.18 mm on the held-out episodes against 6.39 mm on episodes
it trained on -- a **1.751x** gap, measured by `_probe_stream_baseline.py` under
the same probe this trainer uses (CLAUDE.md's 1.89x is a different measurement,
12 hand-picked successful episodes; do not mix them). Train error has been FLAT
at ~7-8 mm across every data scale while only val error falls, as N^-0.20. The
model can always do the task; it cannot generalise.

More unique data is the lever, and it is blocked by STORAGE, not compute: at
67 KB/frame the 4.3x more data we want is ~480 GB against ~440 GB free, and the
32x that would halve the error is ~3.5 TB.

Streaming turns that into a compute problem: `collect_stream_tracks.py` produces
shards, this trains on a FIFO window of the most recent ones, and the rest are
deleted. Bounded disk, unlimited unique frames.

NO DAgger, AND THAT IS NOT AN OMISSION
--------------------------------------
`train_rma.py` needs DAgger because the student ACTS: its own mistakes change the
states it will later encounter, so the data distribution depends on the learner.
A track model is a passive predictor -- it never acts -- so the teacher alone
fixes the distribution. Consequences, all simplifications: no action mixing, no
alpha anneal, and no ON-POLICY REQUIREMENT, which is what makes frame reuse from
a window statistically legitimate. What is wanted here is "streaming/infinite
data", not online RL. The shape borrowed from `train_rma.py:574-580` is only the
unbounded loop and step-count accounting in place of epochs.

WHY A DISK-BACKED WINDOW AND NOT AN IN-PROCESS RING
---------------------------------------------------
The frames go through the unmodified `FricTrackDataset` + `DataLoader(shuffle=
True)`, so the data path is byte-identical to the one that produced every
existing number (verified frame-by-frame by `_probe_stream_equivalence.py`).
Feeding the model straight from a collector's emit point instead would introduce
BATCH CORRELATION silently: frames arrive 32 at a time, one per env, all at
nearly the same episode phase, and the swept value is pinned per env, so such a
batch holds exactly one sample per physics bucket. Global shuffling destroys that
for free.

REUSE IS EMERGENT, AND MUST BE MEASURED
---------------------------------------
Nothing rate-limits either side. The window is always full after warmup, so each
frame is seen (consumption / production) times: measured offline that is
371 / 82 = ~4.5x, versus 20x for the 20-epoch offline run. But 371 f/s came from
a loader over 6 big files; here there are ~30 shard handles per worker, workers
respawn every block, and probe passes interleave -- so consumption will be LOWER
and reuse below 4.5. That is fine for the goal but changes the budget, and if the
collector ends up ahead it blocks on `STREAM_MAX_SHARDS` and stops producing
unique data. So `stream/reuse` is computed from counters and logged, never
assumed.

WHAT IS FROZEN, AND WHY EACH ONE WOULD BE A SILENT FAILURE
----------------------------------------------------------
* `track_rms` -- taken from the reference checkpoint, NEVER refit. It is one
  scalar that sets the units of every loss number, and `_rigid_tracks` reads it
  back live per forward. `fit_track_norm` needs `len(ds)` and 800 evenly spaced
  random accesses, which a stream cannot provide; refitting per window would make
  the loss curve incomparable to the 0.06131 it is supposed to beat.
* the eval corpus + val ids -- a fixed HDF5 set that the stream never touches,
  with the ids loaded from the reference ckpt so `eval_fric_track.sh` works on
  the output unmodified.
* the LR schedule is a pure function of `gstep` (warmup then constant, with an
  optional terminal cosine), so a resume needs no scheduler state and cannot land
  mid-schedule the way `_killed_token_ep9` did.

Knobs: STREAM_{DIR,EVAL_DATA,VAL_IDS_FROM,INIT_FROM,RESUME,WINDOW_SHARDS,
TOTAL_STEPS,BLOCK_STEPS,LR,WARMUP_STEPS,ANNEAL_STEPS,PROBE_FRAMES,
FULL_VAL_EVERY,BATCH,WORKERS,DINO_DIM,DEVICE,SAVE_DIR,SEED,KEEP_SHARDS,STATIC}
"""

import gc
import glob
import json
import math
import os
import re
import sys
import time

import h5py
import numpy as np
import torch as th
import wandb

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.gap.dataset import PHYS_NAMES, count_episodes
from models.fric.dataset import (FricTrackDataset, read_grid, episode_mu,
                                 bucketize, assert_phys_grid)
from models.fric.track_model import FricTrackModel

# Reuse the OFFLINE trainer's DINO wrapper and collate, and the OFFLINE
# evaluator's subset definitions. Both modules guard `main()` with
# `if __name__ == '__main__'`, so importing them is side-effect free beyond
# reading their own env vars. Importing rather than reimplementing is the point:
# a divergent copy of `subsets` or of the projection guard would make these
# numbers quietly incomparable to everything already measured.
from train_fric_track import ProjectedDino, collate, PINNED_ALL
from eval_fric_track import subsets

TRUNK = os.environ.get('STREAM_TRUNK', 'transformer')
PHYS_COND = os.environ.get('STREAM_PHYS_COND', 'film')
ACTION_MODE = os.environ.get('STREAM_ACTION_MODE', 'command')
OBJ_WEIGHT = float(os.environ.get('STREAM_OBJ_WEIGHT', 9.0))
D_MODEL = int(os.environ.get('STREAM_D_MODEL', 512))
DROPOUT = float(os.environ.get('STREAM_DROPOUT', 0.1))
WD = float(os.environ.get('STREAM_WD', 1e-6))
BATCH = int(os.environ.get('STREAM_BATCH', 16))
DINO_DIM = int(os.environ.get('STREAM_DINO_DIM', 64))
WORKERS = int(os.environ.get('STREAM_WORKERS', 8))
DEVICE = os.environ.get('STREAM_DEVICE', 'cuda:0')
SEED = int(os.environ.get('STREAM_SEED', 0))
# Doubles as the DINO projection seed, exactly as FRIC_SPLIT_SEED does offline;
# the guard below asserts it matches what the collector wrote.
SPLIT_SEED = int(os.environ.get('STREAM_SPLIT_SEED', 0))

CHANNEL = os.environ.get('STREAM_CHANNEL', 'mass')
STREAM_DIR = os.environ.get('STREAM_DIR',
                            '/home/user/DyWA/output/%s/stream' % CHANNEL)
EVAL_DATA = os.environ.get('STREAM_EVAL_DATA', '')
VAL_IDS_FROM = os.environ.get('STREAM_VAL_IDS_FROM', '')
INIT_FROM = os.environ.get('STREAM_INIT_FROM', '')
RESUME = os.environ.get('STREAM_RESUME', '0') == '1'
SAVE = os.environ.get('STREAM_SAVE_DIR',
                      '/home/user/DyWA/output/%s/ckpt' % CHANNEL)

WINDOW_SHARDS = int(os.environ.get('STREAM_WINDOW_SHARDS', 30))
MIN_SHARDS = int(os.environ.get('STREAM_MIN_SHARDS', 4))
TOTAL_STEPS = int(os.environ.get('STREAM_TOTAL_STEPS', 1680000))
BLOCK_STEPS = int(os.environ.get('STREAM_BLOCK_STEPS', 20000))
LR = float(os.environ.get('STREAM_LR', 0.0))       # 0 -> chosen by warm-start
WARMUP_STEPS = int(os.environ.get('STREAM_WARMUP_STEPS', 5000))
ANNEAL_STEPS = int(os.environ.get('STREAM_ANNEAL_STEPS', 0))
PROBE_FRAMES = int(os.environ.get('STREAM_PROBE_FRAMES', 12000))
FULL_VAL_EVERY = int(os.environ.get('STREAM_FULL_VAL_EVERY', 10))
KEEP_SHARDS = os.environ.get('STREAM_KEEP_SHARDS', '0') == '1'
POLL_S = float(os.environ.get('STREAM_POLL_S', 20.0))
# Static mode: train on a FIXED file list with no rotation. Used by verification
# step 3 to check this trainer against `train_fric_track.py` on the same data.
STATIC = os.environ.get('STREAM_STATIC', '')
RUN_TAG = os.environ.get('STREAM_TAG', '')

WANDB_PROJECT = os.environ.get('STREAM_WANDB_PROJECT', 'dywa-stream-track')
WANDB_MODE = os.environ.get('STREAM_WANDB_MODE', 'online')

SHARD_RE = re.compile(r'stream_(\d{5})\.hdf5$')


# ── shard window ──────────────────────────────────────────────────────────
class ShardWindow:
    """FIFO window over completed shards, as a ConcatDataset.

    One `FricTrackDataset` is built per shard when it is admitted and then KEPT.
    Index construction costs 13-18 us/frame, so rebuilding a 700k-frame window
    from scratch every block would burn ~10 s each time; per-shard caching makes
    a refresh cost only the newly admitted shard. Concatenating the cached
    datasets preserves global uniform sampling across the whole window, which is
    what `DataLoader(shuffle=True)` then needs to break the per-env correlation.

    Only shards with a `.done` marker are admitted: an unwritten HDF5 region
    reads back as 0, which is finite and passes any value check, so a partially
    written shard is precisely the input that trains fine and poisons the result.
    """

    def __init__(self, d, k, col, grid, expect_pinned, eval_paths, static=None):
        self.d, self.k, self.col = d, k, col
        self.grid, self.expect_pinned = grid, expect_pinned
        self.eval_paths = list(eval_paths)
        self.static = static
        self.resident = []                    # [(idx, path, dataset)] oldest first
        self.n_admitted = 0
        self.n_frames_admitted = 0
        self.n_evicted = 0

    def _ready(self):
        out = []
        for m in glob.glob(os.path.join(self.d, 'stream_*.done')):
            p = m[:-len('.done')] + '.hdf5'
            mt = SHARD_RE.search(p)
            if mt and os.path.exists(p):
                out.append((int(mt.group(1)), p))
        return sorted(out)

    def _build(self, path):
        """Admit one shard, asserting it can be pooled with the corpus.

        `read_grid` is the authoritative cross-file check (it refuses files whose
        grid, column or spacing disagree) and `assert_phys_grid` is what catches
        a pin that drifted in the collector's shell wrapper -- the one piece of
        that wrapper duplicated from `collect_fric_tracks.sh`. Both run per shard
        rather than once, because a mid-run collector restart with different env
        could otherwise slip a divergent shard into the window.
        """
        g = read_grid([path] + self.eval_paths)
        assert g['col'] == self.col
        assert_phys_grid([path], self.expect_pinned, self.col, self.grid)
        with h5py.File(path, 'r', locking=False) as f:
            w = int(f['dino_features'].shape[-1])
            s = int(f.attrs['dino_proj_seed']) if 'dino_proj_seed' in f.attrs \
                else None
        assert s is None or s == SPLIT_SEED, (
            '%s was projected with dino seed %d but STREAM_SPLIT_SEED=%d; the '
            'wrapper would use a different matrix on any raw file and the two '
            'sources would land in different subspaces -- a bad model, never an '
            'error' % (path, s, SPLIT_SEED))
        ds = FricTrackDataset([path], phys_col=self.col)
        if DINO_DIM != 1024:
            ds = ProjectedDino(ds, DINO_DIM, SPLIT_SEED)
        assert w == DINO_DIM or DINO_DIM in (0, 1024), (
            '%s stores %d-d features but STREAM_DINO_DIM=%d' % (path, w, DINO_DIM))
        return ds

    def refresh(self):
        """Admit every new completed shard, evict past the window size."""
        if self.static:
            if not self.resident:
                for i, p in enumerate(self.static):
                    self.resident.append((i, p, self._build(p)))
                    self.n_admitted += 1
                    self.n_frames_admitted += len(self.resident[-1][2])
            return 0, 0
        have = {i for i, _, _ in self.resident}
        added = 0
        for i, p in self._ready():
            if i in have:
                continue
            self.resident.append((i, p, self._build(p)))
            self.n_admitted += 1
            self.n_frames_admitted += len(self.resident[-1][2])
            added += 1
        self.resident.sort(key=lambda t: t[0])
        evicted = 0
        while len(self.resident) > self.k:
            i, p, _ = self.resident.pop(0)
            self.n_evicted += 1
            evicted += 1
            if KEEP_SHARDS:
                continue
            # Deletion is the only irreversible thing this trainer does, so the
            # name is re-checked here rather than trusted from the scan: only a
            # file literally called stream_NNNNN.hdf5, inside STREAM_DIR, can
            # ever be removed. That makes deleting a corpus file (tracks_*.hdf5,
            # 14-27 GB and not regenerable in under an hour) structurally
            # impossible rather than merely unlikely.
            assert SHARD_RE.search(os.path.basename(p)) and \
                os.path.dirname(os.path.abspath(p)) == os.path.abspath(self.d), \
                'refusing to delete %r: not a stream shard in %r' % (p, self.d)
            # Safe only because the caller dropped the DataLoader (and therefore
            # its worker processes, which hold h5py handles) before calling this.
            for q in (p, p[:-len('.hdf5')] + '.done'):
                if os.path.exists(q):
                    os.remove(q)
        return added, evicted

    def wait_for_min(self):
        while not self.static and len(self._ready()) < MIN_SHARDS:
            print('[stream] waiting for shards: %d/%d ready in %s'
                  % (len(self._ready()), MIN_SHARDS, self.d), flush=True)
            time.sleep(POLL_S)

    @property
    def dataset(self):
        return th.utils.data.ConcatDataset([d for _, _, d in self.resident])

    @property
    def oldest(self):
        """The oldest resident shard -- the only frames that have seen the FULL
        reuse. A probe drawn uniformly over the whole window would average over
        0..reuse exposures (a just-admitted shard has been seen ~0 times), which
        biases the window error upward and UNDERSTATES the gap being measured."""
        return self.resident[0] if self.resident else None

    @property
    def n_frames(self):
        return sum(len(d) for _, _, d in self.resident)


# ── probes ────────────────────────────────────────────────────────────────
@th.no_grad()
def probe(model, ds, dev, tag, max_frames=None, batch=None, workers=None):
    """Mean per-step L2 in mm on `mv:object`, plus the per-EPISODE distribution.

    Subset masks come from the imported `eval_fric_track.subsets`, so "moving"
    and "object" mean exactly what `eval_fric_track.sh` means by them. The
    aggregate formula (1000 * sum(err[mask]) / n_patch_steps) is the same one.

    Per-episode as well as aggregate because the goal is stated per trajectory:
    the baseline gap is reported both ways (1.751x aggregate / 1.738x
    per-episode), so both must be available to compare against.
    """
    model.eval()
    # `shuffle=False` and a fixed batch size are load-bearing, not incidental:
    # `predict`'s `noise_seed=2000+bi` is keyed to the BATCH INDEX (only matters
    # for trunk=dit, but it costs nothing to keep the DiT path reproducible),
    # and a deterministic order is what makes two blocks' numbers comparable.
    dl = th.utils.data.DataLoader(ds, batch_size=batch or BATCH, shuffle=False,
                                  num_workers=workers or max(2, WORKERS // 2),
                                  collate_fn=collate, pin_memory=True)
    acc = {k: [0.0, 0.0, 0] for k in ('mv:object', 'mv:arm', 'moving')}
    per_ep = {}
    seen = 0
    for bi, b in enumerate(dl):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        pred = model.predict(b, noise_seed=2000 + bi)
        gt = b['gt_3d_track']
        e_pred, e_zero = (pred - gt).norm(dim=-1), gt.norm(dim=-1)
        sub = subsets(b)
        for name in acc:
            mm = sub[name].unsqueeze(-1).expand_as(e_pred)
            acc[name][0] += float(e_pred[mm].sum())
            acc[name][1] += float(e_zero[mm].sum())
            acc[name][2] += int(mm.sum())
        ob = sub['mv:object']
        ep = b['episode'].reshape(-1).cpu().numpy()
        for j in range(gt.shape[0]):
            m = ob[j]
            if not m.any():
                continue
            a = per_ep.setdefault(int(ep[j]), [0.0, 0])
            a[0] += float(e_pred[j][m].sum())
            a[1] += int(m.sum()) * gt.shape[2]
        seen += gt.shape[0]
        if max_frames and seen >= max_frames:
            break
    out = {k: (1000.0 * v[0] / max(v[2], 1)) for k, v in acc.items()}
    out['zero:object'] = 1000.0 * acc['mv:object'][1] / max(acc['mv:object'][2], 1)
    # Same deterministic teardown as the training loop: a probe loader's
    # workers left to lazy GC can die mid-queue-write and dump a bodyless
    # `Traceback ... multiprocessing/queues.py _feed` into the log. Harmless,
    # but in a 20-hour log it is indistinguishable at a glance from a crash.
    del dl
    gc.collect()
    ep_mm = np.array([1000.0 * a[0] / max(a[1], 1) for a in per_ep.values()])
    out['n_frames'] = seen
    out['n_episodes'] = len(ep_mm)
    if len(ep_mm):
        out['ep_mean'] = float(ep_mm.mean())
        out['ep_median'] = float(np.median(ep_mm))
        out['ep_p90'] = float(np.percentile(ep_mm, 90))
    print('    [%-10s] mv:object %6.2f mm (zero %6.2f)  mv:arm %6.2f  '
          'per-ep mean %6.2f median %6.2f  n=%d frames / %d eps'
          % (tag, out['mv:object'], out['zero:object'], out['mv:arm'],
             out.get('ep_mean', float('nan')),
             out.get('ep_median', float('nan')), seen, len(ep_mm)),
          flush=True)
    return out


def pick_probe_episodes(paths, ids, target_frames, seed=0):
    """A FIXED random subset of episodes totalling ~`target_frames`.

    Random over episodes rather than "the first N frames": frame order follows
    episode order, so a prefix would score only the lowest-id episodes and drift
    with any change in file ordering. Seeded, and stored in the checkpoint.
    """
    lens = {}
    off = 0
    for p in paths:
        with h5py.File(p, 'r', locking=False) as f:
            ee = f['episode_ends'][:]
            st = np.concatenate([[0], ee[:-1]])
            for j in range(len(ee)):
                lens[off + j] = int(ee[j] - st[j])
        off += len(ee)
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(ids, np.int64))
    out, tot = [], 0
    for e in order:
        out.append(int(e))
        tot += lens.get(int(e), 0)
        if tot >= target_frames:
            break
    return np.sort(np.asarray(out, np.int64))


def lr_at(step):
    """Warmup, then constant, then an optional terminal cosine anneal.

    A pure function of the step, deliberately: `OneCycleLR` needs `total_steps`
    up front and raises past it, which is incompatible with "resume and keep
    going if it has not converged" -- and it makes a resume depend on scheduler
    state that the offline checkpoint never stored. `train_rma.py` itself runs a
    flat LR with no scheduler at all.
    """
    if step < WARMUP_STEPS:
        return LR * float(step + 1) / max(WARMUP_STEPS, 1)
    if ANNEAL_STEPS and step >= TOTAL_STEPS - ANNEAL_STEPS:
        p = min(1.0, (step - (TOTAL_STEPS - ANNEAL_STEPS)) / float(ANNEAL_STEPS))
        return LR * 0.5 * (1.0 + math.cos(math.pi * p))
    return LR


def infinite(dl):
    """Cycle a loader forever, respawning workers when it is exhausted."""
    while True:
        for b in dl:
            yield b


def main():
    global LR
    eval_paths = [p for p in EVAL_DATA.split(',') if p and os.path.exists(p)]
    assert eval_paths, ('STREAM_EVAL_DATA must name the frozen eval corpus; the '
                        'stream itself provides no held-out set')
    th.manual_seed(SEED)
    np.random.seed(SEED)
    dev = th.device(DEVICE)

    g = read_grid(eval_paths)
    col, grid = g['col'], g['grid']
    expect_pinned = {k: v for k, v in PINNED_ALL.items()
                     if k != PHYS_NAMES[col]}
    print('[phys] swept=%s (col %d, %s) %d values in [%.4f, %.4f]'
          % (g['channel'], col, g['spacing'], len(grid), g['lo'], g['hi']))

    # ── the frozen eval sets ──────────────────────────────────────────────
    assert VAL_IDS_FROM, (
        'STREAM_VAL_IDS_FROM must point at a reference checkpoint. Re-deriving '
        'the split here would score against different episodes than the runs '
        'this is compared to -- the exact failure FRIC_VAL_POOL caused offline '
        '(94/419 = 22% overlap).')
    ref = th.load(VAL_IDS_FROM, map_location='cpu')
    va_ids = np.sort(np.asarray([int(x) for x in ref['val_ids']], np.int64))
    n_ep_eval = count_episodes(eval_paths)
    assert va_ids.max() < n_ep_eval, (
        'reference val id %d exceeds the eval corpus (%d episodes)'
        % (va_ids.max(), n_ep_eval))
    assert [os.path.realpath(x) for x in ref['paths']] == \
           [os.path.realpath(x) for x in eval_paths[:len(ref['paths'])]], (
        'reference ckpt paths %s are not a prefix of STREAM_EVAL_DATA %s -- '
        'episode id k would mean a different episode in the two runs'
        % (ref['paths'], eval_paths))
    # Episodes the OFFLINE model trained on. Under a warm start these are NOT a
    # control: the initial weights memorised them, so the probe starts at the
    # memorised error and decays. It is a forgetting curve, and it cannot
    # separate memorisation residue from set difficulty. Only from scratch (no
    # STREAM_INIT_FROM) is it the clean "both sets unseen, so both should score
    # the same" control that shows the offline gap was memorisation.
    ot_pool = np.asarray([int(x) for x in ref.get('train_ids', [])], np.int64)
    ot_pool = ot_pool[ot_pool < n_ep_eval]
    ot_ids = np.sort(np.random.default_rng(0).permutation(ot_pool)[:len(va_ids)]) \
        if len(ot_pool) else np.asarray([], np.int64)
    print('[data] eval corpus %d episodes | val %d | oldtrain pool %d -> probe %d'
          % (n_ep_eval, len(va_ids), len(ot_pool), len(ot_ids)))
    # The two probe sets must be disjoint or the "gap" between them is partly a
    # comparison of a set with itself.
    assert not np.intersect1d(ot_ids, va_ids).size, (
        'oldtrain probe overlaps the val set on %d episodes'
        % np.intersect1d(ot_ids, va_ids).size)

    def mk(ids):
        ds = FricTrackDataset(eval_paths, phys_col=col, episodes=ids)
        return ProjectedDino(ds, DINO_DIM, SPLIT_SEED) if DINO_DIM != 1024 else ds

    probe_va_ids = pick_probe_episodes(eval_paths, va_ids, PROBE_FRAMES, 0)
    ds_va_full = mk(va_ids)
    ds_va_quick = mk(probe_va_ids)
    ds_ot = mk(pick_probe_episodes(eval_paths, ot_ids, PROBE_FRAMES, 1)) \
        if len(ot_ids) else None
    print('[data] frames: val_full %d  val_quick %d (%d eps)  oldtrain %d'
          % (len(ds_va_full), len(ds_va_quick), len(probe_va_ids),
             len(ds_ot) if ds_ot else 0))

    # ── model, with the normalizer FROZEN ─────────────────────────────────
    # `track_rms` must come from whichever checkpoint supplies the WEIGHTS, not
    # from the val-ids reference. Two reasons, and the second is a silent
    # failure: (1) the model's learned output scale is matched to the rms it
    # trained under, and (2) `FieldNorm.scale`/`offset` are persistent buffers,
    # so they live inside `state_dict` and `load_state_dict` OVERWRITES anything
    # set here -- writing a different value into `ck['track_rms']` would leave
    # every downstream reader rebuilding a normalizer that disagrees with the
    # weights, with nothing raising. The two checkpoints in play here really do
    # differ (0.031895 for the 2K val-ids reference vs 0.031571 for the 12K film
    # run) because each refit `fit_track_norm` on its own 800-frame draw.
    tag = '_dino%d' % DINO_DIM if DINO_DIM != 1024 else ''
    # STREAM_TAG distinguishes runs that share every other knob -- e.g. a second
    # round warm-started from the first. Without it the name collides,
    # `makedirs(exist_ok=True)` succeeds, and the new run OVERWRITES the previous
    # best.ckpt in place: the result it is meant to be compared against.
    run = '%s_%s%s%s_stream%s%s' % (
        TRUNK, PHYS_COND,
        '' if ACTION_MODE == 'realized_gain' else '_' + ACTION_MODE,
        tag, '' if SEED == 0 else '_s%d' % SEED,
        ('_' + RUN_TAG) if RUN_TAG else '')
    out_dir = os.path.join(SAVE, run)
    os.makedirs(out_dir, exist_ok=True)

    # Resolve the weight source FIRST, because it decides the rms.
    resume_path = os.path.join(out_dir, 'last.ckpt')
    src_ck = None
    if RESUME and os.path.exists(resume_path):
        src_ck, src_kind = th.load(resume_path, map_location='cpu'), 'resume'
    elif INIT_FROM:
        src_ck, src_kind = th.load(INIT_FROM, map_location='cpu'), 'init'
    else:
        src_kind = 'scratch'

    if src_ck is not None:
        rms_src = resume_path if src_kind == 'resume' else INIT_FROM
        rms = float(src_ck['track_rms'])
        d_rms = abs(rms - float(ref['track_rms']))
        if d_rms > 1e-9:
            print('[norm] NOTE: weight source rms %.8f differs from the '
                  'val-ids reference %.8f (%.2f%%). Using the WEIGHT source: '
                  'the reference only pins which episodes are val, and its rms '
                  'came from a different 800-frame draw.'
                  % (rms, float(ref['track_rms']),
                     100.0 * d_rms / float(ref['track_rms'])))
    else:
        rms_src, rms = VAL_IDS_FROM, float(ref['track_rms'])

    model = FricTrackModel(trunk=TRUNK, d_model=D_MODEL,
                           action_mode=ACTION_MODE, obj_weight=OBJ_WEIGHT,
                           phys_cond=PHYS_COND, dropout=DROPOUT,
                           dino_dim=max(DINO_DIM, 1),
                           phys_lo=g['lo'], phys_hi=g['hi'],
                           phys_log=(g['spacing'] == 'log')).to(dev)
    model.normalizer.add('gt_3d_track',
                         FieldNorm(3).fit_isotropic_rms(rms).to(dev))
    print('[norm] track_rms = %.6f FROZEN from %s (never refit: it sets the '
          'units of every loss number below)' % (rms, rms_src))

    gstep, block0, best = 0, 0, float('inf')
    opt = th.optim.AdamW(model.parameters(), lr=LR or 1e-4, weight_decay=WD)
    if src_kind == 'resume':
        model.load_state_dict(src_ck['model'])
        opt.load_state_dict(src_ck['opt'])
        gstep, block0 = int(src_ck['gstep']), int(src_ck['epoch']) + 1
        best = float(src_ck.get('best_val',
                                src_ck.get('val_loss', float('inf'))))
        if 'torch_rng' in src_ck:
            th.set_rng_state(src_ck['torch_rng'].cpu().to(th.uint8))
        if LR <= 0:
            LR = float(src_ck['cfg'].get('stream_lr', 3e-5))
        warm = ('RESUMED at step %d (block %d), best %.3f mm'
                % (gstep, block0, best))
    elif src_kind == 'init':
        # strict: a partial load would leave part of the net random, which is
        # exactly the failure `_assert_ckpt_covers_policy` exists to catch on
        # the TTT side.
        model.load_state_dict(src_ck['model'], strict=True)
        if LR <= 0:
            # 0.3x. The offline run ended on an annealed OneCycle at ~0 LR, so
            # restarting at its max_lr would undo it before it recovered.
            LR = 3e-5
        warm = ('WARM START from %s (its epoch %d, val_loss %.5f)'
                % (INIT_FROM, src_ck.get('epoch', -1),
                   src_ck.get('val_loss', float('nan'))))
    else:
        if LR <= 0:
            LR = 1e-4
        warm = 'from scratch'
    # Verify the buffer actually in the model, not the number we intended: this
    # is the one place the two could disagree without raising.
    eff = 1.0 / float(model.normalizer['gt_3d_track'].scale[0])
    assert abs(eff - rms) < 1e-6, (
        'normalizer buffer says rms %.8f but ck[track_rms] would be written as '
        '%.8f -- every downstream reader rebuilds from the latter' % (eff, rms))
    n_par = sum(p.numel() for p in model.parameters())
    print('[model] %s phys_cond=%s action=%s n_prefix=%d dino=%d %d params'
          % (TRUNK, PHYS_COND, ACTION_MODE, model.n_prefix, DINO_DIM, n_par))
    print('[init] %s | lr=%g warmup=%d anneal=%d total=%d'
          % (warm, LR, WARMUP_STEPS, ANNEAL_STEPS, TOTAL_STEPS))

    # ── the window ────────────────────────────────────────────────────────
    static = [p for p in STATIC.split(',') if p and os.path.exists(p)] or None
    if static:
        print('[stream] STATIC MODE: %d fixed files, no rotation (equivalence '
              'test against train_fric_track.py)' % len(static))
    # Guards BOTH modes: a static file that is also the eval corpus would train
    # on the val episodes. For the equivalence test use the offline files that
    # hold no val episodes (for mass: tracks_m64_*, whose ids are all >= 2100).
    for p in (static or []):
        assert os.path.realpath(p) not in {os.path.realpath(q) for q in eval_paths}, \
            ('static file %s is in the eval corpus -- that would train on val. '
             'Use offline files that contain no val episodes.' % p)
    win = ShardWindow(STREAM_DIR, WINDOW_SHARDS, col, grid, expect_pinned,
                      eval_paths, static=static)
    win.wait_for_min()
    win.refresh()
    print('[stream] window: %d shards, %d frames' % (len(win.resident), win.n_frames))

    wandb.init(project=WANDB_PROJECT, name=run, mode=WANDB_MODE, dir=out_dir,
               config={'trunk': TRUNK, 'phys_cond': PHYS_COND,
                       'action_mode': ACTION_MODE, 'obj_weight': OBJ_WEIGHT,
                       'd_model': D_MODEL, 'batch': BATCH, 'lr': LR,
                       'seed': SEED, 'split_seed': SPLIT_SEED,
                       'track_rms': rms, 'dropout': DROPOUT, 'wd': WD,
                       'dino_dim': DINO_DIM, 'phys_channel': g['channel'],
                       'phys_lo': g['lo'], 'phys_hi': g['hi'],
                       'phys_spacing': g['spacing'],
                       'window_shards': WINDOW_SHARDS,
                       'total_steps': TOTAL_STEPS, 'block_steps': BLOCK_STEPS,
                       'warmup_steps': WARMUP_STEPS,
                       'anneal_steps': ANNEAL_STEPS,
                       'init_from': INIT_FROM, 'val_ids_from': VAL_IDS_FROM,
                       'stream_dir': STREAM_DIR, 'static': bool(static),
                       'n_val_ep': len(va_ids), 'eval_paths': eval_paths})
    print('[run] %s' % out_dir)
    if WANDB_MODE != 'disabled':
        print('[wandb] %s' % wandb.run.get_url(), flush=True)

    t_run = time.time()
    consumed = 0
    block = block0
    gap0 = [None]          # this run's block-0 gap, the local reference
    while gstep < TOTAL_STEPS:
        # Drop the loader (and its workers, which hold h5py handles) BEFORE the
        # refresh, so evicted shard files can actually be deleted.
        dl = th.utils.data.DataLoader(win.dataset, batch_size=BATCH,
                                      shuffle=True, num_workers=WORKERS,
                                      collate_fn=collate, drop_last=True,
                                      pin_memory=True)
        it = infinite(dl)
        model.train()
        t0, tot, k = time.time(), 0.0, 0
        n_this = min(BLOCK_STEPS, TOTAL_STEPS - gstep)
        for _ in range(n_this):
            b = next(it)
            b = {kk: v.to(dev, non_blocking=True) for kk, v in b.items()}
            for pg in opt.param_groups:
                pg['lr'] = lr_at(gstep)
            loss, info = model.compute_loss(b)
            opt.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += info['loss']
            k += 1
            gstep += 1
        tr_loss = tot / max(k, 1)
        consumed += k * BATCH
        dt = time.time() - t0
        # Drop the loader AND force collection, so its worker processes are gone
        # before the probes run and before refresh() deletes evicted files. Left
        # to lazy GC, a worker torn down mid-read emits a bare
        # `Traceback (most recent call last):` with no body into the log --
        # harmless, but indistinguishable at a glance from a real failure in a
        # 20-hour log.
        del it, dl
        gc.collect()

        full = (block % FULL_VAL_EVERY == 0)
        print('[block %4d] step %d/%d  stream/loss %.5f  lr %.2e  %.0fs '
              '(%.0f frame/s)'
              % (block, gstep, TOTAL_STEPS, tr_loss, lr_at(gstep - 1), dt,
                 k * BATCH / max(dt, 1e-9)), flush=True)

        m_val = probe(model, ds_va_full if full else ds_va_quick, dev,
                      'val/full' if full else 'val', None if full else PROBE_FRAMES)
        old = win.oldest
        m_win = None
        if old is not None:
            # Re-pinned every block: the oldest shard changes as the window
            # rotates, and it is the ONLY place frames have seen the full reuse.
            m_win = probe(model, old[2], dev, 'window', PROBE_FRAMES)
        m_ot = probe(model, ds_ot, dev, 'forget' if INIT_FROM else 'oldtrain',
                     PROBE_FRAMES) if ds_ot is not None else None

        added, evicted = (0, 0) if static else win.refresh()
        prod = win.n_frames_admitted
        elapsed = time.time() - t_run
        log = {'block': block, 'gstep': gstep, 'stream/loss': tr_loss,
               'train/lr': lr_at(gstep - 1),
               'stream/frames_consumed': consumed,
               'stream/consume_fps': k * BATCH / max(dt, 1e-9),
               'window/shards': len(win.resident),
               'window/frames': win.n_frames,
               'window/added': added, 'window/evicted': evicted,
               # < 1 means no shard has yet lived a full window lifetime, so
               # gap/window_vs_val is not yet an overfitting measure.
               'window/turnovers': win.n_evicted / float(max(win.k, 1)),
               }
        # The quick probe (a fixed ~PROBE_FRAMES episode subset) and the full
        # 419-episode pass are DIFFERENT SETS, so they get DIFFERENT keys.
        # Sharing one key produced a sawtooth every FULL_VAL_EVERY blocks that
        # reads as training instability but is only two measurements alternating.
        # `val_full/mv_object` is the one comparable to eval_fric_track's 11.18.
        vp = 'val_full' if full else 'val'
        log.update({vp + '/mv_object': m_val['mv:object'],
                    vp + '/mv_arm': m_val['mv:arm'],
                    vp + '/ep_mean': m_val.get('ep_mean', float('nan')),
                    vp + '/ep_median': m_val.get('ep_median', float('nan')),
                    vp + '/ep_p90': m_val.get('ep_p90', float('nan'))})
        if not static:
            # Only meaningful when shards actually arrive. Reuse is emergent
            # (consumption / production) and is the number that says whether
            # the collector is keeping up -- see the module docstring.
            log.update({'stream/frames_produced': prod,
                        'stream/reuse': consumed / max(prod, 1),
                        'stream/produce_fps': prod / max(elapsed, 1e-9)})
        if m_win:
            log.update({'window/mv_object': m_win['mv:object'],
                        'window/ep_mean': m_win.get('ep_mean', float('nan')),
                        # The operational train/val gap. Offline baseline under
                        # this same probe is 1.751x (val 11.18 / seen 6.39,
                        # _probe_stream_baseline.py). Read the caveats below
                        # before treating any single value as a gap.
                        'gap/' + vp + '_vs_window':
                            m_val['mv:object'] / max(m_win['mv:object'], 1e-9),
                        'gap/ep_' + vp + '_vs_window':
                            m_val.get('ep_mean', float('nan'))
                            / max(m_win.get('ep_mean', float('nan')), 1e-9)})
            # Stable aliases. The per-block keys are suffixed with which val
            # measurement they used (val vs val_full), so a fixed-name reader --
            # like the printf below -- must go through these or it silently
            # prints nan on exactly the full-val blocks.
            log['gap/window_vs_val'] = log['gap/' + vp + '_vs_window']
            log['gap/ep_window_vs_val'] = log['gap/ep_' + vp + '_vs_window']
        if m_ot:
            key = 'probe/forgetting' if INIT_FROM else 'probe/oldtrain'
            log[key + '_mv_object'] = m_ot['mv:object']
            log[key + '_ep_mean'] = m_ot.get('ep_mean', float('nan'))
        wandb.log(log, step=block)
        if m_win:
            # Two things this is NOT comparable to. (a) The 1.89x in
            # CLAUDE.md: that came from 12 hand-picked successful episodes via
            # visualize_fric_conditioning.py, a different measurement. Use
            # _probe_stream_baseline.py with STREAM_SEEN_DATA for the
            # like-for-like offline figure. (b) This run's own EARLY blocks:
            # until the window has turned over once, its shards have been
            # sampled ~0 times, so `window` is effectively a SECOND held-out set
            # rather than seen data, and the ratio measures set difficulty, not
            # overfitting. gap0 is recorded for reference only -- read the trend
            # after `window/turnovers` >= 1.
            if gap0[0] is None:
                gap0[0] = log['gap/window_vs_val']
            print('    gap/window_vs_val = %.3fx aggregate, %.3fx per-episode '
                  '(this run at block 0: %.3fx)'
                  % (log['gap/window_vs_val'],
                     log.get('gap/ep_window_vs_val', float('nan')),
                     gap0[0]), flush=True)
            log['gap/vs_block0'] = log['gap/window_vs_val'] / max(gap0[0], 1e-9)
        if static:
            print('    [static mode] window %d frames -- reuse/production are '
                  'not meaningful here' % win.n_frames, flush=True)
        else:
            print('    reuse %.2fx  produced %d  consumed %d  window %d shards '
                  '/ %d frames  (+%d/-%d)'
                  % (log['stream/reuse'], prod, consumed, len(win.resident),
                     win.n_frames, added, evicted), flush=True)

        ck = {'model': model.state_dict(), 'trunk': TRUNK,
              # `eval_fric_track.py:213` hard-indexes `epoch`; a stream has none,
              # so the block index stands in for it.
              'epoch': block,
              'track_rms': rms, 'val_loss': m_val['mv:object'],
              'cfg': {'d_model': D_MODEL, 'action_mode': ACTION_MODE,
                      'obj_weight': OBJ_WEIGHT, 'phys_cond': PHYS_COND,
                      'phys_lo': g['lo'], 'phys_hi': g['hi'],
                      'phys_log': (g['spacing'] == 'log'),
                      'phys_channel': g['channel'], 'dropout': DROPOUT,
                      'dino_dim': DINO_DIM, 'wd': WD,
                      'split_seed': SPLIT_SEED, 'val_pool': 0,
                      'val_ids_from': VAL_IDS_FROM, 'phys_col': col,
                      'shuffle': False, 'success_only': False,
                      'heldout_mu': 0,
                      # stream-only extras; every reader uses .get for these
                      'stream_lr': LR, 'stream_dir': STREAM_DIR,
                      'init_from': INIT_FROM, 'total_steps': TOTAL_STEPS},
              # `paths` MUST be the frozen eval corpus, not the shard list:
              # eval/visualisation read `paths` + `val_ids` straight out of the
              # ckpt, and the shards are deleted long before then.
              'paths': eval_paths, 'val_ids': va_ids,
              'train_ids': np.asarray([], np.int64),
              # resume state
              'opt': opt.state_dict(), 'gstep': gstep, 'best_val': best,
              'probe_ids': {'val_quick': probe_va_ids, 'oldtrain': ot_ids},
              'stream_stats': {'produced': prod, 'consumed': consumed,
                               'shards_admitted': win.n_admitted,
                               'shards_evicted': win.n_evicted,
                               'elapsed_s': elapsed},
              'torch_rng': th.get_rng_state()}
        th.save(ck, os.path.join(out_dir, 'last.ckpt'))
        if full and m_val['mv:object'] < best:
            best = m_val['mv:object']
            ck['best_val'] = best
            th.save(ck, os.path.join(out_dir, 'best.ckpt'))
            print('    new best (full val): %.3f mm' % best, flush=True)
        block += 1

    print('[done] %d steps, best full-val mv:object %.3f mm -> %s'
          % (gstep, best, out_dir))
    wandb.finish()


if __name__ == '__main__':
    main()
