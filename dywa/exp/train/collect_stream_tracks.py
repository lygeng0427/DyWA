#!/usr/bin/env python3
"""Collect track shards CONTINUOUSLY, rotating output files, for streaming training.

This is a thin driver around `collect_fric_tracks.py` (itself a thin driver around
`collect_gap_tracks.py`). Neither is edited. All this file does is:

  1. translate the channel-neutral `STREAM_*` knobs into the legacy
     `FRIC_*` / `GAP_*` ones, BEFORE importing anything that reads them,
  2. swap `EpisodeWriter` for one that closes the current file every
     `STREAM_SHARD_EPISODES` episodes and opens the next,
  3. apply backpressure so a dead trainer cannot fill the disk.

WHY SHARDS RATHER THAN A SHARED-MEMORY RING
-------------------------------------------
The trainer consumes the shards through the unmodified `FricTrackDataset` +
`DataLoader(shuffle=True)`, so the data path stays byte-identical to the offline
one that produced every existing number. The alternative -- handing frames
straight from the collector's emit point to the trainer -- gets BATCH CORRELATION
for free and silently: frames arrive 32 at a time, one per env, all at nearly the
same episode phase, and the swept value is pinned per env for the whole run, so a
batch drawn that way holds exactly one sample per physics bucket. Offline
shuffling destroys that structure for nothing; reproducing it by hand is where a
plausible-looking wrong result would come from.

CHANNEL-NEUTRAL ON PURPOSE
--------------------------
`STREAM_CHANNEL` is passed through to `FRIC_CHANNEL`, so mass / table friction /
object friction (and any later channel the underlying code grows) all work here
with no branch in this file. Range, spacing and the success-only default are the
underlying code's per-channel defaults; nothing about a specific channel is
hardcoded below.

`num_env` IS THE PHYSICS GRID
-----------------------------
`collect_fric_tracks._bake_channel` builds the grid as
`geomspace|linspace(LO, HI, env.num_env)`, so raising `STREAM_NUM_ENV` for
throughput would silently change the swept axis, and `read_grid` would then
refuse to pool the shards with the reference corpus. The precheck below catches
that in seconds instead of at train time. Raising it properly needs a TILED grid
(`np.tile(base, n_env // G)`) so E envs still cover exactly G values -- there is
no clean seam for that in `_bake_channel`, which computes the grid itself, so it
is deliberately NOT done here. Do it as its own change, with its own
verification, and only after the trainer is measured to be starved.

Knobs (everything else is inherited from GAP_*/FRIC_*):
    STREAM_CHANNEL          mass | table | object          (default mass)
    STREAM_DIR              shard output directory
    STREAM_SHARD_EPISODES   episodes per shard             (default 200)
    STREAM_MAX_SHARDS       backpressure threshold         (default 45)
    STREAM_NUM_ENV          must equal the reference grid size
    STREAM_DINO_PROJ        stored DINO width              (default 64)
    STREAM_EVAL_DATA        comma list; the reference corpus to check against
"""

import glob
import json
import os
import re
import time

import numpy as np

# ── STREAM_* -> legacy FRIC_*/GAP_* ───────────────────────────────────────
# This block MUST precede the imports below. `collect_gap_tracks.py:82-120` and
# `collect_fric_tracks.py:84-122` read every knob at MODULE IMPORT time, so a
# translation done after the import is silently ignored.
CHANNEL = os.environ.get('STREAM_CHANNEL', 'mass')
STREAM_DIR = os.environ.get('STREAM_DIR',
                            '/home/user/DyWA/output/%s/stream' % CHANNEL)
SHARD_EPISODES = int(os.environ.get('STREAM_SHARD_EPISODES', 200))
MAX_SHARDS = int(os.environ.get('STREAM_MAX_SHARDS', 45))
NUM_ENV = int(os.environ.get('STREAM_NUM_ENV', 32))
DINO_PROJ = int(os.environ.get('STREAM_DINO_PROJ', 64))
EVAL_DATA = [p for p in os.environ.get('STREAM_EVAL_DATA', '').split(',') if p]
POLL_S = float(os.environ.get('STREAM_POLL_S', 20.0))

SHARD_RE = re.compile(r'stream_(\d{5})\.hdf5$')


def _shard_path(i):
    return os.path.join(STREAM_DIR, 'stream_%05d.hdf5' % i)


def _done_path(p):
    return p[:-len('.hdf5')] + '.done'


def _sweep_orphans():
    """Delete shards with no `.done` marker -- partial files from a dead run.

    An unwritten HDF5 region reads back as 0, which is finite and passes any
    naive value check, so a truncated shard is exactly the kind of input that
    trains fine and poisons the result. The marker is the only evidence a shard
    is complete, so anything lacking one is garbage by definition.

    Also catches the shard in flight when a previous run exited: `G.main` calls
    `writer.close()` and then `os._exit(0)`, which bypasses everything, so the
    final partial shard is closed but never marked. Discarding it costs at most
    `STREAM_SHARD_EPISODES` episodes out of the tens of thousands a run
    produces, which is not worth fragile exit handling to recover.
    """
    n = 0
    for p in sorted(glob.glob(os.path.join(STREAM_DIR, 'stream_*.hdf5'))):
        if not os.path.exists(_done_path(p)):
            os.remove(p)
            n += 1
    if n:
        print('[stream] removed %d unmarked shard(s) -- partial, or the final '
              'shard of a previous run' % n)


def _next_index():
    """One past the highest shard index present, so a restart appends."""
    hi = -1
    for p in glob.glob(os.path.join(STREAM_DIR, 'stream_*.hdf5')):
        m = SHARD_RE.search(p)
        if m:
            hi = max(hi, int(m.group(1)))
    return hi + 1


def _adopt_reference_range():
    """Set FRIC_LO/HI/SPACING from the reference corpus's own attrs.

    Not optional bookkeeping. The per-channel Python defaults are NOT the values
    the existing corpora were collected at -- for mass the default is
    (0.05, 1.0) linear-in-log while `tracks_m_*` used [0.1, 1.2] -- so relying on
    the defaults produces shards on a different grid that `read_grid` then
    refuses to pool. Reading the range off the corpus makes the driver
    self-configuring and channel-agnostic: nothing here knows what `mass` or
    `table` means.

    Uses h5py directly rather than `models.fric.dataset.read_grid` because this
    has to run BEFORE the isaacgym import, and that module pulls in torch.
    An explicit caller override still wins.
    """
    if not EVAL_DATA:
        return
    import h5py
    with h5py.File(EVAL_DATA[0], 'r', locking=False) as f:
        a = f.attrs
        need = ('fric_channel', 'fric_lo', 'fric_hi', 'fric_grid')
        missing = [k for k in need if k not in a]
        assert not missing, (
            '%s lacks %s -- it was not written by collect_fric_tracks.py and '
            'cannot serve as the reference corpus' % (EVAL_DATA[0], missing))
        assert str(a['fric_channel']) == CHANNEL, (
            'STREAM_CHANNEL=%r but the reference corpus sweeps %r'
            % (CHANNEL, str(a['fric_channel'])))
        ref = {'FRIC_LO': repr(float(a['fric_lo'])),
               'FRIC_HI': repr(float(a['fric_hi'])),
               'FRIC_SPACING': str(a.get('fric_spacing', 'linear'))}
    for k, v in ref.items():
        if os.environ.get(k):
            print('[stream] %s=%s kept (overrides the corpus value %s)'
                  % (k, os.environ[k], v))
        else:
            os.environ[k] = v
    print('[stream] adopted reference range: %s in [%s, %s]'
          % (os.environ['FRIC_SPACING'], os.environ['FRIC_LO'],
             os.environ['FRIC_HI']))


os.makedirs(STREAM_DIR, exist_ok=True)
_sweep_orphans()
_STREAM = {'idx': _next_index(), 'blocked_s': 0.0, 'n_written': 0}

os.environ['FRIC_CHANNEL'] = CHANNEL
os.environ['GAP_OUT'] = _shard_path(_STREAM['idx'])
_adopt_reference_range()
os.environ.setdefault('GAP_DINO_PROJ', str(DINO_PROJ))
# No `image`: it is the largest per-frame cost after the tracks, and the trainer
# never reads it. Consequence, stated so it is not discovered later -- overlay
# videos and visualize_fric_conditioning.py cannot run on stream shards, so
# `fan/err` can only ever be recomputed on the frozen corpus.
os.environ.setdefault('GAP_STORE_IMAGE', '0')
# Effectively unbounded: the run is ended by killing it or by the trainer
# finishing, not by an episode quota.
os.environ.setdefault('GAP_NUM_EPISODES', str(10 ** 9))
os.environ.setdefault('GAP_MAX_STEPS', str(10 ** 9))

# isaacgym must be imported before torch; `collect_gap_tracks` does that, and
# `collect_fric_tracks` imports it first. Keep this import ahead of any torch
# import of our own.
import collect_fric_tracks as F                                   # noqa: E402

G = F.G


# ── grid precheck ─────────────────────────────────────────────────────────
def _check_grid_matches_reference():
    """Fail in seconds if the shards could not be pooled with the corpus.

    `read_grid` asserts across files at train time, so a mismatch WOULD be
    caught -- 20 hours later, after the collection is worthless. The same check
    up front costs nothing. Everything channel-specific is read from the corpus;
    no range or spacing is hardcoded.
    """
    if not EVAL_DATA:
        print('[stream] STREAM_EVAL_DATA unset -- skipping the grid precheck. '
              'The shards will still be checked against each other at train '
              'time, but nothing here proves they match the reference corpus.')
        return
    from models.fric.dataset import read_grid
    g = read_grid(EVAL_DATA)
    want = (np.geomspace(F.LO, F.HI, NUM_ENV) if F.SPACING == 'log'
            else np.linspace(F.LO, F.HI, NUM_ENV))
    assert g['channel'] == CHANNEL, (
        'reference corpus sweeps %r but STREAM_CHANNEL is %r'
        % (g['channel'], CHANNEL))
    assert g['col'] == F.PHYS_COL, (
        'phys_params column disagrees: corpus %d vs this run %d'
        % (g['col'], F.PHYS_COL))
    assert g['spacing'] == F.SPACING, (
        'spacing disagrees: corpus %r vs this run %r'
        % (g['spacing'], F.SPACING))
    assert len(g['grid']) == NUM_ENV, (
        'STREAM_NUM_ENV=%d but the reference grid has %d values. num_env IS '
        'the grid (collect_fric_tracks.py:170), so this run would sweep a '
        'DIFFERENT axis and read_grid would refuse to pool the shards with the '
        'corpus. Raising num_env for throughput needs a tiled grid -- see the '
        'module docstring.' % (NUM_ENV, len(g['grid'])))
    err = float(np.abs(np.asarray(g['grid'], np.float64) - want).max())
    assert err < 1e-4, (
        'grid mismatch vs the reference corpus (max|err| = %.3e). corpus '
        '[%.4f, %.4f], this run [%.4f, %.4f] -- set STREAM_LO/STREAM_HI to '
        'match.' % (err, g['grid'][0], g['grid'][-1], want[0], want[-1]))
    print('[stream] grid precheck OK: %d values, %s, [%.4f, %.4f], col %d '
          '-- matches %d reference file(s)'
          % (NUM_ENV, F.SPACING, want[0], want[-1], F.PHYS_COL, len(EVAL_DATA)))


# ── rotating writer ───────────────────────────────────────────────────────
def _n_ready():
    return len(glob.glob(os.path.join(STREAM_DIR, 'stream_*.done')))


def _wait_for_room():
    """Block until the trainer has consumed enough shards to make room.

    Without this a dead or stalled trainer means the collector fills the disk and
    takes both processes down. Blocking here stalls the whole collector, which is
    exactly the backpressure we want -- but it also means the run stops producing
    UNIQUE data for the duration, which is the entire point of streaming. So the
    blocked time is accumulated and reported rather than hidden.
    """
    t0 = None
    while _n_ready() >= MAX_SHARDS:
        if t0 is None:
            t0 = time.time()
            print('[stream] %d/%d shards on disk -- BLOCKING until the trainer '
                  'consumes some (no unique data is produced while blocked)'
                  % (_n_ready(), MAX_SHARDS), flush=True)
        time.sleep(POLL_S)
    if t0 is not None:
        dt = time.time() - t0
        _STREAM['blocked_s'] += dt
        print('[stream] unblocked after %.0f s (total blocked %.0f s)'
              % (dt, _STREAM['blocked_s']), flush=True)


class RotatingEpisodeWriter(G.EpisodeWriter):
    """`EpisodeWriter`, closed and reopened every `SHARD_EPISODES` episodes.

    Overrides `write_episode` ONLY. It must not define `close`:
    `collect_fric_tracks.main()` assigns `G.EpisodeWriter.close = close` to add
    the `fric_grid` / `dino_dim` attrs, and since that assignment lands on
    whichever class is bound at the time, defining `close` here would be
    clobbered by it (or clobber it, depending on order) and the shards would
    lose the grid metadata that `read_grid` requires.

    `_init_path` resets `ds`/`T`/`n_ep`; `cam_cfg`, `T_cam`, `act_lo` and
    `act_hi` are instance attributes set once and survive the reopen, so every
    shard carries identical attrs.
    """

    def write_episode(self, frames, success=False):
        n = super().write_episode(frames, success)
        if n and self.n_ep >= SHARD_EPISODES:
            self._rotate()
        return n

    def _rotate(self):
        path = self.f.filename
        n_ep, n_fr = self.n_ep, self.T
        self.close()                      # patched: writes the fric_*/dino_* attrs
        with open(_done_path(path), 'w') as fp:
            json.dump({'episodes': int(n_ep), 'frames': int(n_fr),
                       'channel': CHANNEL,
                       'closed_at': time.time()}, fp)
        _STREAM['n_written'] += 1
        print('[stream] shard %s closed: %d episodes, %d frames '
              '(%d total shards written, %d ready on disk)'
              % (os.path.basename(path), n_ep, n_fr,
                 _STREAM['n_written'], _n_ready()), flush=True)
        _wait_for_room()
        _STREAM['idx'] += 1
        self._init_path(_shard_path(_STREAM['idx']))


def main():
    print('[stream] channel=%s dir=%s shard=%d episodes max_shards=%d '
          'num_env=%d dino=%d' % (CHANNEL, STREAM_DIR, SHARD_EPISODES,
                                  MAX_SHARDS, NUM_ENV, DINO_PROJ))
    print('[stream] first shard: %s' % _shard_path(_STREAM['idx']))
    _check_grid_matches_reference()
    # Bind the subclass BEFORE F.main(): `G.main` resolves `EpisodeWriter` as a
    # module global when it constructs the writer, and F.main()'s `close` patch
    # then lands on our subclass. `_orig_close` inside collect_fric_tracks was
    # captured at ITS import time, i.e. the original parent method, so the chain
    # stays fric-close -> parent-close.
    G.EpisodeWriter = RotatingEpisodeWriter
    _wait_for_room()
    F.main()


if __name__ == '__main__':
    main()
