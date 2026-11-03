#!/usr/bin/env python3
"""Dataset over the GAP-flat HDF5 written by collect_gap_tracks.py.

Scope: the SINGLE-FRICTION setting. Every episode in these files was generated
under one pinned dynamics config, so there is no hidden physics variable and
nothing here splits on one -- the earlier friction-value split machinery is
gone with it.

What "held out" means now: a val episode has a different initial object pose,
goal draw and seed, but the SAME physics. This is a track-generalization test,
not a physics-generalization test. Do not read a good val number as evidence
about unseen dynamics.

Differences from GAP's own TrackDataset, all forced or deliberate:

  * no zarr / numba / ReplayBuffer -- h5py directly, and the index is built in
    plain python (it runs once).
  * frames whose H-step future does not exist carry track_valid_mask_gt=False
    (the collector emits them so the goal-reaching tail stays visible). They
    are excluded from the sampling index -- filter on the MASK, never on frame
    index.
  * every frame carries `action_seq` -- the EEF pose at each of the next H
    steps, the SAME horizon as the track. Without it the future track is
    underdetermined: the same scene moves differently depending on where the
    arm goes next. `action[t]` is the realized next-step EEF pose (verified
    equal to `robot_state[t+1]` on every non-tail frame), so `action[t..t+H-1]`
    is exactly the H-step EEF path the tracks were generated under. It is
    REALIZED, not commanded -- fine for "predict the scene motion this arm
    motion causes", but note it if you ever use the model to score candidate
    action chunks.
  * `action_cmd_seq` is the POLICY's own 20-D command over the same window --
    3 dpos, 3 dori (axis-angle), 7 joint stiffness KP, 7 joint damping KD. It
    is a different object from `action_seq`: the command is what was ASKED for,
    the realized pose is what the arm DID after numerical IK, an impedance
    controller, control noise and a randomized delay. Measured correlation
    between commanded and realized dpos is only 0.79/0.92/0.42 per axis, and the
    14 gain dims have no counterpart in the realized pose at all -- they set the
    contact FORCE, which pusher kinematics cannot express. Stored as the raw
    normalised mu; `act_lo`/`act_hi` in the file attrs map it to metres/rad/N.
  * `owner` (patch_owner_body) rides along so eval can break the error down
    per body. With the action given, arm-link tracks become nearly determined
    while the OBJECT's stay genuinely uncertain, so an aggregate number mixes
    two very different problems.
"""
import os
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch as th

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

PHYS_NAMES = ['mass', 'table_fric', 'obj_fric', 'hand_fric', 'restitution']


class GapTrackDataset(th.utils.data.Dataset):
    """One item = one frame: the model inputs, the GT track, and its episode id.

    `episodes` selects which episodes participate; the ids are indices into the
    concatenated episode list across `paths`, in file order. The same id is
    returned as `episode`, so a per-episode latent table can be indexed with it
    directly and train/val ids never alias.
    """

    def __init__(self,
                 paths,
                 episodes: Optional[Sequence[int]] = None,
                 track_key: str = 'gt_3d_track',
                 mask_key: str = 'track_valid_mask_gt',
                 load_dino: bool = True,
                 hist_k: int = 0):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = list(paths)
        self.track_key = track_key
        self.mask_key = mask_key
        self.load_dino = load_dino
        self.hist_k = hist_k
        self._files: Dict[int, h5py.File] = {}      # lazily opened per worker

        # ---- episode table across all files -------------------------------
        self.ep: List[Tuple[int, int, int]] = []     # (file_idx, start, end)
        for fi, p in enumerate(self.paths):
            with h5py.File(p, 'r', locking=False) as f:
                ends = f['episode_ends'][:]
                starts = np.concatenate([[0], ends[:-1]])
                for s, e in zip(starts, ends):
                    self.ep.append((fi, int(s), int(e)))
        self.n_ep_total = len(self.ep)

        with h5py.File(self.paths[0], 'r', locking=False) as f:
            self.act_lo = (np.asarray(f.attrs['act_lo'], np.float32)
                           if 'act_lo' in f.attrs else None)
            self.act_hi = (np.asarray(f.attrs['act_hi'], np.float32)
                           if 'act_hi' in f.attrs else None)
        assert self.act_lo is not None, (
            '%s has no act_lo/act_hi attrs -- it predates action_cmd; '
            're-collect with the current collect_gap_tracks.py' % self.paths[0])

        keep = (np.arange(self.n_ep_total) if episodes is None
                else np.asarray(episodes, dtype=np.int64))
        self.episodes = keep

        # ---- frame index: only frames with a real future window ------------
        # (ep, file, t, ep_end, ep_start) -- ep_start bounds the history lookup
        self.index: List[Tuple[int, int, int, int, int]] = []
        self.H = None
        for k in keep:
            fi, s, e = self.ep[int(k)]
            with h5py.File(self.paths[fi], 'r', locking=False) as f:
                m = f[mask_key][s:e]                 # [T, N, H]
                self.H = int(m.shape[-1])
                valid = m.reshape(len(m), -1).any(-1)
            for t in np.nonzero(valid)[0]:
                self.index.append((int(k), fi, s + int(t), e, s))

    # -- physics, one row per episode (for the pin assertion / reporting) ----
    def episode_phys(self) -> np.ndarray:
        out = np.zeros((len(self.episodes), 5), np.float32)
        for i, k in enumerate(self.episodes):
            fi, s, _ = self.ep[int(k)]
            with h5py.File(self.paths[fi], 'r', locking=False) as f:
                out[i] = f['phys_params'][s]
        return out

    def _f(self, fi: int) -> h5py.File:
        if fi not in self._files:
            self._files[fi] = h5py.File(self.paths[fi], 'r', locking=False)
        return self._files[fi]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep_id, fi, t, ep_end, ep_start = self.index[i]
        f = self._f(fi)
        # The H-step action window is why this frame is valid at all, so it
        # must fit inside the episode. Assert rather than clamp: a silent clamp
        # would feed a stalled arm path for the last frames and teach the model
        # that the object stops moving.
        assert t + self.H <= ep_end, (
            'action window [%d, %d) leaves episode ending at %d'
            % (t + 1, t + 1 + self.H, ep_end))
        out = {
            'patch_centers': th.from_numpy(
                f['image_patch_centre_3d_position'][t].astype(np.float32)),
            'agent_pos': th.from_numpy(f['robot_state'][t].astype(np.float32)),
            'gt_3d_track': th.from_numpy(f[self.track_key][t].astype(np.float32)),
            'track_valid_mask': th.from_numpy(f[self.mask_key][t].astype(np.bool_)),
            'episode': th.tensor(ep_id, dtype=th.long),
            # EEF pose at t+1 .. t+H (== robot_state[t+1..t+H])
            'action_seq': th.from_numpy(
                f['action'][t:t + self.H].astype(np.float32)),
            'action_cmd_seq': th.from_numpy(
                f['action_cmd'][t:t + self.H].astype(np.float32)),
            'owner': th.from_numpy(f['patch_owner_body'][t].astype(np.int64)),
        }
        if self.load_dino:
            d = f['dino_features'][t].astype(np.float32)     # [1, N, D]
            out['dino_features'] = th.from_numpy(d.reshape(-1, d.shape[-1]))
        if self.hist_k > 0:
            # PAST frames, newest first: t-1, t-2, ... A single frame cannot
            # show velocity, and the object's short-horizon motion depends on
            # whether it is already moving -- which is exactly where the model
            # has almost no skill (85% relative error at h=1).
            #
            # Clamped to ep_start rather than dropping short-history frames, so
            # the sampling INDEX is unchanged and these runs stay comparable
            # frame-for-frame with the no-history ones. At an episode's first
            # frames the clamp gives zero velocity, which is also roughly true
            # there (the arm has not reached the object yet).
            #
            # Read one contiguous slice and index into it in memory: h5py fancy
            # indexing requires strictly increasing, unique indices, and the
            # clamp produces neither (t-1, t-2 is decreasing, and near ep_start
            # it repeats).
            lo = max(t - self.hist_k, ep_start)
            pc_blk = f['image_patch_centre_3d_position'][lo:t + 1]
            rs_blk = f['robot_state'][lo:t + 1]
            rel = [max(t - j, ep_start) - lo for j in range(1, self.hist_k + 1)]
            out['patch_centers_hist'] = th.from_numpy(
                pc_blk[rel].astype(np.float32))
            out['agent_pos_hist'] = th.from_numpy(rs_blk[rel].astype(np.float32))
        return out


def count_episodes(paths) -> int:
    if isinstance(paths, str):
        paths = [paths]
    n = 0
    for p in paths:
        with h5py.File(p, 'r', locking=False) as f:
            n += len(f['episode_ends'])
    return n


def split_by_episode(paths, val_ratio: float = 0.2, seed: int = 0,
                     val_pool: Optional[int] = None):
    """Hold out whole EPISODES (never frames).

    A frame-level split would put neighbouring frames of the same rollout on
    both sides -- consecutive frames share almost all of their patches and most
    of their future, so val error would mostly measure interpolation within a
    trajectory the model already saw.

    Physics is identical on both sides here by construction (one pinned
    config); what differs is the initial object pose, goal and seed.

    `val_pool` draws the val episodes only from the FIRST n episodes, sending
    everything beyond into train. With the same seed that reproduces an earlier
    split's val set EXACTLY while new files only enlarge train -- which is the
    only way a "more data" comparison isolates the data from a reshuffled val
    set. Without it, appending a file repermutes everything and the two runs are
    scored on different episodes.
    """
    n_ep = count_episodes(paths)
    pool = n_ep if val_pool is None else min(int(val_pool), n_ep)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool)
    n_val = max(1, int(round(pool * val_ratio)))
    va = np.sort(perm[:n_val])
    tr = np.sort(np.concatenate([perm[n_val:], np.arange(pool, n_ep)]))
    return tr, va, n_ep


def assert_single_physics(paths, expect: Optional[Dict[str, float]] = None,
                          tol: float = 1e-6):
    """Every episode must share ONE dynamics config -- and the RIGHT one.

    Two failures to catch, not one. Reset-time domain randomisation has been
    silently inert on the GPU pipeline before, and the previous varying-friction
    collection shipped hand_friction=1.5 on every episode while the script
    asked for 1.25. So "all episodes agree" is necessary but not sufficient:
    constant-but-wrong looks identical to a uniqueness check and is exactly the
    failure mode that has bitten this project.

    `expect` maps a name in PHYS_NAMES to the requested value.
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
        u = np.unique(np.round(rows[:, c], 6))
        if len(u) != 1:
            bad.append('%s varies: %d unique values in [%.4f, %.4f]'
                       % (name, len(u), u.min(), u.max()))
        elif expect is not None and name in expect:
            if abs(float(u[0]) - float(expect[name])) > tol:
                bad.append('%s is constant at %.6f but %.6f was requested'
                           % (name, u[0], expect[name]))
    assert not bad, ('phys_params check failed over %d episodes:\n  %s'
                     % (len(rows), '\n  '.join(bad)))
    return {n: float(rows[0, c]) for c, n in enumerate(PHYS_NAMES)}, len(rows)
