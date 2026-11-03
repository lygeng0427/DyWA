#!/usr/bin/env python3
"""Collect GAP-style 3D tracks with ONE physics channel swept over a per-env grid.

This is a thin driver around `collect_gap_tracks.py`, not a fork of it. The
collector's frame loop carries the `ep_start` / reset-staleness bookkeeping that
decides whether a track window spans an episode reset; duplicating it to add a
friction sweep would duplicate that risk for no benefit. So this file only:

  1. bakes a per-env friction grid at the ONE point in the lifecycle where such
     a write reaches the simulator,
  2. mutes the reset-time DR that would otherwise overwrite it,
  3. stamps the labels BY CONSTRUCTION and the grid into the file attrs,

and then calls the unmodified `collect_gap_tracks.main()`.

WHY THE BAKE (do not "simplify" this away)
------------------------------------------
Friction DR at RESET calls `set_actor_rigid_shape_properties`, which is inert on
the GPU pipeline: the host-side struct takes the new value while the GPU-resident
material keeps the old one. `read_raw_phys` then reads that host struct back and
returns the value the DR *intended*, not the one that ran. That is exactly how
`output/gap/tracks_fricvar_v2_*.hdf5` ended up with 100 distinct friction labels
over a simulation that only ever ran 8 -- pure fiction, and a model conditioned
on it is conditioned on noise.

Only writes issued BEFORE `gym.prepare_sim` land. `push_env.py` calls
`env.setup()` and then `prepare_sim` on the next line, so wrapping `PushEnv.setup`
is the seam. This machinery is ported from `diag_friction_split.py`, where the
readback was verified truthful at all three checkpoints (after bake / after
prepare_sim / after first reset) and the resulting sweep produced r2(log) = 0.89
on free-slide distance -- i.e. it demonstrably reaches the physics.

Muting the reset DR cannot be done from the command line: `Config.__post_init__`
recomputes `use_dr = <value> | is_phase2` (train_ppo_arm.py:351-352) and every
later `dataclasses.replace` inside `load_env` re-runs it, so with
`+run=teacher_base` (is_phase2=True) a `++...use_dr=False` is silently discarded.
The scene reads the flag off its own live cfg instance at reset, so muting it
there is the only place that survives.

WHY linspace AND NOT THE ENGINE'S OWN uniform_ DRAW
---------------------------------------------------
The scene already samples a per-env `cur_table_friction` via `uniform_`, but two
GPUs collecting in parallel run different seeds, so they would land on two
different sets of 32 values and the friction axis would fragment into 64 buckets
of ~39 episodes each. A `linspace` is identical on both GPUs regardless of
`GAP_SEED` (which still varies object poses and goals), and the held-out-mu split
downstream needs the grid to be known exactly.

MASS IS A DIFFERENT WRITE PATH, AND A LESS FORGIVING ONE
--------------------------------------------------------
Friction is a rigid *shape* property; mass is a rigid *body* property, written
with `set_actor_rigid_body_properties(..., recomputeInertia=True)`. Two
differences that matter:

  * a mistimed friction write is silently INERT; a mistimed (mid-sim) mass write
    CORRUPTS the simulation -- the object explodes. The pre-`prepare_sim` seam is
    therefore mandatory, not merely correct.
  * the reset-time DR never touches mass (`enable_mass` is not passed at
    tabletop_with_object_scene.py:806-830), so `read_raw_phys`'s mass readback is
    truthful on its own. The by-construction stamp below is kept anyway: it costs
    nothing and the readback is precisely the evidence that was fiction for
    friction.

The machinery is ported from `_probe_param_headroom.py`, where a per-env mass
bake in this same seam was verified to survive to rollout time.

Env knobs (on top of every GAP_* knob of collect_gap_tracks.py):
    FRIC_CHANNEL   table | object | mass   which quantity is swept
    FRIC_LO        grid minimum            (friction 0.1, mass 0.05)
    FRIC_HI        grid maximum            (friction 1.5, mass 1.0)
    FRIC_SPACING   linear | log            (friction linear, mass log)
"""

import os
import sys

import numpy as np

# collect_gap_tracks imports isaacgym before torch; keep it first here too.
import collect_gap_tracks as G                                   # noqa: E402
import torch as th                                               # noqa: E402
from env.push_env import PushEnv                                 # noqa: E402

CHANNEL = os.environ.get('FRIC_CHANNEL', 'table')

# Defaults are per-channel because the two quantities are not alike. Friction is
# a dimensionless coefficient over a modest range, so a linear grid resolves it
# evenly. Mass enters through a ~ F/m, so what matters is the RELATIVE step: a
# linear grid over [0.05, 1.0] would put 20 of 32 buckets where consecutive
# masses differ by 3% and only 2 where they differ by 60%.
_DEFAULT_RANGE = {'table': (0.1, 1.5), 'object': (0.1, 1.5), 'mass': (0.05, 1.0)}
_DEFAULT_SPACING = {'table': 'linear', 'object': 'linear', 'mass': 'log'}

assert CHANNEL in _DEFAULT_RANGE, \
    f'FRIC_CHANNEL must be table|object|mass, got {CHANNEL!r}'

_LO_D, _HI_D = _DEFAULT_RANGE[CHANNEL]
LO = float(os.environ.get('FRIC_LO', _LO_D))
HI = float(os.environ.get('FRIC_HI', _HI_D))
SPACING = os.environ.get('FRIC_SPACING', _DEFAULT_SPACING[CHANNEL])

assert HI > LO, f'FRIC_HI ({HI}) must exceed FRIC_LO ({LO})'
assert SPACING in ('linear', 'log'), \
    f'FRIC_SPACING must be linear|log, got {SPACING!r}'
assert SPACING != 'log' or LO > 0, 'a log grid needs FRIC_LO > 0'

# Column in `phys_params`: [mass, table_fric, obj_fric, hand_fric, restitution]
PHYS_COL = {'table': 1, 'object': 2, 'mass': 0}[CHANNEL]

# DINO width to STORE. 0/1024 = untouched. k>0 stores a fixed random projection
# 1024 -> k, which is the whole reason large-scale collection is possible here:
# the features are 782 of the 805 KB per frame (93%), and a measured ablation put
# their worth at 2% of mv:object at k=64 and 6% at k=0. So k=64 costs 2% accuracy
# for 11x more episodes per GB -- and at the measured data-scaling exponent
# (-0.20) that trade is strongly net-positive.
#
# The projection matrix is regenerated from DINO_PROJ_SEED, and BOTH the seed and
# the width go into the file attrs. A consumer that projects with a different
# matrix than the file was written with sees garbage that looks like a bad model,
# not an error -- so the seed is recorded, never assumed.
DINO_PROJ = int(os.environ.get('GAP_DINO_PROJ', 0))
DINO_PROJ_SEED = int(os.environ.get('GAP_DINO_PROJ_SEED', 0))

_STATE = {}


def _mute_reset_dr(env):
    """Stop the reset-time DR from re-drawing (and mislabelling) friction.

    Physically the reset write is inert, so this changes no dynamics. What it
    changes is the BOOKKEEPING: without it, `scene.cur_*_friction` and the
    host-side shape properties both get overwritten with a fresh random draw at
    every reset, and any readback after the first episode is fiction.
    """
    for attr, holder in (('cfg', env.scene), ('scene_cfg', env)):
        sc = getattr(holder, attr, None)
        if sc is None:
            continue
        for flag in ('use_dr', 'use_dr_on_setup'):
            if hasattr(sc, flag):
                object.__setattr__(sc, flag, False)


def _read_channel(env):
    """The SWEPT quantity, per env, straight out of the simulator.

    Only the swept channel: this readback exists to catch a bake that did not
    land, and a value the bake never wrote proves nothing either way.
    """
    gym, scene = env.gym, env.scene
    out = np.zeros(env.num_env, float)
    for i in range(env.num_env):
        if CHANNEL == 'mass':
            pr = gym.get_actor_rigid_body_properties(
                env.envs[i], int(scene.obj_handles[i, 0]))
            out[i] = float(pr[0].mass)
        else:
            handle = (scene.table_handles[i] if CHANNEL == 'table'
                      else int(scene.obj_handles[i, 0]))
            out[i] = float(gym.get_actor_rigid_shape_properties(
                env.envs[i], handle)[0].friction)
    return out


def _bake_channel(env):
    """Write the per-env grid onto the swept body. Runs pre-`prepare_sim`."""
    gym, scene = env.gym, env.scene
    n_env = env.num_env
    n_obj = int(scene.obj_handles.shape[1])
    grid = (np.geomspace(LO, HI, n_env) if SPACING == 'log'
            else np.linspace(LO, HI, n_env))

    for i in range(n_env):
        if CHANNEL == 'mass':
            # Every object actor in the env, not only the currently active one:
            # which handle is current changes at reset, and unlike friction,
            # nothing rewrites mass afterwards to fix it up.
            for j in range(n_obj):
                oh = int(scene.obj_handles[i, j])
                pr = gym.get_actor_rigid_body_properties(env.envs[i], oh)
                # `for p in pr` writes EVERY body of the actor, so a multi-body
                # asset would end up at len(pr) x the grid value while the label
                # still said grid[i]. DGN objects are single-body (AddPhysParams
                # asserts the same); assert rather than assume.
                assert len(pr) == 1, (
                    f'object actor has {len(pr)} bodies -- a per-body mass '
                    'write would not sum to the labelled value')
                for p in pr:
                    p.mass = float(grid[i])
                # 4th arg is recomputeInertia. WITHOUT IT the inertia tensor is
                # left at the config mass while the mass changes, so the object
                # translates as if heavy and rotates as if light. A pushed
                # bottle's track is largely rotation, so that would corrupt
                # exactly the signal under study -- and the readback below would
                # still look perfect.
                if not gym.set_actor_rigid_body_properties(
                        env.envs[i], oh, pr, True):
                    raise RuntimeError(f'mass write failed (env {i}, obj {j})')
        else:
            handle = (scene.table_handles[i] if CHANNEL == 'table'
                      else int(scene.obj_handles[i, 0]))
            props = gym.get_actor_rigid_shape_properties(env.envs[i], handle)
            for p in props:
                p.friction = float(grid[i])
            if not gym.set_actor_rigid_shape_properties(
                    env.envs[i], handle, props):
                raise RuntimeError(f'{CHANNEL} friction write failed (env {i})')

    _mute_reset_dr(env)
    _STATE['grid'] = grid
    _STATE['rb_after_bake'] = _read_channel(env)

    err = np.abs(_STATE['rb_after_bake'] - grid).max()
    step = ('x%.4f' % (grid[1] / grid[0]) if SPACING == 'log'
            else '+%.4f' % (grid[1] - grid[0]))
    print(f'[fric] baked {CHANNEL} over {n_env} envs: '
          f'{grid[0]:.4f} .. {grid[-1]:.4f}  ({SPACING}, step {step})')
    print(f'[fric] readback max|err| after bake = {err:.3e}')
    print(f'[fric] reset-time DR muted on live scene cfg '
          f'(use_dr={getattr(env.scene.cfg, "use_dr", "n/a")})')
    assert err < 1e-5, f'{CHANNEL} bake did not read back -- write path is wrong'


def install_bake_hooks():
    """Run the bake between `scene.setup()` and `gym.prepare_sim`."""
    orig_setup = PushEnv.setup

    def setup(self):
        out = orig_setup(self)
        _bake_channel(self)
        return out

    PushEnv.setup = setup


# ── label stamping ────────────────────────────────────────────────────────
_orig_read_raw_phys = G.read_raw_phys


def read_raw_phys(env, E, device):
    """`collect_gap_tracks.read_raw_phys`, with the swept column made truthful.

    The readback is believed correct here (the bake survives prepare_sim and
    reset, and the reset DR is muted), but "believed correct" is what produced
    the fictional labels last time. So the swept column is written from the grid
    BY CONSTRUCTION and the readback is only used to assert agreement, once.
    """
    out = _orig_read_raw_phys(env, E, device)
    if 'grid' not in _STATE:               # the bake hook never fired
        raise RuntimeError('read_raw_phys called before the friction bake')
    # Built lazily: at bake time (inside PushEnv.setup) the env's torch device
    # is not necessarily settled yet, and `out` names the right one anyway.
    grid = _STATE.get('grid_t')
    if grid is None or grid.device != out.device:
        grid = th.as_tensor(_STATE['grid'], dtype=out.dtype, device=out.device)
        _STATE['grid_t'] = grid
    if not _STATE.get('checked'):
        err = float((out[:, PHYS_COL] - grid).abs().max())
        print(f'[fric] readback max|err| at first frame = {err:.3e}')
        assert err < 1e-4, (
            f'{CHANNEL} friction readback disagrees with the baked grid by '
            f'{err:.3e} -- something overwrote it (reset DR still live?)')
        _STATE['checked'] = True
    out[:, PHYS_COL] = grid
    return out


# ── attrs ─────────────────────────────────────────────────────────────────
_orig_close = G.EpisodeWriter.close


def close(self):
    """Record the grid so training normalises against the KNOWN range.

    Normalising mu against dataset statistics instead would make the held-out-mu
    run inconsistent with the others: its training set is missing 8 of the 32
    values, so its mean/std -- and therefore the meaning of every conditioning
    input -- would differ from the runs it is compared against.
    """
    grid = _STATE.get('grid')
    if grid is not None and getattr(self, 'f', None) is not None:
        self.f.attrs['fric_channel'] = CHANNEL
        self.f.attrs['fric_spacing'] = SPACING
        self.f.attrs['fric_col'] = PHYS_COL
        self.f.attrs['fric_grid'] = np.asarray(grid, np.float32)
        self.f.attrs['fric_lo'] = float(LO)
        self.f.attrs['fric_hi'] = float(HI)
    if getattr(self, 'f', None) is not None and DINO_PROJ:
        self.f.attrs['dino_dim'] = int(DINO_PROJ)
        self.f.attrs['dino_proj_seed'] = int(DINO_PROJ_SEED)
    return _orig_close(self)


def _install_dino_projection():
    """Store narrowed DINO features instead of the raw 1024-d ones."""
    import torch as _th
    k = DINO_PROJ
    G.FEAT_DIM = k                       # the writer sizes its dataset from this
    gen = _th.Generator().manual_seed(DINO_PROJ_SEED)
    P = _th.randn(1024, k, generator=gen) / float(np.sqrt(1024.0))
    _STATE['dino_P'] = P
    orig = G.dino_features

    def dino_features(model, color, mean, std):
        f = orig(model, color, mean, std)          # (E, N, 1024)
        return f @ P.to(f.device, f.dtype)

    G.dino_features = dino_features
    print(f'[fric] DINO features stored PROJECTED 1024 -> {k} '
          f'(seed {DINO_PROJ_SEED}); ~{100 * (1 - k / 1024):.0f}% fewer '
          f'feature bytes per frame')


_orig_close2 = G.EpisodeWriter.close


def main():
    print(f'[fric] channel={CHANNEL} (phys_params col {PHYS_COL}) '
          f'range=[{LO}, {HI}] spacing={SPACING}')
    if DINO_PROJ:
        _install_dino_projection()
    install_bake_hooks()
    G.read_raw_phys = read_raw_phys
    G.EpisodeWriter.close = close
    # hydra_cli reads sys.argv; collect_gap_tracks.main is the decorated entry.
    G.main()


if __name__ == '__main__':
    sys.exit(main())
