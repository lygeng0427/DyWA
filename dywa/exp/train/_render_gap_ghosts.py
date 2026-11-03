#!/usr/bin/env python3
"""Render translucent goal-pose 'ghosts' for already-collected GAP datasets.

The dataset stores `goal_pose` but not a rendered ghost, and a ghost is a real
nvdiffrast render of the object's visual mesh -- it cannot be reconstructed
from the stored frames. This does a RENDER-ONLY pass: it never steps physics
and never touches the datasets, it just asks the same nvdr camera to draw the
object at each stored goal pose with every other body pushed beyond z_far
(exactly what NvdrRecordEpisode._render_ghost_batch does for the TTT videos).

Output: an .npz of {key: rgb[H,W,3] uint8, key_mask: [H,W] bool}, keyed
"<tag>:<episode>", consumed by visualize_gap_tracks.py --ghost.

Run (inside container), one dataset per --spec, repeated:
    python3 _render_gap_ghosts.py \
        --spec mass:/home/user/DyWA/output/gap/tracks_100succ_massvar.hdf5:97,10 \
        --spec fric:/home/user/DyWA/output/gap/tracks_100succ_fricvar.hdf5:2,63 \
        --out /home/user/DyWA/output/gap/ghosts.npz \
        <the usual hydra args>
"""

from isaacgym import gymtorch  # noqa: F401
from isaacgym import gymapi    # noqa: F401

import os
import sys
from dataclasses import replace

import h5py
import numpy as np
import torch as th

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from util.math_util import matrix_from_quaternion
from util.torch_util import dcn
from env.util import set_seed
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper

from train_ppo_arm import AddTensorboardWriter, setup as setup_logging, load_env
from train_rma import Config as RMAConfig, get_config_path
from collect_gap_tracks import make_cam_cfg

# --spec / --out are consumed here so hydra never sees them.
SPECS, OUT_NPZ = [], None
_argv = []
_it = iter(sys.argv[1:])
for a in _it:
    if a == '--spec':
        SPECS.append(next(_it))
    elif a == '--out':
        OUT_NPZ = next(_it)
    else:
        _argv.append(a)
sys.argv = [sys.argv[0]] + _argv


@hydra_cli(config_path=get_config_path(), config_name='train_rl')
def main(cfg: RMAConfig):
    assert SPECS and OUT_NPZ, 'need --spec tag:path:ep,ep and --out file.npz'
    jobs = []                                     # (tag, path, episode)
    for spec in SPECS:
        tag, path, eps = spec.split(':')
        for ep in eps.split(','):
            jobs.append((tag, path, int(ep)))
    print(f'[ghost] {len(jobs)} ghosts to render')

    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path_out = setup_logging(cfg)
    set_seed(cfg.env.seed)

    cfg, env = load_env(cfg, path_out, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(None)

    cam_cfg = make_cam_cfg()
    img_env = NvdrCameraWrapper(env, cam_cfg)
    img_env.reset()

    dev = env.device
    N = env.num_env
    assert N >= len(jobs), (
        f'need num_env >= {len(jobs)} to render all ghosts in one batch, got {N}')

    # goal poses from the datasets (constant within an episode)
    goals = th.zeros(N, 7, device=dev)
    goals[:, 6] = 1.0
    for i, (tag, ds, ep) in enumerate(jobs):
        with h5py.File(ds, 'r', locking=False) as f:
            ends = f['episode_ends'][:]
            s = 0 if ep == 0 else int(ends[ep - 1])
            goals[i] = th.as_tensor(f['goal_pose'][s], device=dev)
            # the ghost must match the camera the frames were rendered with
            T_file = np.asarray(f.attrs['T_cam'], np.float64).reshape(4, 4)
        print(f'[ghost] {tag} ep{ep}: goal {dcn(goals[i]).round(4)}')

    from env.env.help.with_nvdr_camera import camera_matrices
    T_cam, _ = camera_matrices(cam_cfg, 1, dev)
    d = float(np.abs(dcn(T_cam).reshape(4, 4) - T_file).max())
    print(f'[ghost] camera matches dataset attrs: max|dT| = {d:.6f}')
    assert d < 1e-4, 'camera differs from the one that rendered the frames'

    # object body row, and a pose stack with everything else beyond z_far
    obj_row = env.scene.obj_body_ids[:, 0].long()
    g4 = th.eye(4, device=dev, dtype=th.float32).repeat(N, 1, 1)
    g4[:, :3, 3] = goals[:, :3]
    matrix_from_quaternion(goals[:, 3:7], g4[:, :3, :3])

    P = img_env.body_poses_4x4.shape[1]
    ghost = th.eye(4, device=dev, dtype=th.float32).repeat(N, P, 1, 1)
    ghost[:, :, :3, 3] = 1.0e3                     # push everything off-screen
    ghost[th.arange(N, device=dev), obj_row] = g4

    with th.no_grad():
        out = img_env.cam(ghost)
    color = out['color']                           # [N,H,W,3] float
    mask = (color.max(dim=-1).values > 0.05)
    color = (255.0 * color.clamp(0.0, 1.0)).to(th.uint8)
    color, mask = dcn(color), dcn(mask)

    blob = {}
    for i, (tag, ds, ep) in enumerate(jobs):
        k = f'{tag}:{ep}'
        blob[k] = color[i]
        blob[k + '_mask'] = mask[i]
        print(f'[ghost] {k}: {int(mask[i].sum())} pixels drawn')
    os.makedirs(os.path.dirname(os.path.abspath(OUT_NPZ)), exist_ok=True)
    np.savez_compressed(OUT_NPZ, **blob)
    print(f'[ghost] wrote {OUT_NPZ}')
    os._exit(0)


if __name__ == '__main__':
    main()
