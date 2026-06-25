#!/usr/bin/env python3
"""
Render one scene image per object in the (filtered) object set.

Used to visualize the TRAINING can/bottle instances: point the filter_file at
can_bottle_train_gallery.json and every distinct object gets one PNG showing it
resting on the table (the arm is hidden by NvdrCameraWrapper.hide_arm).

No policy is needed — the env is stepped with zero actions; we just let each
object settle and grab a camera frame, deduplicated by object key.

MUST be run inside the DyWA Docker container (needs Isaac Gym).
"""

from isaacgym import gymtorch  # noqa: F401  (must import before torch)
from isaacgym import gymapi    # noqa: F401

import json
from pathlib import Path

import numpy as np
import torch as th
import cv2
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from util.path import ensure_directory
from util.torch_util import dcn
from util.vis.img import tile_images, to_hwc
from env.util import set_seed
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper

from train_ppo_arm import (
    AddTensorboardWriter,
    setup as setup_logging,
    load_env,
)
from test_rma import Config, get_config_path


def _to_uint8_hwc(color: th.Tensor) -> np.ndarray:
    """NCHW float/uint8 camera tensor -> NHWC uint8 numpy (same path as
    NvdrRecordEpisode)."""
    rgb = dcn(color)
    if rgb.dtype != np.uint8:
        rgb = (255 * rgb.clip(0.0, 1.0)).astype(np.uint8)
    return to_hwc(rgb)


@hydra_cli(config_path=get_config_path(), config_name='show')
def main(cfg: Config):
    th.backends.cudnn.benchmark = True
    ic.configureOutput(includeContext=True)
    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})

    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    writer = SummaryWriter(path.tb_train)
    set_seed(cfg.env.seed)

    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(writer)

    # The shared hide_arm blocklist hides only the arm links, leaving the
    # gripper (panda_hand/fingers) floating in frame. For a clean object-only
    # gallery, hide the *whole* robot -- patched locally so the perception
    # pipeline (which uses the same function) is untouched.
    import env.env.wrap.nvdr_camera_wrapper as _ncw
    _orig_blocklist = _ncw.get_hacky_blocklist_for_arm_links

    def _full_robot_blocklist():
        bl = _orig_blocklist()
        extra = ['panda_link8', 'panda_hand', 'panda_tool',
                 'panda_leftfinger', 'panda_rightfinger']
        for asset_path, links in bl.items():
            if 'panda' in asset_path:
                links.extend([e for e in extra if e not in links])
        return bl
    _ncw.get_hacky_blocklist_for_arm_links = _full_robot_blocklist

    # Off-screen camera (robot hidden) used only for rendering.
    img_env = NvdrCameraWrapper(
        env,
        NvdrCameraWrapper.Config(
            img_size=(512, 512),
            use_color=True,
            use_col=True,
            # Frame each object centered + scaled by its radius (same framing
            # the perception pipeline uses) so every object fills the view.
            track_object=True,
        ),
    )

    out_dir = ensure_directory(
        cfg.gallery_dir
        if cfg.gallery_dir is not None
        else (Path(cfg.path.root) / 'gallery'))

    # Target object set = the filter file the scene was built with.
    filter_file = cfg.env.single_object_scene.filter_file
    targets = None
    if filter_file is not None and Path(filter_file).is_file():
        targets = set(str(s) for s in json.load(open(filter_file)))
        print(f'gallery targets: {len(targets)} objects from {filter_file}')

    # Zero actions; we only need the objects, not a policy.
    act = th.zeros((env.num_env,) + tuple(env.action_space.shape),
                   device=env.device)

    seen = {}
    obs = env.reset()
    for step in range(cfg.gallery_max_steps):
        obs, rew, done, info = env.step(act)

        # Per-env episode step counter (object has settled but not been pushed).
        steps = dcn(env.buffers['step'])
        names = env.scene.cur_names
        if names is None:
            continue

        # Capture only when there is at least one freshly-settled new object.
        fresh = [i for i in range(env.num_env)
                 if (steps[i] >= cfg.gallery_settle)
                 and (names[i] not in seen)
                 and (targets is None or names[i] in targets)]
        if not fresh:
            if targets is not None and len(seen) >= len(targets):
                break
            continue

        alt = img_env._wrap_obs(obs)
        rgb = _to_uint8_hwc(alt['color'])  # NHWC uint8
        for i in fresh:
            key = names[i].split('/')[-1]
            cv2.imwrite(str(out_dir / f'{key}.png'), rgb[i])
            seen[names[i]] = rgb[i]
            ic(len(seen), key)

        if targets is not None and len(seen) >= len(targets):
            break

    print(f'captured {len(seen)} object images -> {out_dir}')
    if targets is not None:
        missing = sorted(targets - set(seen.keys()))
        if missing:
            print(f'WARNING: {len(missing)} objects never captured '
                  f'(increase ++gallery_max_steps): {missing}')

    # Montage of all captured objects for convenience.
    if len(seen) > 0:
        imgs = [seen[k] for k in sorted(seen.keys())]
        montage = tile_images(np.stack(imgs, axis=0))
        cv2.imwrite(str(out_dir / '_gallery_montage.png'), montage)
        print(f'montage -> {out_dir / "_gallery_montage.png"}')


if __name__ == '__main__':
    main()
