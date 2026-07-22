#!/usr/bin/env python3

from typing import Tuple, Optional, Callable

from isaacgym import gymtorch
from isaacgym import gymapi
from pathlib import Path

from dataclasses import dataclass
from util.config import ConfigBase

from einops import rearrange
import numpy as np
import torch as th


from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv

from env.env.help.apply_camera_transform import ApplyCameraTransform
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper
from env.robot.cube_poker import CubePoker
from env.robot.ur5_fe import UR5FE
from models.common import merge_shapes

from util.path import ensure_directory
from util.torch_util import dcn
from util.vis.img import tile_images, to_hwc

import cv2
import imageio.v2 as imageio

from icecream import ic


class NvdrRecordEpisode(WrapperEnv):
    """
    Wrapper to record images from gym viewer.
    """

    @dataclass
    class Config(ConfigBase):
        record_dir: str = '/tmp/docker/record'
        episode_type: str = 'succ'
        img_size: Tuple[int, int] = (256, 256)
        cam_eye: Tuple[float, float, float] = (0.54, 0.0, 0.9)
        cam_at: Tuple[float, float, float] = (-0.2, 0.0, 0.4)
        # NOTE: render `visual mesh` for objects, by default.
        use_col: bool = True

        video_fps: int = 30

        # Cap videos written per subdir ('succ'/'fail', or '' when not 'both').
        # 0 = unlimited. Bounds disk/time when recording during a long eval.
        max_per_type: int = 0

    def __init__(self, cfg: Config, env: EnvIface, **kwds):
        super().__init__(env)
        self.cfg = cfg

        self.img_env = NvdrCameraWrapper(
            env,
            NvdrCameraWrapper.Config(
                img_size=cfg.img_size,

                eye=cfg.cam_eye,
                at=cfg.cam_at,

                use_flow=False,
                use_label=False,
                use_color=True,

                use_col=cfg.use_col,
                **kwds
            )
        )

        self._record_dir: Path = ensure_directory(
            cfg.record_dir)

        # self.__eps_count: int = 0
        self.eps_count_batch: list = [0] * env.num_env
        # Videos already written per subdir, for the max_per_type cap.
        self._written_per_type: dict = {}
        # Create a giant buffer for storing images.
        self.__images = np.zeros(
            merge_shapes(env.timeout, env.num_env, cfg.img_size, 3),
            dtype=np.uint8)
        self.__index = np.zeros((env.num_env,),
                                dtype=np.int32)
        self.project = ApplyCameraTransform()
        self.project.reset(self)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def __export_episode(self, images: np.ndarray, count: int, env_id: int = 0,
                         subdir: str = '', label: Optional[str] = None):
        if count <= 0:
            return
        out_dir = ensure_directory(
            self._record_dir / subdir /
            F'env_{env_id:04d}')
        ic(count)
        # for i in range(count):
        #     cv2.imwrite(F'{out_dir}/{i:04d}.png', images[i])

        # Tag the file with the object key so can/bottle/scale is identifiable.
        tag = '' if label is None else f'_{label}'
        video_path = out_dir / f'episode_{self.eps_count_batch[env_id]:04d}{tag}.mp4'
        # Write H.264/yuv420p directly (via imageio's bundled ffmpeg) so the mp4
        # plays in browsers / QuickTime / most players. OpenCV inside the
        # container can only encode 'mp4v', which many players reject.
        # Frames are RGB (camera color), which is what imageio expects.
        writer = imageio.get_writer(
            str(video_path),
            fps=self.cfg.video_fps,
            codec='libx264',
            pixelformat='yuv420p',
            ffmpeg_params=['-movflags', '+faststart'])
        for frame in images[:count]:
            writer.append_data(frame)
        writer.close()

        # self.__eps_count += 1
        self.eps_count_batch[env_id] +=1

    def __get_lines(self, lines) -> Tuple[np.ndarray, np.ndarray]:
        vss = []
        css = []
        nss = []
        for env_id in sorted(lines.keys()):
            # self.lines[env_id].append((num_lines, vertices.copy(), colors.copy()))
            vs = []
            cs = []
            ns = []
            for line in lines[env_id]:
                n, v, c = line
                vs.append(v)
                cs.append(c)
                ns.append(n)
            vss.append(np.concatenate(vs, axis=0).reshape(-1, 6))
            css.append(np.concatenate(cs, axis=0).reshape(-1, 3))
            nss.append(ns)

        if len(vss) <= 0:
            vss = np.empty((self.env.num_env, 0, 6))
            css = np.empty((self.env.num_env, 0, 3))
            nss = np.empty((self.env.num_env, 0))
        else:
            vss = np.stack(vss, axis=0)
            css = np.stack(css, axis=0)
            nss = np.stack(nss, axis=0)
        return (vss, css, nss)

    def __record(self, obs, rew, done, info):
        # Render & Query.
        alt = self.img_env._wrap_obs(obs)
        rgb_imgs = alt['color']  # NCHW

        # Convert.
        rgb_imgs_np = dcn(rgb_imgs)
        if rgb_imgs_np.dtype != np.uint8:
            rgb_imgs_np = (255 * rgb_imgs_np.clip(0.0, 1.0)).astype(np.uint8)
        rgb_imgs_np = to_hwc(rgb_imgs_np)
        done_np = dcn(done)

        # Draw all debug lines on our image as well (only when the gym is
        # wrapped by TrackDebugLines; the raw isaacgym Gym has no `.lines`).
        if getattr(self.env.gym, 'lines', None):
            vertices, colors, counts = self.__get_lines(self.env.gym.lines)
            vertices = vertices.reshape(self.env.num_env, -1, 3)
            vertices = dcn(self.project(vertices, self.cfg.img_size))
            lines = vertices.reshape(self.env.num_env, -1, 2, 2)
            lines_np = dcn(lines)

            def _to_cvpt(x):
                return tuple(int(e) for e in x)

            for i in range(self.env.num_env):
                for l in range(lines_np.shape[1]):
                    color = (255 *
                             (colors[i, l]).clip(0.0, 1.0)).astype(np.uint8)
                    cv2.line(
                        rgb_imgs_np[i],
                        _to_cvpt(lines_np[i, l, 0]),
                        _to_cvpt(lines_np[i, l, 1]),
                        _to_cvpt(color),
                        thickness=1
                    )

        # Clamp to the buffer length: an episode that runs the full `timeout`
        # can otherwise push __index past the (timeout-sized) buffer before the
        # done resets it.
        T = self.__images.shape[0]
        write_idx = np.minimum(self.__index, T - 1)
        self.__images[write_idx, th.arange(self.env.num_env)] = rgb_imgs_np
        self.__index = np.minimum(self.__index + 1, T)

        # Process episodes.
        # `both` records successes and failures into separate subdirs
        # in a single run; `succ`/`fail` keep the original behavior.
        record_both = (self.cfg.episode_type == 'both')
        sel = None
        if self.cfg.episode_type == 'succ':
            assert (('success' in info))
            sel = info['success']
        elif self.cfg.episode_type == 'fail':
            assert (('success' in info))
            sel = ~info['success']
        elif record_both:
            assert (('success' in info))

        for env_id in np.argwhere(done_np).ravel():
            if self.env.buffers['step'][env_id] < 30:
                continue
            if (sel is not None) and (not sel[env_id]):
                continue
            # Object key for this env (same source as CountCategoricalSuccess).
            label = None
            if getattr(self.env.scene, 'cur_names', None) is not None:
                label = self.env.scene.cur_names[env_id].split('/')[-1]
            subdir = ''
            if record_both:
                subdir = 'succ' if bool(info['success'][env_id]) else 'fail'
            # Enforce the per-type cap (0 = unlimited).
            if self.cfg.max_per_type > 0:
                if self._written_per_type.get(subdir, 0) >= self.cfg.max_per_type:
                    continue
                self._written_per_type[subdir] = \
                    self._written_per_type.get(subdir, 0) + 1
            self.__export_episode(
                self.__images[:, env_id],
                self.__index[env_id],
                env_id,
                subdir=subdir,
                label=label)

        # Reset counts for completed episodes.
        self.__index[done_np] = 0

    def step(self, *args, **kwds):
        # Step original env.
        obs, rew, done, info = self.env.step(*args, **kwds)
        # Record.
        self.__record(obs, rew, done, info)
        # Return original outputs.
        return (obs, rew, done, info)
