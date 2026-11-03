#!/usr/bin/env python3

import time

from isaacgym import gymtorch
from isaacgym import gymapi

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Iterable
from util.config import ConfigBase

import torch as th
import numpy as np

from env.env.iface import EnvIface
# from env.common import create_camera
from util.torch_util import dcn

from gym import spaces
import nvtx
import cv2, os
from multiprocessing import Process, Queue, Event
from queue import Empty as QueueEmpty

def wrap_flow_tensor(gym_tensor, offsets=None, counts=None):
    data = gym_tensor.data_ptr
    device = gym_tensor.device
    dtype = int(gym_tensor.dtype)
    shape = gym_tensor.shape
    shape = tuple(shape) + (2,)
    if offsets is None:
        offsets = tuple([0] * len(shape))
    if counts is None:
        counts = shape
    return gymtorch.wrap_tensor_impl(
        data, device, dtype, shape, offsets, counts)


class WithCamera:
    """
    Isaac Gym helper class for adding
    image-based observations to envs.
    """
    KEYS: Tuple[str, ...] = ('color', 'depth', 'label', 'flow')

    @dataclass
    class Config(ConfigBase):
        height: int = 512
        width: int = 512
        fov: float = 65.
        use_collision_geometry: bool = False
        use_color: bool = True
        use_depth: bool = True
        use_label: bool = False
        use_flow: bool = False

        use_dr: bool = False

        pos: Tuple[float, float, float] = (0.1, 0.1, 5)
        target: Tuple[float, float, float] = (0, 0, 0)
        rot: Tuple[float,...] = (0, 0, 0, 1)

        pos0: Tuple[float, float, float] = (0.458, -0.4, 0.74)
        target0: Tuple[float, float, float] = (0, 0, 0.5)
        rot0: Tuple[float,...] = (0, 0, 0, 1)
        pos1: Tuple[float, float, float] = (-0.238, 0.388, 0.694)
        target1: Tuple[float, float, float] = (0, 0, 0.5)
        rot1: Tuple[float,...] = (0, 0, 0, 1)
        pos2: Tuple[float, float, float] = (-0.408, -0.328, 0.706)
        target2: Tuple[float, float, float] = (0, 0, 0.5)
        rot2: Tuple[float,...] = (0, 0, 0, 1)

        use_transform: bool = False
        device: str = 'cuda:0'
        save_images: bool = True

        # NOTE: please enable all 3 cameras
        enable_cam0: bool = True
        enable_cam1: bool = True
        enable_cam2: bool = True

        enable_save_img_mp: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        prop = gymapi.CameraProperties()
        prop.height = cfg.height
        prop.width = cfg.width
        prop.enable_tensors = True
        prop.use_collision_geometry = cfg.use_collision_geometry
        # prop.near_plane = 0.01
        # prop.far_plane = 5.0
        if cfg.fov >0.:
            prop.horizontal_fov = cfg.fov

        self.prop = prop
        print(prop)
        self.buffers: Dict[str, th.Tensor] = {}
        self.sensors = []
        self.tensors = {}
        self.step_idx = 0

        # FIXME: reduce code duplications
        h, w = cfg.height, cfg.width

        self.use_maps = {
            'color': cfg.use_color,
            'depth': cfg.use_depth,
            'label': cfg.use_label,
            'flow': cfg.use_flow
        }
        self.space_maps = {
            'color': spaces.Box(0, 255, (h, w, 4), np.uint8),
            'depth': spaces.Box(0, np.inf, (h, w), np.float),
            'label': spaces.Box(0, np.inf, (h, w), np.int32),

            # NOTE: for now we're going to assume
            # no conversion?
            'flow': spaces.Box(0, np.inf, (h, w), np.int16)
        }
        self.observation_space = spaces.Dict({
            k: self.space_maps[k] for k in self.KEYS if self.use_maps[k]
        })

        # NOTE: always define these so callers can start/stop unconditionally;
        # they stay None when enable_save_img_mp is off.
        self.save_img_stop_event = None
        self.save_img_queue0 = self.save_img_queue1 = self.save_img_queue2 = None
        self.save_img_p0 = self.save_img_p1 = self.save_img_p2 = None
        if cfg.enable_save_img_mp:
            self.save_img_stop_event = Event()
            # daemon=True is load-bearing: multiprocessing's atexit handler
            # JOINS non-daemon children, and these workers block forever on an
            # empty queue, so a non-daemon writer wedges interpreter shutdown
            # indefinitely (observed: a 15-min run that never exited).
            self.save_img_queue0 = Queue()
            self.save_img_p0 = Process(target=self.save_images_mp, daemon=True,
                                    args=(self.save_img_queue0, self.save_img_stop_event))
            self.save_img_queue1 = Queue()
            self.save_img_p1 = Process(target=self.save_images_mp, daemon=True,
                                    args=(self.save_img_queue1, self.save_img_stop_event))
            self.save_img_queue2 = Queue()
            self.save_img_p2 = Process(target=self.save_images_mp, daemon=True,
                                    args=(self.save_img_queue2, self.save_img_stop_event))

    def start_writers(self):
        """Start the image-writer processes, if any. No-op when
        enable_save_img_mp is off (the Process objects are then None)."""
        for p in (self.save_img_p0, self.save_img_p1, self.save_img_p2):
            if p is not None:
                p.start()

    def stop_writers(self, timeout: float = 2.0):
        """Signal the writers to drain and exit. Safe to call more than once."""
        if self.save_img_stop_event is not None:
            self.save_img_stop_event.set()
        for p in (self.save_img_p0, self.save_img_p1, self.save_img_p2):
            if p is not None and p.is_alive():
                p.join(timeout)
                if p.is_alive():
                    p.terminate()

    ###@nvtx.annotate("WithCamera.save_images_mp()")
    @staticmethod
    def save_images_mp(queue, stop_event):
        # A bare blocking get() can never observe stop_event once the queue
        # goes quiet; poll with a timeout so shutdown is actually possible.
        while not stop_event.is_set():
            try:
                img, dir = queue.get(timeout=0.5)
            except QueueEmpty:
                continue
            try:
                cv2.imwrite(dir, img)
            except Exception as e:
                print(f"Failed to save {dir}: {e}")

    def setup(self, env):
        """
        * load assets.
        * allocate buffers related to {scene, robot, task}.
        """
        cfg = self.cfg

        self.num_env = env.num_env
        env.gym.step_graphics(env.sim)

        for e in env.envs:
            if cfg.enable_cam0:
                camera, tensors = self.create_camera(env.gym, env.sim, e, env,
                                                        cfg.pos0, cfg.target0, cfg.rot0)
                self.sensors.append(camera)
                for k, v in tensors.items():
                    if k not in self.tensors:
                        self.tensors[k] = []
                    self.tensors[k].append(v)
            if cfg.enable_cam1:
                camera, tensors = self.create_camera(env.gym, env.sim, e, env,
                                                     cfg.pos1, cfg.target1, cfg.rot1)
                self.sensors.append(camera)
                for k, v in tensors.items():
                    if k not in self.tensors:
                        self.tensors[k] = []
                    self.tensors[k].append(v)
            if cfg.enable_cam2:
                camera, tensors = self.create_camera(env.gym, env.sim, e, env,
                                                     cfg.pos2, cfg.target2, cfg.rot2)
                self.sensors.append(camera)
                for k, v in tensors.items():
                    if k not in self.tensors:
                        self.tensors[k] = []
                    self.tensors[k].append(v)

        h, w = cfg.height, cfg.width
        img_shape = (h, w)
        n_cams_per_env = cfg.enable_cam0 + cfg.enable_cam1 + cfg.enable_cam2 

        if cfg.use_color:
            self.buffers['color'] = (
                th.empty(
                    (n_cams_per_env * self.num_env,) + img_shape + (4,),
                    device=cfg.device,
                    dtype=th.uint8))
        if cfg.use_depth:
            self.buffers['depth'] = (
                th.empty(
                    (n_cams_per_env * self.num_env,) + img_shape,
                    device=cfg.device,
                    dtype=th.float32))
        if cfg.use_label:
            self.buffers['label'] = (
                th.empty(
                    (n_cams_per_env * self.num_env,) + img_shape,
                    device=cfg.device,
                    dtype=th.int32))
        if cfg.use_flow:
            # NOTE: 2x 16-bit signed int??
            self.buffers['flow'] = (
                th.empty(
                    (n_cams_per_env * self.num_env,) + img_shape + (2,),
                    device=cfg.device,
                    dtype=th.int16))
    
        if cfg.save_images:
            if not os.path.exists("/tmp/docker/record_rgb_cam"):
                os.mkdir("/tmp/docker/record_rgb_cam")
            if not os.path.exists("/tmp/docker/record_rgb_cam/cam_0"):
                os.mkdir("/tmp/docker/record_rgb_cam/cam_0")
            if not os.path.exists("/tmp/docker/record_rgb_cam/cam_1"):
                os.mkdir("/tmp/docker/record_rgb_cam/cam_1")
            if not os.path.exists("/tmp/docker/record_rgb_cam/cam_2"):
                os.mkdir("/tmp/docker/record_rgb_cam/cam_2")
            if not os.path.exists("/tmp/docker/record_rgb_cam/cam_all"):
                os.mkdir("/tmp/docker/record_rgb_cam/cam_all")

    def create_camera(self, gym, sim, env, envs,
                      pos: Tuple[float, float, float] = (0.1, 0.1, 5),
                      target: Tuple[int, int, int] = (0, 0, 0),
                      rot: Tuple[float,...] = (0, 0, 0, 1)):
        cfg = self.cfg
        camera = gym.create_camera_sensor(env, self.prop)
        if camera is None or camera < 0:
            # Isaac Gym returns -1 when the sim has no graphics device
            # (create_sim with graphics_device_id=-1). Fail here, where the
            # cause is nameable; otherwise it surfaces much later as a "could
            # not find camera with handle -1" log line and then a th.stack
            # TypeError on a list of Nones.
            raise RuntimeError(
                "create_camera_sensor returned -1: this sim has no graphics "
                "device. Isaac Gym camera sensors require graphics_device_id "
                ">= 0 -- see train_ppo_arm.py, which forces -1 unless "
                "use_viewer or use_isaac_camera is set.")
        x, y, z = pos
        z_offset = envs.scene.table_dims[0, 2]
        x_offset = (envs.scene.table_pos[0, 0] - 
                    0.5 * envs.scene.table_dims[0, 0]-
                    envs.robot.cfg.keepout_radius)
        if cfg.use_transform:
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(x+x_offset, y, z+z_offset)
            transform.r = gymapi.Quat(*rot)
            gym.set_camera_transform(camera, env, transform)
        else:
            gym.set_camera_location(camera,
                                    env,
                                    gymapi.Vec3(x, y, z),
                                    gymapi.Vec3(*target))
        # print(np.linalg.inv(np.matrix(gym.get_camera_view_matrix(sim, env, camera))))
        USE_MAPS = {
            'color': cfg.use_color,
            'depth': cfg.use_depth,
            'label': cfg.use_label,
            'flow': cfg.use_flow}

        TYPE_MAPS = {'color': gymapi.IMAGE_COLOR,
                     'depth': gymapi.IMAGE_DEPTH,
                     'label': gymapi.IMAGE_SEGMENTATION,
                     'flow': gymapi.IMAGE_OPTICAL_FLOW}

        tensors = {}
        for key in self.KEYS:
            if not USE_MAPS.get(key, False):
                continue
            descriptor = gym.get_camera_image_gpu_tensor(
                sim, env, camera, TYPE_MAPS[key]
            )
            if key == 'flow':
                tensor = wrap_flow_tensor(descriptor)
            else:
                tensor = gymtorch.wrap_tensor(descriptor)
            # print(key, tensor)
            # # assert (tensor is not None)
            tensors[key] = tensor
        return (camera, tensors)

    ###@nvtx.annotate("WithCamera.reset()")
    def reset(self, gym, env,
              env_ids: Optional[Iterable[int]] = None):
        cfg = self.cfg

        if env_ids is None:
            env_ids = range(self.num_env)

        # [0] Camera pose randomization.
        # cameras = [env.sensors['scene'][i]['camera']
        #            for i in dcn(env_ids)]
        cameras = self.sensors

        # TODO: set camera position only during
        # the first reset?

        # TODO: figure out how to handle
        # camera-position DR...
        if cfg.use_dr:
            for env_id, camera in zip(env_ids, cameras):
                # Randomize sensor camera position.
                x = cfg.table_pos[0] + np.random.uniform(-1.0, 1.0)
                y = cfg.table_pos[1] + np.random.uniform(-1.0, 1.0)
                z = cfg.table_pos[2] + 0.5 + np.random.uniform(0.0, 1.0)
                x, y, z = cfg.pos
                if cfg.use_transform:
                    transform = gymapi.Transform()
                    transform.p = gymapi.Vec3(x, y, z)
                    transform.r = gymapi.Quat(*cfg.rot)
                    gym.set_camera_transform(camera, env.envs[env_id], transform)
                else:
                    gym.set_camera_location(camera,
                                            env.envs[env_id],
                                            gymapi.Vec3(x, y, z),
                                            gymapi.Vec3(*cfg.target))

    @nvtx.annotate("WithCamera.step()")
    def step(self, env):
        cfg = self.cfg
        # t0 = (time.time())

        with nvtx.annotate("step_graphics()"):
            env.gym.step_graphics(env.sim)
           # env.gym.fetch_results(env.sim, True)

        with nvtx.annotate("render_all()"):
            env.gym.render_all_camera_sensors(env.sim)

        with nvtx.annotate("access()"):
            # Access and convert all image-related tensors.
            # TODO: the image access and related rendering utilities
            # should only be conditioned on image-based environments.
            with nvtx.annotate("start()"):
                env.gym.start_access_image_tensors(env.sim)

            if cfg.use_color:
                color_tensors = self.tensors['color']
                th.stack(color_tensors, out=self.buffers['color'])

            if cfg.use_depth:
                depth_tensors = self.tensors['depth']
                th.stack(depth_tensors, out=self.buffers['depth'])

            if cfg.use_label:
                label_tensors = self.tensors['label']
                th.stack(label_tensors, out=self.buffers['label'])

            if cfg.use_flow:
                flow_tensors = self.tensors['flow']
                th.stack(flow_tensors, out=self.buffers['flow'])

            with nvtx.annotate("end()"):
                env.gym.end_access_image_tensors(env.sim)

        # t1 = (time.time())
        # dt = t1 - t0
        # print(F'dt={dt}')
        return self.buffers

    ###@nvtx.annotate("WithCamera.save_images()")

    def save_images(self, 
                    env_ids: Optional[Iterable[int]] = None,
                    image_tensors = None ):

        cfg = self.cfg

        # Nothing to do when image dumping is off, or when there is no colour
        # buffer to dump. Previously this ran unconditionally (base.py calls it
        # every step), so save_images=False did not actually disable anything
        # and use_color=False raised KeyError.
        if (not cfg.save_images) or (self.buffers.get('color') is None):
            self.step_idx += 1
            return {'dir': [], 'img': []}

        if image_tensors == None:
            camera_rgba_tensor = self.buffers['color'] # 'color': (n_cams*n_envs,H,W,4), 'depth': (n_cams*n_envs,H,W,1)
            camera_depth_tensor = self.buffers['depth']
        else:
            camera_rgba_tensor = image_tensors['color']
            camera_depth_tensor = image_tensors['depth']
        if env_ids == None:
            env_ids = [0]
        
        n_cams_per_env = cfg.enable_cam0 + cfg.enable_cam1 + cfg.enable_cam2

        # NOTE: capture images in the first env only
        # TODO: record successful or failed episodes?
        
        image_lists={'dir': [], 'img': []}

        for env_id in env_ids:
            # saving img: ~20ms per env
            # if we use 4 sim envs and save imgs from env_0, the evaluation speed is ~10 it/s
            # if we disable saving imgs, the speed is ~12.5 it/s
            # use multiprocessing ~12it/s ?
            # TODO: multiprocessing
            # t1 = time.time()
            if cfg.enable_cam0:
                rgba_img0 = camera_rgba_tensor[0 + n_cams_per_env * env_id].clone().cpu().numpy()
                rgb_img0 = cv2.cvtColor(rgba_img0, cv2.COLOR_RGBA2BGR)
                rgb_filename0 = F"/tmp/docker/record_rgb_cam/cam_0/{self.step_idx:04d}.png" 
                #cv2.imwrite(rgb_filename0,rgb_img0)
            
            if cfg.enable_cam1:
                rgba_img1 = camera_rgba_tensor[1 + n_cams_per_env * env_id].clone().cpu().numpy()
                rgb_img1 = cv2.cvtColor(rgba_img1, cv2.COLOR_RGBA2BGR)
                rgb_filename1 = F"/tmp/docker/record_rgb_cam/cam_1/{self.step_idx:04d}.png" 
                #cv2.imwrite(rgb_filename1,rgb_img1)

            if cfg.enable_cam2:
                rgba_img2 = camera_rgba_tensor[2 + n_cams_per_env * env_id].clone().cpu().numpy()
                rgb_img2 = cv2.cvtColor(rgba_img2, cv2.COLOR_RGBA2BGR)
                rgb_filename2 = F"/tmp/docker/record_rgb_cam/cam_2/{self.step_idx:04d}.png" 
                #cv2.imwrite(rgb_filename2,rgb_img2)
            
            # Only append cameras that actually exist -- appending
            # rgb_filename1/2 unconditionally raised NameError whenever
            # enable_cam1/enable_cam2 were off, which is why this class
            # effectively required all three cameras.
            if cfg.enable_cam0:
                image_lists['dir'].append(rgb_filename0)
                image_lists['img'].append(rgb_img0)
            if cfg.enable_cam1:
                image_lists['dir'].append(rgb_filename1)
                image_lists['img'].append(rgb_img1)
            if cfg.enable_cam2:
                image_lists['dir'].append(rgb_filename2)
                image_lists['img'].append(rgb_img2)

            #t2 = time.time()
            #print(f'Saving images cost {t2-t1} secs.')
        self.step_idx += 1

        return image_lists

    def enqueue_images_mp(self, image_lists):
        queues = [q for q in (self.save_img_queue0,
                              self.save_img_queue1,
                              self.save_img_queue2) if q is not None]
        if not queues:
            return
        imgs = image_lists.get('img', [])
        dirs = image_lists.get('dir', [])
        for i, (img, dir) in enumerate(zip(imgs, dirs)):
            queues[i % len(queues)].put((img, dir))