#!/usr/bin/env python3
"""Generate GAP-style ground-truth 3D point-flow tracks from DyWA rollouts.

Port of third_party/GAP/scripts/generate_gt_tracks.py's *method* to Isaac Gym.
No tracking, no drift: every visible patch centre is unprojected to a 3D point,
expressed in the frame of the rigid body that owns it, and pushed forward
through that body's recorded future poses.

Design decisions (see _plan_gap_tracks.md):

  * Camera: a DEDICATED NvdrCameraWrapper with its own obs keys (gap_color /
    gap_depth / gap_label). The student's point-cloud camera is untouched, so
    this can run against the existing pipeline without disturbing it.
    Isaac Gym's own camera cannot be used -- it never renders the object mesh.
  * Frame: nvdr's unproject() returns ENV-LOCAL coordinates, the same frame as
    tensors['body'], so there is NO env-origin correction anywhere here.
    (Measured by _probe_nvdr_frame.py. Note this is the OPPOSITE of the Isaac
    Gym camera, whose view matrix is global -- do not copy that code path.)
  * Ownership: nvdr labels are `1 + rigid_body_index`, 0 = background
    (render_nvdr.py:645, confirmed by _probe_nvdr_frame.py). Every labelled
    body is tracked -- object, table and all arm links -- so gripper-link
    tracks come for free.
  * Resolution: rendered natively at 238x322 so the 17x23=391 patch grid needs
    no resize, and patch centres land on EXACT integer pixels (14*i+7). This
    removes both the resampler mismatch and the sub-pixel depth/label sampling
    inconsistency that GAP's own script carries.
  * Episode boundaries: a done env is reset during the NEXT step's
    pre_physics, so a window spanning that boundary would transform an anchor
    through a teleport. Frames whose H-step window crosses a reset are dropped.

Output: GAP-flat HDF5 using the key names GAP's CONSUMERS expect
(`gt_3d_track` / `track_valid_mask_gt`), not the ones its generator writes.

Env knobs:
    GAP_OUT           output .hdf5
    GAP_NUM_EPISODES  stop after this many completed episodes (default 50)
    GAP_MAX_STEPS     hard cap on env steps (default 20000)
    GAP_MAX_EP_LEN    drop episodes longer than this; 0 = no cap (default 0)
    GAP_MIN_EP_LEN    drop episodes shorter than this (default 4, as TTT)
    GAP_SUCCESS_ONLY  1/0, keep only successful episodes (default 1, as TTT)
    GAP_TRACK_HORIZON H, future steps per track (default 16)
    GAP_DINO          1/0, extract DINOv2 features (default 1)
    GAP_STORE_IMAGE   1/0, store the RGB frames (default 1)
    GAP_DINO_WEIGHTS  safetensors path
    GAP_EMIT_TAIL     1/0, keep the final H frames of each episode with an
                      all-False track mask so the goal is visible (default 1)
"""

from isaacgym import gymtorch  # noqa: F401  (before torch)
from isaacgym import gymapi    # noqa: F401

import os
import sys
import time
from collections import deque
from dataclasses import replace

import numpy as np
import torch as th
import h5py

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from util.math_util import quat_rotate
from util.torch_util import dcn
from env.util import set_seed
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper, unproject

from train_ppo_arm import (
    AddTensorboardWriter, setup as setup_logging, load_env, load_agent,
)
from rma_env import setup_rma_env_v2
from train_rma import Config as RMAConfig, update_net_cfg, get_config_path

# ── patch grid (GAP: DINOv2 ViT-L/14, 17x23 = 391 tokens) ──────────────────
PATCH = 14
IMG_H, IMG_W = 238, 322
GRID_H, GRID_W = IMG_H // PATCH, IMG_W // PATCH      # 17, 23
N_PATCH = GRID_H * GRID_W                             # 391
FEAT_DIM = 1024

OUT = os.environ.get('GAP_OUT', '/home/user/DyWA/output/gap/tracks.hdf5')
NUM_EPISODES = int(os.environ.get('GAP_NUM_EPISODES', 50))
MAX_STEPS = int(os.environ.get('GAP_MAX_STEPS', 20000))
# Length policy is deliberately IDENTICAL to collect_teacher_dataset.py (TTT):
# a MIN_EP_LEN floor and NO upper cap. An upper cap is not a neutral memory
# knob -- episodes that hit the task timeout (300 steps -> ~283 frames) are
# exactly the ones it discards, so it silently biases the dataset. 0 disables.
MAX_EP_LEN = int(os.environ.get('GAP_MAX_EP_LEN', 0))
MIN_EP_LEN = int(os.environ.get('GAP_MIN_EP_LEN', 4))
# Keep only episodes ending in success, as the TTT collector does.
SUCCESS_ONLY = os.environ.get('GAP_SUCCESS_ONLY', '1') == '1'
HORIZON = int(os.environ.get('GAP_TRACK_HORIZON', 16))
USE_DINO = os.environ.get('GAP_DINO', '1') == '1'
STORE_IMAGE = os.environ.get('GAP_STORE_IMAGE', '1') == '1'
DINO_WEIGHTS = os.environ.get(
    'GAP_DINO_WEIGHTS', '/home/user/.cache/pkm/dinov2/model.safetensors')
# A 16-step window cannot legitimately move a surface point this far; anything
# beyond it means the window crossed an episode reset (teleport).
TELEPORT_M = float(os.environ.get('GAP_TELEPORT_M', 1.0))
# A frame can only carry a track once its full H-step future exists, so the
# last H frames of every episode would otherwise be dropped -- and those are
# exactly the frames where the object reaches the goal. Emit them with
# track_valid_mask_gt=False (images/poses kept, no invented supervision).
EMIT_TAIL = os.environ.get('GAP_EMIT_TAIL', '1') == '1'


# Set from the first teacher_action seen; the writer's spec needs it before the
# first episode is flushed, which always happens after several steps.
ACT_CMD_DIM = 20


def _v3(name, default):
    s = os.environ.get(name)
    return tuple(float(x) for x in s.split(',')) if s else default


CAM_FOV_DEG = float(os.environ.get('GAP_CAM_FOV_DEG', 45.0))
CAM_EYE = _v3('GAP_CAM_EYE', (0.60, -0.05, 0.85))
CAM_AT = _v3('GAP_CAM_AT', (-0.12, 0.0, 0.45))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_raw_phys(env, E: int, device) -> th.Tensor:
    """[object mass, table friction, object friction, hand friction,
    restitution] per env, in PHYSICAL units, queried straight from Isaac Gym."""
    gym = env.gym
    out = th.zeros(E, 5, dtype=th.float32, device=device)
    try:
        handles = dcn(env.scene.cur_handles)
    except Exception:                                     # noqa: BLE001
        return out
    for k, (e, h) in enumerate(zip(env.envs, handles)):
        try:
            props = gym.get_actor_rigid_body_properties(e, int(h))
            out[k, 0] = float(props[0].mass)
            shp = gym.get_actor_rigid_shape_properties(e, int(h))
            if shp:
                out[k, 2] = float(shp[0].friction)
                out[k, 4] = float(shp[0].restitution)
        except Exception:                                 # noqa: BLE001
            pass

    def _actor_friction(group, name, col, last=False):
        try:
            entry = env.actors[group][0]
            handle = entry[name] if isinstance(entry, dict) else entry
            if isinstance(handle, (list, tuple)):
                handle = handle[0]
            for k, e in enumerate(env.envs):
                shp = gym.get_actor_rigid_shape_properties(e, int(handle))
                if shp:
                    out[k, col] = float(shp[-1 if last else 0].friction)
        except Exception:                                 # noqa: BLE001
            pass

    _actor_friction('scene', 'table', 1)
    _actor_friction('robot', 'robot', 3, last=True)
    return out


def quat_conj(q: th.Tensor) -> th.Tensor:
    """Conjugate of a unit quaternion in Isaac Gym's (x, y, z, w) order."""
    return th.cat([-q[..., :3], q[..., 3:]], dim=-1)


def make_cam_cfg():
    """The one definition of the track camera.

    Anything that needs to re-project these tracks (e.g. visualize_gap_tracks)
    must use the SAME geometry, so it lives here and is also written into the
    output file's attrs.

    FRAMING MATTERS MORE THAN IT LOOKS. nvdr's default fov is 90 deg, which at
    this eye distance spans ~1.8 m while the region of interest (table top +
    reachable workspace) is ~0.8 m. That wastes most of the 391 patches on
    black background and leaves ~1.6 patches per frame on the object -- almost
    no supervision on the thing the task is about. The default below is
    tightened accordingly; measure with `_verify_gap_tracks.py` /
    `visualize_gap_tracks.py` before changing it.
    """
    return NvdrCameraWrapper.Config(
        img_size=(IMG_H, IMG_W),
        use_color=True, use_depth=True, use_label=True,
        use_col=False,          # visual mesh, not collision
        hide_arm=False,         # arm links are tracked like any other body
        track_object=False,     # fixed scene camera -> stable patch grid
        eye=CAM_EYE, at=CAM_AT, up=(0.0, 0.0, 1.0),
        fov=float(np.deg2rad(CAM_FOV_DEG)),
        key_color='gap_color', key_depth='gap_depth', key_label='gap_label',
    )


def load_dino(device):
    """DINOv2 ViT-L/14 with registers, via timm.

    GAP's vendored facebookresearch/dinov2 does NOT run here: it uses PEP-604
    syntax (needs py>=3.10) and F.scaled_dot_product_attention (needs torch
    >=2.0); this container is py3.8 / torch 1.11. timm's implementation of the
    same LVD-142M weights works.
    """
    import timm
    from safetensors.torch import load_file

    os.environ.setdefault('XFORMERS_DISABLED', '1')
    model = timm.create_model('vit_large_patch14_reg4_dinov2',
                              pretrained=False, num_classes=0,
                              dynamic_img_size=True)
    missing, unexpected = model.load_state_dict(load_file(DINO_WEIGHTS),
                                                strict=False)
    bad = [k for k in missing if not k.startswith('head')]
    assert not bad, f'DINOv2 weights missing non-head keys: {bad[:5]}'
    return model.eval().to(device)


@th.no_grad()
def dino_features(model, color, mean, std):
    """color: (E,3,H,W) float in ~[0,1] -> (E, N_PATCH, 1024) float16.

    The frames are greyscale-in-3-channels (DyWA's meshes carry no materials),
    which is accepted: DINOv2 keys off shape and shading here.
    """
    x = ((color.clamp(0.0, 1.0) - mean) / std)
    feats = model.forward_features(x)
    patch = feats[:, model.num_prefix_tokens:, :]     # drop CLS + 4 registers
    assert patch.shape[1] == N_PATCH, \
        f'expected {N_PATCH} patch tokens, got {patch.shape[1]}'
    return patch.half()


class EpisodeWriter:
    """GAP-flat HDF5, one contiguous run of frames per episode."""

    def __init__(self, path, cam_cfg=None, T_cam=None):
        self.cam_cfg = cam_cfg
        self.T_cam = T_cam
        self._init_path(path)

    def _init_path(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = h5py.File(path, 'w')
        self.ds = {}
        self.T = 0
        self.n_ep = 0

    def _spec(self):
        spec = {
            'image_patch_centre_3d_position': ((N_PATCH, 3), np.float32),
            'gt_3d_track': ((N_PATCH, HORIZON, 3), np.float32),
            'track_valid_mask_gt': ((N_PATCH, HORIZON), np.bool_),
            'patch_owner_body': ((N_PATCH,), np.int16),
            'robot_state': ((7,), np.float32),
            'action': ((7,), np.float32),
            # Task goal pose for the object (pos ‖ quat_xyzw), env-local, same
            # frame as the tracks. Constant within an episode; stored per frame
            # so a shuffled frame-level loader still sees it.
            'goal_pose': ((7,), np.float32),
            # The per-episode domain-randomisation draw that generated this
            # rollout: [object mass, table friction, object friction, hand
            # friction, restitution]. Without it the dynamics behind each
            # episode are unrecoverable from the file -- which matters for any
            # downstream work that conditions on or adapts to hidden physics.
            'phys_params': ((5,), np.float32),
            # The POLICY's action, raw normalised mu as emitted (DyWA's 20-D
            # cpos_n action: 3 dpos, 3 dori axis-angle, 7 joint KP, 7 joint KD).
            # `action` above is a different thing entirely -- the REALIZED next
            # EEF pose read back out of the sim. Keep both: one is the command,
            # the other is what the arm actually did.
            'action_cmd': ((ACT_CMD_DIM,), np.float32),
            # Object pose at this frame, so a consumer can compute the
            # object->goal residual without re-deriving it from the tracks.
            'object_pose': ((7,), np.float32),
        }
        if STORE_IMAGE:
            spec['image'] = ((IMG_H, IMG_W, 3), np.uint8)
        if USE_DINO:
            spec['dino_features'] = ((1, N_PATCH, FEAT_DIM), np.float16)
        return spec

    def _init(self):
        # lzf, not gzip. Measured on this data: gzip-4 costs 26.4 ms/frame and
        # caps the whole collector at 38 frames/s; lzf costs 3.3 ms/frame for a
        # file only 11% larger (ratio 0.77 vs 0.69). The bulk is float16 DINOv2
        # features, which barely compress either way.
        for k, (shape, dt) in self._spec().items():
            self.ds[k] = self.f.create_dataset(
                k, shape=(0,) + shape, maxshape=(None,) + shape, dtype=dt,
                chunks=(1,) + shape, compression='lzf')
        self.ds['episode_ends'] = self.f.create_dataset(
            'episode_ends', shape=(0,), maxshape=(None,), dtype=np.int64)
        # Outcome is NOT implied by anything else in the file: the collector
        # keeps every episode that terminates, success or not.
        self.ds['episode_success'] = self.f.create_dataset(
            'episode_success', shape=(0,), maxshape=(None,), dtype=np.bool_)
        self.ds['episode_length'] = self.f.create_dataset(
            'episode_length', shape=(0,), maxshape=(None,), dtype=np.int64)

    def write_episode(self, frames, success=False):
        if not frames:
            return 0
        if not self.ds:
            self._init()
        n = len(frames)
        for k in self._spec():
            arr = np.stack([fr[k] for fr in frames], axis=0)
            self.ds[k].resize(self.T + n, axis=0)
            self.ds[k][self.T:] = arr
        self.T += n
        self.n_ep += 1
        self.ds['episode_ends'].resize(self.n_ep, axis=0)
        self.ds['episode_ends'][self.n_ep - 1] = self.T
        self.ds['episode_success'].resize(self.n_ep, axis=0)
        self.ds['episode_success'][self.n_ep - 1] = bool(success)
        self.ds['episode_length'].resize(self.n_ep, axis=0)
        self.ds['episode_length'][self.n_ep - 1] = n
        self.f.flush()          # crash safety: each episode persisted
        return n

    def close(self):
        # Camera geometry, so re-projection never has to guess or re-derive it.
        if getattr(self, 'act_lo', None) is not None:
            self.f.attrs['act_lo'] = np.asarray(self.act_lo, np.float32)
            self.f.attrs['act_hi'] = np.asarray(self.act_hi, np.float32)
        if self.cam_cfg is not None:
            c = self.cam_cfg
            self.f.attrs['cam_eye'] = np.asarray(c.eye, np.float32)
            self.f.attrs['cam_at'] = np.asarray(c.at, np.float32)
            self.f.attrs['cam_up'] = np.asarray(c.up, np.float32)
            self.f.attrs['cam_fov'] = float(c.fov)
            self.f.attrs['cam_z_near'] = float(c.z_near)
            self.f.attrs['cam_z_far'] = float(c.z_far)
        if self.T_cam is not None:
            # world->view 4x4; p_cam = T_cam[:3,:3] @ p_world + T_cam[:3,3]
            self.f.attrs['T_cam'] = np.asarray(self.T_cam, np.float32)
        self.f.attrs['patch_grid'] = (GRID_H, GRID_W)
        self.f.attrs['track_horizon'] = HORIZON
        self.f.attrs['img_size'] = (IMG_H, IMG_W)
        self.f.attrs['note'] = (
            'single-arm 7-D robot_state/action (EEF pose xyz+quat); '
            'GAP is bimanual 16-D. env-local coordinates.')
        self.f.close()


@hydra_cli(config_path=get_config_path(), config_name='train_rl')
def main(cfg: RMAConfig):
    cfg.project = 'rma'
    cfg = recursive_replace_map(cfg, {'finalize': True})
    cfg = replace(cfg, train_student_policy=False, dagger=True)
    # Record the DR draw per episode; without this the physics that produced
    # each rollout cannot be recovered from the dataset.
    cfg = replace(cfg, add_phys_params=True)
    if cfg.global_device is not None:
        th.cuda.set_device(cfg.global_device)
    path = setup_logging(cfg)
    set_seed(cfg.env.seed)

    cfg, env = load_env(cfg, path, freeze_env=True, check_viewer=False)
    env.unwrap(target=AddTensorboardWriter).set_writer(None)
    cfg = replace(cfg, net=update_net_cfg(cfg.net, env, cfg.state_net_blocklist))
    teacher = load_agent(cfg, env, None, None)
    teacher.eval()
    env = setup_rma_env_v2(cfg, env, teacher,
                           state_size=128, is_student=False, dagger=True)

    cam_cfg = make_cam_cfg()
    img_env = NvdrCameraWrapper(env, cam_cfg)

    device = env.device
    E = env.num_env
    n_body = env.gym.get_env_rigid_body_count(env.envs[0])
    tip = env.robot.tip_body_indices.long()

    print(f'\n[gap] envs={E} bodies/env={n_body} horizon={HORIZON} '
          f'grid={GRID_H}x{GRID_W}={N_PATCH} dino={USE_DINO}')

    dino = mean = std = None
    if USE_DINO:
        dino = load_dino(device)
        mean = th.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = th.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        print(f'[gap] DINOv2 loaded from {DINO_WEIGHTS}')

    # exact-integer patch-centre pixels: (k + 0.5) * 14 == 14k + 7
    rows = th.arange(GRID_H, device=device) * PATCH + PATCH // 2
    cols = th.arange(GRID_W, device=device) * PATCH + PATCH // 2

    from env.env.help.with_nvdr_camera import camera_matrices
    T_cam, _ = camera_matrices(cam_cfg, 1, device)
    writer = EpisodeWriter(OUT, cam_cfg=cam_cfg,
                           T_cam=dcn(T_cam.reshape(4, 4)))
    # action_cmd is stored as the policy's RAW normalised mu, so record the
    # physical bounds it maps onto: physical = clamp(mu, -1, 1) * scale + centre
    # with scale=(hi-lo)/2, centre=(hi+lo)/2 (normalize_env.py:_wrap_act).
    # Without these the stored numbers are uninterpretable in metres/N.
    try:
        from env.env.wrap.normalize_env import NormalizeEnv
        nz = img_env.unwrap(target=NormalizeEnv)
        writer.act_lo, writer.act_hi = dcn(nz.lo), dcn(nz.hi)
        print('[gap] action bounds: dpos +-%.3g m  dori +-%.3g rad  '
              'KP [%.3g, %.3g]  KD [%.3g, %.3g]  (dim %d)'
              % (writer.act_hi[0], writer.act_hi[3], writer.act_lo[6],
                 writer.act_hi[6], writer.act_lo[13], writer.act_hi[13],
                 len(writer.act_hi)))
    except Exception as ex:                                   # noqa: BLE001
        writer.act_lo = writer.act_hi = None
        print('[gap] WARNING: could not read action bounds (%s) -- action_cmd '
              'will be stored but not unnormalisable' % ex)
    hist = deque()                       # rolling window of the last H+1 steps
    ep_start = np.zeros(E, dtype=np.int64)     # step index of episode start
    pending = [[] for _ in range(E)]           # emitted frames, per env
    n_ep_done = 0
    n_frames = 0
    n_dropped = 0
    n_teleport = 0
    n_succ = 0
    n_tail = 0
    n_rejected = n_rej_fail = n_rej_short = n_rej_long = 0

    obs = img_env.reset()
    t_start = time.time()
    for step in range(MAX_STEPS):
        # ---- capture the current frame -------------------------------------
        depth = obs['gap_depth']                       # (E,1,H,W) >0 = valid
        label = obs['gap_label']                       # (E,1,H,W) 1+body_idx
        color = obs['gap_color']                       # (E,3,H,W)

        pts = unproject(depth, cam_cfg)                # (E,3,H,W) ENV-LOCAL
        anchors = pts[:, :, rows][:, :, :, cols]       # (E,3,GH,GW)
        anchors = anchors.reshape(E, 3, N_PATCH).permute(0, 2, 1)   # (E,N,3)

        lab_p = label[:, 0][:, rows][:, :, cols].reshape(E, N_PATCH)
        dep_p = depth[:, 0][:, rows][:, :, cols].reshape(E, N_PATCH)

        # label is float32 and interpolates across silhouette edges; keep only
        # patches sitting cleanly inside one body.
        lab_r = th.round(lab_p)
        clean = (lab_p - lab_r).abs() < 1e-3
        valid = clean & (dep_p > 0) & (lab_r > 0)
        # label = 1 + body_index. `owner` is -1 wherever the patch is invalid
        # (background/edge) so a consumer that forgets to apply the mask gets an
        # obviously-wrong index rather than silently attributing background to
        # body 0 (the table). `gather` needs a clamped copy.
        owner = th.where(valid, lab_r.long() - 1,
                         th.full_like(lab_r, -1, dtype=th.long))
        owner_idx = owner.clamp(0, n_body - 1)

        body = env.tensors['body'].reshape(E, n_body, 13)[..., :7]   # (E,nb,7)
        eef = env.tensors['body'][tip, :7].reshape(E, 7)
        obj_pose = env.tensors['root'][env.scene.cur_ids.long(), :7]  # (E,7) raw
        # RAW physics, straight from the sim -- NOT obs['phys_params'], which
        # the env normalizer standardises (it is absent from the identity
        # `constlist`, so it comes out with negative "masses"). Same trap as
        # obs['object_state'].
        phys = read_raw_phys(env, E, device)
        goal = getattr(env.task, 'goal', None)
        if goal is None:
            goal_pose = th.zeros_like(obj_pose)
        else:
            goal = goal.reshape(E, -1)
            if goal.shape[-1] >= 7:
                goal_pose = goal[:, :7].clone()
            else:   # position-only goal: reuse the object's orientation
                goal_pose = th.cat([goal[:, :3], obj_pose[:, 3:7]], dim=-1)

        feats = None
        if USE_DINO:
            feats = dino_features(dino, color, mean, std)             # (E,N,D)

        hist.append({
            'anchor': anchors.clone(),
            'owner': owner.clone(),
            'owner_idx': owner_idx.clone(),
            'valid': valid.clone(),
            'pose': body.clone(),
            'eef': eef.clone(),
            'goal': goal_pose.clone(),
            'obj': obj_pose.clone(),
            'phys': phys.clone(),
            'feat': feats,
            'image': (color.clamp(0, 1) * 255).to(th.uint8).clone()
                     if STORE_IMAGE else None,
        })
        if len(hist) > HORIZON + 1:
            hist.popleft()

        # ---- emit the frame that now has a full future window ---------------
        if len(hist) == HORIZON + 1:
            t0 = step - HORIZON
            rec = hist[0]
            # future body poses, oldest->newest: hist[1..H]
            fut = th.stack([hist[i]['pose'] for i in range(1, HORIZON + 1)], 1)
            track, mask = compute_tracks(rec, fut, owner_clamp=n_body)
            nxt_eef = hist[1]['eef']

            for k in range(E):
                if t0 < ep_start[k]:          # window straddles a reset
                    n_dropped += 1
                    continue
                # Belt-and-braces against any residual boundary slip: a window
                # spanning a reset produces metre-scale displacements, far
                # beyond what 16 steps of pushing can do. Drop, don't ship.
                mx = float(track[k].norm(dim=-1).max())
                if mx > TELEPORT_M:
                    n_teleport += 1
                    continue
                fr = {
                    'image_patch_centre_3d_position':
                        dcn(rec['anchor'][k]).astype(np.float32),
                    'gt_3d_track': dcn(track[k]).astype(np.float32),
                    'track_valid_mask_gt': dcn(mask[k]).astype(np.bool_),
                    'patch_owner_body': dcn(rec['owner'][k]).astype(np.int16),
                    'robot_state': dcn(rec['eef'][k]).astype(np.float32),
                    'action': dcn(nxt_eef[k]).astype(np.float32),
                    'goal_pose': dcn(rec['goal'][k]).astype(np.float32),
                    'phys_params': dcn(rec['phys'][k]).astype(np.float32),
                    'object_pose': dcn(rec['obj'][k]).astype(np.float32),
                    'action_cmd': dcn(rec['cmd'][k]).astype(np.float32),
                }
                if STORE_IMAGE:
                    fr['image'] = dcn(rec['image'][k]).transpose(1, 2, 0)
                if USE_DINO:
                    fr['dino_features'] = \
                        dcn(rec['feat'][k]).astype(np.float16)[None]
                pending[k].append(fr)

        # ---- step -----------------------------------------------------------
        action = obs['teacher_action'][..., 0, :]
        # Attach to the frame just appended: this is the command applied FROM
        # that state, so frame t's action_cmd drives the t -> t+1 transition --
        # the same alignment `action` (realized EEF at t+1) already has.
        hist[-1]['cmd'] = action.clone()
        obs, rew, done, info = img_env.step(action)
        done = done.reshape(-1).bool()
        if 'success' in info:
            succ = dcn(info['success']).astype(bool).reshape(-1)
        else:
            succ = np.zeros(E, dtype=bool)

        # A done env is reset during the NEXT step, so the frame we read on the
        # next iteration is the new episode's first.
        if done.any():
            # The TERMINAL frame: a done env is not reset until the next
            # step's pre_physics, so right now obs/tensors still hold this
            # episode's final state -- the frame where the goal is reached.
            # (Same staleness that poisons cross-reset windows, used here on
            # purpose.) Captured once, before the per-env flush below.
            term_anchor = term_owner = term_eef = None
            if EMIT_TAIL and done.any():
                t_depth, t_label = obs['gap_depth'], obs['gap_label']
                t_pts = unproject(t_depth, cam_cfg)
                t_a = t_pts[:, :, rows][:, :, :, cols]
                term_anchor = t_a.reshape(E, 3, N_PATCH).permute(0, 2, 1)
                t_lab = t_label[:, 0][:, rows][:, :, cols].reshape(E, N_PATCH)
                t_dep = t_depth[:, 0][:, rows][:, :, cols].reshape(E, N_PATCH)
                t_r = th.round(t_lab)
                t_ok = ((t_lab - t_r).abs() < 1e-3) & (t_dep > 0) & (t_r > 0)
                term_owner = th.where(t_ok, t_r.long() - 1,
                                      th.full_like(t_r, -1, dtype=th.long))
                term_eef = env.tensors['body'][tip, :7].reshape(E, 7)
                term_obj = env.tensors['root'][
                    env.scene.cur_ids.long(), :7]
                term_img = ((obs['gap_color'].clamp(0, 1) * 255).to(th.uint8)
                            if STORE_IMAGE else None)
                term_feat = (dino_features(dino, obs['gap_color'], mean, std)
                             if USE_DINO else None)

            for k in th.where(done)[0].tolist():
                # Tail: frames still inside `hist` that never got a full
                # future window. hist[0] was already emitted above; hist[1:]
                # covers steps step-H+1 .. step, i.e. through the terminal
                # frame where the goal is actually reached.
                if EMIT_TAIL and pending[k]:
                    for j in range(1, len(hist)):
                        rec = hist[j]
                        nxt = hist[j + 1] if j + 1 < len(hist) else hist[j]
                        fr = {
                            'image_patch_centre_3d_position':
                                dcn(rec['anchor'][k]).astype(np.float32),
                            'gt_3d_track':
                                np.zeros((N_PATCH, HORIZON, 3), np.float32),
                            'track_valid_mask_gt':
                                np.zeros((N_PATCH, HORIZON), np.bool_),
                            'patch_owner_body':
                                dcn(rec['owner'][k]).astype(np.int16),
                            'robot_state': dcn(rec['eef'][k]).astype(np.float32),
                            'action': dcn(nxt['eef'][k]).astype(np.float32),
                            'goal_pose': dcn(rec['goal'][k]).astype(np.float32),
                            'object_pose': dcn(rec['obj'][k]).astype(np.float32),
                            'phys_params': dcn(rec['phys'][k]).astype(np.float32),
                            'action_cmd': dcn(rec['cmd'][k]).astype(np.float32),
                        }
                        if STORE_IMAGE:
                            fr['image'] = dcn(rec['image'][k]).transpose(1, 2, 0)
                        if USE_DINO:
                            fr['dino_features'] = \
                                dcn(rec['feat'][k]).astype(np.float16)[None]
                        pending[k].append(fr)
                        n_tail += 1

                    # ... then the terminal frame itself.
                    if term_anchor is not None:
                        fr = {
                            'image_patch_centre_3d_position':
                                dcn(term_anchor[k]).astype(np.float32),
                            'gt_3d_track':
                                np.zeros((N_PATCH, HORIZON, 3), np.float32),
                            'track_valid_mask_gt':
                                np.zeros((N_PATCH, HORIZON), np.bool_),
                            'patch_owner_body':
                                dcn(term_owner[k]).astype(np.int16),
                            'robot_state': dcn(term_eef[k]).astype(np.float32),
                            'action': dcn(term_eef[k]).astype(np.float32),
                            'goal_pose':
                                dcn(hist[-1]['goal'][k]).astype(np.float32),
                            'object_pose': dcn(term_obj[k]).astype(np.float32),
                            'phys_params':
                                dcn(hist[-1]['phys'][k]).astype(np.float32),
                            # The episode ended here, so no further command was
                            # issued; carry the last one. This frame has an
                            # all-False track mask and is never trained on.
                            'action_cmd':
                                dcn(hist[-1]['cmd'][k]).astype(np.float32),
                        }
                        if STORE_IMAGE:
                            fr['image'] = dcn(term_img[k]).transpose(1, 2, 0)
                        if USE_DINO:
                            fr['dino_features'] = \
                                dcn(term_feat[k]).astype(np.float16)[None]
                        pending[k].append(fr)
                        n_tail += 1

                n = len(pending[k])
                ok_len = (n >= MIN_EP_LEN) and (MAX_EP_LEN <= 0 or n <= MAX_EP_LEN)
                ok_succ = bool(succ[k]) or not SUCCESS_ONLY
                if ok_len and ok_succ:
                    n_frames += writer.write_episode(pending[k],
                                                     success=bool(succ[k]))
                    n_ep_done += 1
                    n_succ += int(bool(succ[k]))
                else:
                    n_rejected += 1
                    if not ok_succ:
                        n_rej_fail += 1
                    elif n < MIN_EP_LEN:
                        n_rej_short += 1
                    else:
                        n_rej_long += 1
                        print(f'[gap] env{k}: dropping {n}-frame episode '
                              f'(> GAP_MAX_EP_LEN={MAX_EP_LEN})')
                pending[k] = []
                # Two-step delay, not one. At iteration `i` we capture from the
                # obs produced at the END of iteration i-1. A done at iteration
                # `step` is reset during step()'s pre_physics at `step+1`, whose
                # obs is captured as frame `step+2`. Frame `step+1` still holds
                # the OLD episode's terminal state, so admitting it would replay
                # an old-episode anchor through new-episode poses.
                ep_start[k] = step + 2

        if step % 200 == 0:
            # Rate matters: this loop runs for hours and used to print no
            # timing at all, so a slow run was indistinguishable from a normal
            # one. frames/s is the number to watch -- it scales with num_env,
            # since almost everything per step is fixed overhead.
            el = time.time() - t_start
            print(f'[gap] step {step:6d}  episodes {n_ep_done:4d}/{NUM_EPISODES}'
                  f' (succ {n_succ}, rejected {n_rejected}:'
                  f' {n_rej_fail} failed/{n_rej_short} short/{n_rej_long} long)'
                  f'  frames {n_frames:7d}  dropped(reset) {n_dropped}'
                  f'  dropped(teleport) {n_teleport}'
                  f'  | {el:6.0f}s  {n_frames / max(el, 1e-6):6.1f} frame/s'
                  f'  {step / max(el, 1e-6):5.2f} step/s', flush=True)

        if n_ep_done >= NUM_EPISODES:
            break

    writer.close()
    kept_total = n_ep_done + n_rejected
    print(f'\n[gap] done: {n_ep_done} episodes kept ({n_succ} successful), '
          f'{n_frames} frames -> {OUT}')
    print(f'[gap] {n_tail} tail frames emitted with no valid track '
          f'(the final {HORIZON} of each episode, where the goal is reached)')
    print(f'[gap] rejected {n_rejected}/{kept_total} terminated episodes: '
          f'{n_rej_fail} failed, {n_rej_short} shorter than {MIN_EP_LEN}, '
          f'{n_rej_long} longer than {MAX_EP_LEN}')
    if kept_total:
        print(f'[gap] raw teacher success rate over all terminated episodes: '
              f'{100.0 * n_succ / kept_total:.1f}%')
    print(f'[gap] {n_dropped} frame-slots dropped for crossing an episode reset')
    if n_teleport:
        print(f'[gap] WARNING: {n_teleport} frame-slots exceeded '
              f'{TELEPORT_M} m and were dropped as teleports -- if this is '
              f'nonzero the episode-boundary bookkeeping has slipped')
    os._exit(0)


def compute_tracks(rec, fut_pose, owner_clamp: int):
    """Rigidly replay each anchor through its owning body's future poses.

    rec['anchor'] (E,N,3), rec['owner'] (E,N), rec['pose'] (E,nb,7)
    fut_pose      (E,H,nb,7)
    returns track (E,N,H,3) displacements, mask (E,N,H)
    """
    anchor = rec['anchor']
    owner = rec['owner_idx']        # clamped: -1 would break gather
    valid = rec['valid']
    E, N, _ = anchor.shape
    H = fut_pose.shape[1]

    idx = owner.unsqueeze(-1).expand(-1, -1, 7)             # (E,N,7)
    p0 = th.gather(rec['pose'], 1, idx)                     # (E,N,7) owner @ t
    t0, q0 = p0[..., :3], p0[..., 3:7]

    # anchor expressed in the owner's local frame
    local = quat_rotate(quat_conj(q0), anchor - t0)         # (E,N,3)

    idx_h = owner[:, None, :, None].expand(-1, H, -1, 7)    # (E,H,N,7)
    ph = th.gather(fut_pose, 2, idx_h)                      # (E,H,N,7)
    th_, qh = ph[..., :3], ph[..., 3:7]

    world = quat_rotate(qh, local[:, None].expand(-1, H, -1, -1)) + th_
    track = (world - anchor[:, None]).permute(0, 2, 1, 3)   # (E,N,H,3)

    mask = valid[..., None].expand(-1, -1, H).clone()
    track = th.where(mask.unsqueeze(-1), track, th.zeros_like(track))
    return track, mask


if __name__ == '__main__':
    main()
