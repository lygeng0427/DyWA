#!/usr/bin/env python3
"""Draw 3D tracks on the rendered frames -> mp4 / PNGs.

Same idea as third_party/GAP/scripts/visualize_tracks.py (project each patch's
anchor and its future trajectory onto the image, dot at the anchor, polyline
through the trajectory), with two differences:

  * by default these are GROUND-TRUTH tracks read from the file, so this
    visualises the DATA. Pass `--pred <ckpt>` to also draw a trained model's
    PREDICTED tracks (cyan) over the GT (green->red), which is the picture that
    shows what the model actually got right;
  * the projection is nvdr's, not a pinhole K/extrinsic pair. nvdr's
    unproject() maps a pixel to
        p_world = R^T ([X, Y, 1] * depth - t),   R = T_cam[:3,:3], t = T_cam[:3,3]
    with X, Y spanning [-tan(fov/2), +tan(fov/2)] across the image (the SAME
    tangent on both axes -- the config's aspect is 1.0 even though the frame is
    not square). Inverting:
        p_cam = R p_world + t;  X = p_cam.x/p_cam.z;  Y = p_cam.y/p_cam.z
        col = (X + x)/(2x) * (W-1);   row = (x - Y)/(2x) * (H-1)
    Camera parameters come from the file's attrs, written by the collector, so
    this cannot drift from whatever geometry actually produced the data.

Self-check: every anchor MUST re-project onto its own patch centre
(14*i+7, 14*j+7). The script asserts this before drawing anything -- if the
projection were wrong, the overlay would look plausible but be meaningless.

Usage (inside container):
    python3 visualize_gap_tracks.py output/gap/tracks.hdf5 --episode 0 \
        --out /tmp/docker/gap_tracks.mp4
    # only the object + arm, skip the static table:
    python3 visualize_gap_tracks.py ... --skip-bodies 0
    # overlay a model's predictions (cyan) on the GT:
    python3 visualize_gap_tracks.py ... --skip-bodies 0 \
        --pred output/gap/ckpt/transformer_d512/best.ckpt
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

PATCH = 14


class VideoOut:
    """H.264/yuv420p writer, falling back to OpenCV.

    cv2.VideoWriter's 'mp4v' fourcc produces MPEG-4 Part 2, which most players
    and browsers refuse ("unsupported codec"). The container has no system
    ffmpeg but does ship imageio_ffmpeg's bundled binary, so use that; only
    fall back to mp4v if it is genuinely unavailable.
    """

    def __init__(self, path, w, h, fps):
        self.path, self.w, self.h = path, w, h
        self.kind = None
        try:
            import imageio_ffmpeg
            # yuv420p needs even dimensions; macro_block_size=1 stops
            # imageio silently resizing to a multiple of 16.
            assert w % 2 == 0 and h % 2 == 0, (w, h)
            self.w_ = imageio_ffmpeg.write_frames(
                path, size=(w, h), fps=fps, codec='libx264',
                pix_fmt_in='bgr24', pix_fmt_out='yuv420p',
                macro_block_size=1, quality=8)
            self.w_.send(None)
            self.kind = 'h264'
        except Exception as ex:                       # noqa: BLE001
            print(f'  imageio_ffmpeg unavailable ({ex}); falling back to mp4v '
                  f'-- the file may not play in some viewers')
            self.w_ = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'),
                                      fps, (w, h))
            self.kind = 'mp4v'

    def write(self, img_bgr):
        if self.kind == 'h264':
            self.w_.send(np.ascontiguousarray(img_bgr).tobytes())
        else:
            self.w_.write(img_bgr)

    def close(self):
        if self.kind == 'h264':
            self.w_.close()
        else:
            self.w_.release()


class Predictor:
    """Runs a trained TrackModel over one episode's frames.

    The model has no latent, so there is nothing to adapt per episode and every
    frame is scored the same way -- no support/query split is needed here.
    """

    def __init__(self, ckpt_path, h5_path, episode, device='cuda:0',
                 infer_steps=None):
        import torch as th
        sys.path.insert(0, '/home/user/DyWA/dywa/src')
        from models.gap.common import FieldNorm
        from models.gap.track_model import TrackModel
        self.th = th
        self.dev = th.device(device)

        ck = th.load(ckpt_path, map_location='cpu')
        m = TrackModel(trunk=ck['trunk'], d_model=ck['cfg']['d_model'],
                       action_mode=ck['cfg'].get('action_mode', 'realized'))
        m.normalizer.add('gt_3d_track',
                         FieldNorm(3).fit_isotropic_rms(ck['track_rms']))
        m.load_state_dict(ck['model'])
        # The DiT's DDIM chain degrades its own first estimate (measured
        # monotone: 1 step 54 mm -> 50 steps 125 mm), so visualise it at the
        # setting that actually represents the trunk.
        if ck['trunk'] == 'dit':
            m.num_inference_steps = infer_steps or 1
        self.model = m.to(self.dev).eval()
        self.trunk = ck['trunk']
        self.action_mode = ck['cfg'].get('action_mode', 'realized')

        # global episode id: the dataset concatenates files in ckpt['paths']
        # order, so a local index must be offset by the preceding files'
        # episode counts or the split membership reported below is wrong.
        import h5py as _h5
        rp = os.path.realpath(h5_path)
        off, gid = 0, None
        for p in ck['paths']:
            with _h5.File(p, 'r', locking=False) as fp:
                n = len(fp['episode_ends'])
            if os.path.realpath(p) == rp:
                gid = off + episode
                break
            off += n
        assert gid is not None, (
            '%s is not one of the files this checkpoint was trained on: %s'
            % (h5_path, ck['paths']))
        split = 'val' if gid in set(int(x) for x in ck['val_ids']) else 'train'
        print('  [pred] %s trunk, action=%s, episode global id %d (%s split)'
              % (self.trunk, self.action_mode, gid, split))
        if split == 'train':
            print('  [pred] WARNING: this episode was TRAINED on -- the overlay '
                  'flatters the model')
        self.cut = None                      # no support/query split any more

    def _batch(self, f, t):
        th = self.th
        d = f['dino_features'][t].astype(np.float32)
        H = self.model.H
        # The model is action-conditioned, so the H-step future EEF path must be
        # supplied exactly as the dataset builds it -- action[t..t+H-1]. Tail
        # frames have no such window; they also have no GT track, so they are
        # never drawn with a prediction.
        act = f['action'][t:t + H].astype(np.float32)
        if len(act) < H:
            return None
        # Every action_mode except 'realized' also needs the 20-D command, and
        # a file collected before action_cmd existed simply cannot feed them.
        cmd = None
        if self.action_mode != 'realized':
            if 'action_cmd' not in f:
                raise KeyError(
                    "%s has no 'action_cmd' -- this checkpoint uses "
                    "action_mode=%s, which needs it. Re-collect with the "
                    "current collect_gap_tracks.py."
                    % (f.filename, self.action_mode))
            cmd = f['action_cmd'][t:t + H].astype(np.float32)
        out = {
            'dino_features': th.from_numpy(d.reshape(1, -1, d.shape[-1])).to(self.dev),
            'patch_centers': th.from_numpy(
                f['image_patch_centre_3d_position'][t].astype(np.float32)
            ).unsqueeze(0).to(self.dev),
            'agent_pos': th.from_numpy(
                f['robot_state'][t].astype(np.float32)).unsqueeze(0).to(self.dev),
            'action_seq': th.from_numpy(act).unsqueeze(0).to(self.dev),
        }
        if cmd is not None:
            out['action_cmd_seq'] = th.from_numpy(cmd).unsqueeze(0).to(self.dev)
        return out

    def predict(self, f, t):
        th = self.th
        b = self._batch(f, t)
        if b is None:
            return None
        with th.no_grad():
            out = self.model.predict(b, noise_seed=2000)
        return out[0].float().cpu().numpy()          # (N, H, 3) metres


class PredLatent(Predictor):
    """`Predictor` for a LATENT-conditioned model (`models.fric.LatentTrackModel`).

    Two differences from the base, and both matter for what the overlay MEANS:

    * the model needs `batch['q']`, so a latent must be supplied per frame;
    * that latent is fitted per EPISODE by test-time training, which is the
      thing being visualised. `mode` selects which latent is drawn:

        blind   q = q0, the learned prior. No adaptation at all.
        ttt     CAUSAL: at frame t the latent has descended the loss only on
                windows with `s + H <= t` -- what a robot could actually have
                known at that moment. Before the first window elapses this is
                bit-identical to `blind`, which is the built-in causality check.
        ceil    fitted on the WHOLE episode. Uses the future, so it is a
                reference for what the latent is worth, never a result.

    Measured on 419 val episodes: ttt is +0.07 mm over blind (a coin flip per
    episode) while ceil is +1.32. The overlays are the picture of that gap.
    """

    def __init__(self, ckpt_path, h5_path, episode, device='cuda:0',
                 mode='ttt', k=2, z_lr=0.003, prior_w=0.05, m_max=8,
                 ceil_steps=60, ceil_lr=0.02, infer_steps=None):
        import torch as th
        sys.path.insert(0, '/home/user/DyWA/dywa/src')
        from models.gap.common import FieldNorm
        from models.fric.latent_model import LatentTrackModel
        self.th = th
        self.dev = th.device(device)
        ck = th.load(ckpt_path, map_location='cpu')
        c = ck['cfg']
        m = LatentTrackModel(
            trunk=ck['trunk'], d_model=c['d_model'],
            action_mode=c['action_mode'], obj_weight=c['obj_weight'],
            dropout=0.0, dino_dim=max(int(c['dino_dim']), 1),
            phys_lo=c['phys_lo'], phys_hi=c['phys_hi'],
            phys_log=c['phys_log'], z_dim=c['z_dim'],
            q0_init_std=c['q0_init_std'],
            alpha_init=c.get('alpha_init', 0.1))
        m.normalizer.add('gt_3d_track',
                         FieldNorm(3).fit_isotropic_rms(float(ck['track_rms'])))
        m.load_state_dict(ck['model'], strict=True)
        self.model = m.to(self.dev).eval()
        self.trunk = ck['trunk']
        self.action_mode = c['action_mode']
        self.mode = mode
        self.cut = None
        self.dino_dim = int(c['dino_dim'])
        self.k_inner = int(os.environ.get('META_K_EVAL', c['k_inner']))
        # The support cap must come from the CHECKPOINT, not from this
        # signature's default: r2b trained at m_max=8 (which the default
        # happened to match) but r3 trained at 4, and a mismatch here renders a
        # latent the model was never meta-trained to receive -- silently, since
        # the overlay still draws.
        self.m_max = int(os.environ.get('META_M_MAX_EVAL', c.get('m_max', m_max)))
        self.every = int(c['every'])
        self.split_seed = int(c.get('split_seed', 0))

        # Global episode id, for the train/val warning. The file list in the
        # ckpt is the concatenation order, so a local index needs the preceding
        # files' episode counts added.
        import h5py as _h5
        rp = os.path.realpath(h5_path)
        off, gid = 0, None
        for q in ck['paths']:
            with _h5.File(q, 'r', locking=False) as fp:
                n = len(fp['episode_ends'])
            if os.path.realpath(q) == rp:
                gid = off + episode
                break
            off += n
        assert gid is not None, '%s not among %s' % (h5_path, ck['paths'])
        self.gid = gid
        split = 'val' if gid in set(int(x) for x in ck['val_ids']) else 'train'
        print('  [pred] LATENT z_dim=%d mode=%s K=%d m_max=%d every=%d '
              '| episode global id %d (%s)'
              % (c['z_dim'], mode, self.k_inner, self.m_max, self.every,
                 gid, split))
        if split == 'train':
            print('  [pred] WARNING: TRAINED on -- the overlay flatters it')

        self.q0 = self.model.q0.detach()
        self.qmap = None
        # `_batch` is called with an ABSOLUTE row index into the file, while the
        # latents are keyed by the frame's index WITHIN its episode (which is
        # what the causality rule `s + H <= t` is defined on). Without this
        # offset every lookup misses and silently falls back to the prior --
        # making every mode render identically, which is exactly what happened.
        import h5py as _h5
        with _h5.File(h5_path, 'r', locking=False) as _f:
            _ends = _f['episode_ends'][:]
        self.ep_start = int(_ends[episode - 1]) if episode > 0 else 0
        if mode != 'blind':
            self._fit(h5_path, k, z_lr, prior_w, self.m_max,
                      ceil_steps, ceil_lr)

    def _fit(self, h5_path, k, z_lr, prior_w, m_max, ceil_steps, ceil_lr):
        """Fit the per-frame latents once, up front, with the SAME code the
        evaluation uses -- importing rather than reimplementing, so the picture
        cannot drift from the numbers it illustrates."""
        import numpy as _np
        import meta_ttt as M
        from models.fric.dataset import FricTrackDataset
        from train_fric_track import ProjectedDino
        th = self.th
        d0 = FricTrackDataset([h5_path], phys_col=0, episodes=[self.gid])
        ds = ProjectedDino(d0, self.dino_dim, self.split_seed) \
            if self.dino_dim != 1024 else d0
        data, t_rel, sc = M.cache_episode(ds, self.dev)
        H = self.model.H
        if self.mode == 'ttt':
            # EXACTLY `meta_ttt.walk_episode`: one fit per anchor, restarted
            # from q0, K steps of the LEARNED alpha. Not the shelved
            # block-coordinate loop (z_lr / prior_w), which is a different
            # algorithm and would illustrate numbers we never measured.
            M.assert_causal(t_rel, H, m_max, self.every)
            rows_all = _np.nonzero(sc.cpu().numpy())[0]
            groups = {}
            for i in rows_all:
                groups.setdefault(M.anchor_row(t_rel, int(i), self.every),
                                  []).append(int(i))
            self.qmap = {}
            for a, idx in groups.items():
                rws = M.support(t_rel, a, H, m_max)
                if len(rws) == 0:
                    q = self.q0.reshape(-1)             # causality: prior
                else:
                    sup = M.pack_support([M.take(data, rws)], len(rws))
                    with th.enable_grad():
                        qk, _ = M.adapt(self.model, sup,
                                        self.q0.reshape(1, -1).clone(),
                                        self.k_inner, create_graph=False)
                    q = qk.reshape(-1).detach()
                for i in idx:
                    self.qmap[int(t_rel[i])] = q
        else:                                       # 'ceil' -- uses the future
            rows = _np.nonzero(sc.cpu().numpy())[0]
            sel = rows[_np.unique(_np.linspace(0, len(rows) - 1,
                                               min(16, len(rows))).astype(int))]
            sup = M.pack_support([M.take(data, sel)], len(sel))
            with th.enable_grad():
                qf, _ = M.adapt(self.model, sup, self.q0.reshape(1, -1).clone(),
                                ceil_steps, create_graph=False)
            qf = qf.reshape(-1).detach()
            self.qmap = {int(t): qf for t in t_rel}
        print('  [pred] latents fitted for %d frames (mode=%s)'
              % (len(self.qmap), self.mode))
        del data

    def _batch(self, f, t):
        b = super()._batch(f, t)
        if b is None:
            return None
        # Frames outside the fitted set (no valid future window) fall back to
        # the prior rather than silently reusing a neighbour's latent.
        rel = int(t) - self.ep_start
        q = self.q0 if self.qmap is None else self.qmap.get(rel, self.q0)
        self._n_hit = getattr(self, '_n_hit', 0) + (self.qmap is not None
                                                    and rel in self.qmap)
        self._n_ask = getattr(self, '_n_ask', 0) + 1
        b['q'] = q.reshape(1, -1)
        return b

    def report(self):
        """How many drawn frames actually used a fitted latent. A run where
        this is 0 is rendering the prior under a different name."""
        h, n = getattr(self, '_n_hit', 0), getattr(self, '_n_ask', 0)
        print('  [pred] latent applied on %d/%d drawn frames (mode=%s)'
              % (h, n, self.mode))
        assert self.mode == 'blind' or h > 0, (
            'no drawn frame received a fitted latent -- the qmap keys do not '
            'match the frame indices, so this render is just the prior')


def quat_to_R(q):
    """(x, y, z, w) unit quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(p_world, T_cam, fov, H, W):
    """world (...,3) -> (col, row) pixels (...,2) and camera-z (...,)."""
    R, t = T_cam[:3, :3], T_cam[:3, 3]
    p_cam = p_world @ R.T + t
    z = np.where(np.abs(p_cam[..., 2]) < 1e-9, 1e-9, p_cam[..., 2])
    X = p_cam[..., 0] / z
    Y = p_cam[..., 1] / z
    x = np.tan(0.5 * fov)
    col = (X + x) / (2 * x) * (W - 1)
    row = (x - Y) / (2 * x) * (H - 1)
    return np.stack([col, row], -1), z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--out', default='/tmp/docker/gap_tracks.mp4')
    ap.add_argument('--upscale', type=int, default=2)
    ap.add_argument('--skip-bodies', type=int, nargs='*', default=[],
                    help='body ids to omit (e.g. 0 = the static table)')
    ap.add_argument('--only-bodies', type=int, nargs='*', default=None,
                    help='draw ONLY these body ids (1 = the pushed object; '
                         '7-12 are franka links, which dominate the frame and '
                         'bury the object under ~40x more patches)')
    ap.add_argument('--png-every', type=int, default=0,
                    help='also dump every Nth frame as a PNG next to --out')
    ap.add_argument('--no-goal', action='store_true',
                    help='do not draw the goal pose marker')
    ap.add_argument('--ghost', default=None,
                    help='.npz from _render_gap_ghosts.py: blends a translucent '
                         'render of the object at its goal pose (as the TTT '
                         'videos do) instead of an axis marker')
    ap.add_argument('--ghost-key', default=None,
                    help='key within the ghost npz (default "<stem>:<episode>")')
    ap.add_argument('--ghost-alpha', type=float, default=0.45)
    ap.add_argument('--pred', default=None,
                    help='checkpoint whose PREDICTED tracks to draw (cyan) over '
                         'the GT')
    ap.add_argument('--pred-only', action='store_true',
                    help='draw only the prediction, not the GT')
    ap.add_argument('--pred-device', default='cuda:0')
    ap.add_argument('--latent', default=None,
                    choices=['blind', 'ttt', 'ceil'],
                    help='treat --pred as a LATENT-conditioned checkpoint and '
                         'draw it with this latent: blind (prior, no '
                         'adaptation), ttt (causal test-time training), or '
                         'ceil (fitted on the whole episode -- uses the '
                         'future, a reference not a result)')
    ap.add_argument('--latent-k', type=int, default=2)
    ap.add_argument('--latent-lr', type=float, default=0.003)
    ap.add_argument('--latent-anchor', type=float, default=0.05)
    ap.add_argument('--gt-white', action='store_true',
                    help='draw ground truth in flat WHITE (implies --flat-gt)')
    ap.add_argument('--flat-gt', action='store_true',
                    help='draw ground truth in flat RED instead of the '
                         'green->red displacement ramp (and drop the green '
                         'anchor dots), so the only colours are red = GT and '
                         'cyan = prediction')
    ap.add_argument('--infer-steps', type=int, default=None,
                    help='DDIM steps for a dit checkpoint (default 1 -- more '
                         'steps measurably degrade it)')
    args = ap.parse_args()

    f = h5py.File(args.path, 'r', locking=False)
    if 'T_cam' not in f.attrs:
        sys.exit('file has no T_cam attr -- regenerate with the current '
                 'collect_gap_tracks.py, which records the camera')
    T_cam = np.asarray(f.attrs['T_cam'], np.float64).reshape(4, 4)
    fov = float(f.attrs['cam_fov'])
    GH, GW = [int(v) for v in f.attrs['patch_grid']]
    IH, IW = [int(v) for v in f.attrs['img_size']]

    ghost_rgb = ghost_mask = None
    if args.ghost:
        gz = np.load(args.ghost)
        key = args.ghost_key
        if key is None:
            cands = [k for k in gz.files if not k.endswith('_mask')
                     and k.endswith(f':{args.episode}')]
            key = cands[0] if cands else None
        if key is None or key not in gz.files:
            print(f'  ghost key not found (have: {sorted(gz.files)[:6]})')
        else:
            ghost_rgb = gz[key].astype(np.float32)
            ghost_mask = gz[key + '_mask'].astype(bool)
            print(f'  ghost "{key}": {int(ghost_mask.sum())} px, '
                  f'alpha {args.ghost_alpha}')

    has_goal = ('goal_pose' in f) and (not args.no_goal)
    if 'goal_pose' not in f:
        print('  (no goal_pose in file -- regenerate to draw the goal)')

    ends = f['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    s, e = int(starts[args.episode]), int(ends[args.episode])
    print(f'episode {args.episode}: frames {s}..{e}  grid {GH}x{GW}  '
          f'img {IH}x{IW}  fov {np.rad2deg(fov):.1f} deg')

    # ── self-check: anchors must re-project onto their own patch centres ────
    exp_col = np.tile((np.arange(GW) * PATCH + PATCH // 2)[None, :], (GH, 1)).ravel()
    exp_row = np.tile((np.arange(GH) * PATCH + PATCH // 2)[:, None], (1, GW)).ravel()
    anch0 = f['image_patch_centre_3d_position'][s].astype(np.float64)
    m0 = f['track_valid_mask_gt'][s][:, 0]
    uv0, _ = project(anch0, T_cam, fov, IH, IW)
    err = np.hypot(uv0[m0, 0] - exp_col[m0], uv0[m0, 1] - exp_row[m0])
    print(f'reprojection self-check on {int(m0.sum())} valid anchors: '
          f'max {err.max():.3f} px, median {np.median(err):.3f} px')
    assert err.max() < 1.0, (
        f'anchors do not re-project onto the patch grid (max {err.max():.2f} px)'
        ' -- the projection does not match the collector')
    print('  [ok] projection matches the collector')

    # ── optional model predictions ──────────────────────────────────────────
    pred = None
    if args.pred:
        if args.latent:
            pred = PredLatent(args.pred, args.path, args.episode,
                              device=args.pred_device, mode=args.latent,
                              k=args.latent_k, z_lr=args.latent_lr,
                              prior_w=args.latent_anchor)
        else:
            pred = Predictor(args.pred, args.path, args.episode,
                             device=args.pred_device,
                             infer_steps=args.infer_steps)

    U = max(1, args.upscale)
    # H.264/yuv420p wants even dimensions
    OW, OH = (IW * U) // 2 * 2, (IH * U) // 2 * 2
    writer = VideoOut(args.out, OW, OH, args.fps)
    skip = set(args.skip_bodies)
    only = set(args.only_bodies) if args.only_bodies is not None else None
    n_out = 0
    err_sum = err_n = 0.0
    for t in range(s, e, args.stride):
        img = f['image'][t]                                  # (H,W,3) uint8 RGB
        if ghost_rgb is not None:
            # alpha-blend at NATIVE resolution, before upscaling, so the ghost
            # stays pixel-aligned with the frame it was rendered against.
            a = (args.ghost_alpha * ghost_mask)[..., None]
            img = (img.astype(np.float32) * (1 - a) + ghost_rgb * a)
            img = img.clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (OW, OH), interpolation=cv2.INTER_NEAREST)
        sx, sy = OW / IW, OH / IH

        anch = f['image_patch_centre_3d_position'][t].astype(np.float64)
        trk = f['gt_3d_track'][t].astype(np.float64)          # (N,H,3)
        msk = f['track_valid_mask_gt'][t]
        own = f['patch_owner_body'][t]

        a_uv, _ = project(anch, T_cam, fov, IH, IW)
        fut_uv, fut_z = project(anch[:, None, :] + trk, T_cam, fov, IH, IW)

        ptrk = puv = pz = None
        if pred is not None and msk[:, 0].any():
            ptrk = pred.predict(f, t)
            if ptrk is not None:
                ptrk = ptrk.astype(np.float64)
                puv, pz = project(anch[:, None, :] + ptrk, T_cam, fov, IH, IW)

        def polyline(uv, z, i, col, a_px, halo=False, w=1):
            # `halo` strokes a dark underlay first. The renders are GREYSCALE,
            # so a white track is invisible against the arm and the lit half of
            # the table; the outline is what makes white usable at all.
            p = a_px
            for h in range(uv.shape[1]):
                if z[i, h] <= 1e-6:
                    break
                q = (int(round(uv[i, h, 0] * sx)), int(round(uv[i, h, 1] * sy)))
                if not (0 <= q[0] < OW and 0 <= q[1] < OH):
                    break
                if halo:
                    # Must stay THINNER-looking than the core, or the outline
                    # reads as the track: a 3px halo under a 1px line leaves
                    # black covering two thirds of the stroke.
                    cv2.line(img, p, q, (0, 0, 0), w + 2, cv2.LINE_AA)
                cv2.line(img, p, q, col, w, cv2.LINE_AA)
                p = q

        for i in range(anch.shape[0]):
            if not msk[i, 0] or int(own[i]) in skip:
                continue
            if only is not None and int(own[i]) not in only:
                continue
            a_px = (int(round(a_uv[i, 0] * sx)), int(round(a_uv[i, 1] * sy)))
            if not args.pred_only:
                if args.gt_white:
                    col = (255, 255, 255)              # BGR flat white
                elif args.flat_gt:
                    col = (0, 0, 255)                  # BGR flat red
                else:
                    moved = np.linalg.norm(trk[i, -1])
                    # static stays green; movers ramp to red with displacement
                    w = float(np.clip(moved / 0.15, 0, 1))
                    col = (int(60 * (1 - w)), int(220 * (1 - w) + 40 * w),
                           int(255 * w))
                polyline(fut_uv, fut_z, i, col, a_px,
                         halo=args.gt_white, w=2 if args.gt_white else 1)
            if ptrk is not None:
                polyline(puv, pz, i, (255, 255, 0), a_px)   # BGR cyan = predicted
                # accumulate the same metric eval reports, on MOVING patches
                if np.linalg.norm(trk[i, -1]) > 0.01:
                    err_sum += float(np.linalg.norm(ptrk[i] - trk[i], axis=-1).sum())
                    err_n += trk.shape[1]
            # anchor marker; white under --flat-gt so no green appears at all
            cv2.circle(img, a_px, 1,
                       (255, 255, 255) if (args.flat_gt or args.gt_white)
                       else (0, 255, 0), -1)

        if has_goal:
            g = f['goal_pose'][t].astype(np.float64)
            op = f['object_pose'][t].astype(np.float64)
            # goal frame axes, 4 cm, so orientation is visible (not just position)
            ax = np.eye(3) * 0.04
            R = quat_to_R(g[3:7])
            tips = g[:3] + (R @ ax).T                       # (3,3)
            pts, zs = project(np.vstack([g[:3], tips, op[:3]]), T_cam, fov, IH, IW)
            if np.all(zs > 1e-6):
                o = (int(round(pts[0, 0] * sx)), int(round(pts[0, 1] * sy)))
                # BGR: x=blue, y=green, z=red -- but under a flat-GT render
                # the whole point is that exactly two colours appear, and the
                # goal triad's green/red would read as track colours. Draw it
                # neutral there; it is labelled 'goal' in text anyway.
                _triad = ([(190, 190, 190)] * 3
                          if (args.flat_gt or args.gt_white)
                          else [(255, 120, 0), (0, 255, 0), (0, 0, 255)])
                for j, c in enumerate([] if ghost_rgb is not None else _triad):
                    q = (int(round(pts[1 + j, 0] * sx)),
                         int(round(pts[1 + j, 1] * sy)))
                    cv2.line(img, o, q, c, 2, cv2.LINE_AA)
                objp = (int(round(pts[4, 0] * sx)), int(round(pts[4, 1] * sy)))
                cv2.line(img, objp, o, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.circle(img, o, 5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(img, 'goal', (o[0] + 8, o[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
                            cv2.LINE_AA)
                d = float(np.linalg.norm(g[:3] - op[:3]))
                cv2.putText(img, f'obj->goal {100*d:.1f} cm', (6, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                            cv2.LINE_AA)

        cv2.putText(img, f'frame {t - s}', (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if pred is not None:
            _gtc = (255, 255, 255) if args.gt_white else (
                (0, 0, 255) if args.flat_gt else (0, 90, 255))
            _gtn = 'white' if args.gt_white else (
                'red' if args.flat_gt else 'green->red')
            # two runs of text so each colour name is PRINTED in its own colour
            cv2.putText(img, 'CYAN = prediction', (6, OH - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
                        cv2.LINE_AA)
            (tw, _), _b = cv2.getTextSize('CYAN = prediction',
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.putText(img, '%s = ground truth' % _gtn.upper(),
                        (6 + tw + 18, OH - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        _gtc, 1, cv2.LINE_AA)
        writer.write(img)
        if args.png_every and (n_out % args.png_every == 0):
            cv2.imwrite(os.path.splitext(args.out)[0] + f'_{t - s:04d}.png', img)
        n_out += 1

    writer.close()
    if pred is not None and hasattr(pred, 'report'):
        pred.report()
    if err_n:
        # NOT comparable to eval_gap_track.py: restricted to the bodies
        # actually drawn (--only-bodies / --skip-bodies) and to one episode.
        # Both shift it by tens of mm. eval_gap_track.py is the number to quote.
        print(f'  [pred] mean per-step L2 on MOVING patches, AMONG DRAWN BODIES,'
              f' over the drawn frames: {1000.0 * err_sum / err_n:.2f} mm '
              f'(one episode, drawn bodies only -- not comparable to '
              f'eval_gap_track.py)')
    print(f'wrote {args.out}  ({n_out} frames, {OW}x{OH}, codec={writer.kind})')


if __name__ == '__main__':
    main()
