#!/usr/bin/env python3
"""Make the effect of friction conditioning VISIBLE.

The obvious picture -- `phys_cond=none` next to `phys_cond=both` -- shows
nothing, and would be misleading if it did: that difference is 1.0% (0.15 mm)
against a 0.13 mm between-seed spread. The effect is only crisp as a
COUNTERFACTUAL: hold the frame, the image, the patches and the arm's future path
fixed, and change only the friction number handed to the model. Same weights,
same input, one scalar different -- so anything that moves in the picture is
friction and cannot be init noise.

Three modes, each answering something the others cannot:

  sweep    One frame, the object's predicted tracks drawn at several mu values
           at once, blue (slippery) -> red (grippy). Ground truth in white, the
           episode's TRUE mu boxed. This is the qualitative money shot: does
           telling the model "the table is slippery" make it predict the bottle
           slides further?

  curve    The quantitative version, over many val frames: mean predicted object
           displacement as a function of the mu fed in. Physics says this must
           SLOPE DOWN -- more friction, less motion. Nothing measured so far has
           checked the model got the DIRECTION right, only that it reacts.
           Plotted against the ground-truth dependence for scale.

  stills   Is the mu fan actually VISIBLE? Saves zoomed crops around the object
           and reports, per frame, how far apart the predicted endpoints are in
           MILLIMETRES and in PIXELS. Pixels is the honest unit for "can you see
           it": the camera spans ~0.8 m over 322 px, so ~2.5 mm/px -- a 10 mm
           effect is 4 px at native resolution and invisible without a zoom.

  heatmap  Per-patch |pred(mu_lo) - pred(mu_hi)|, drawn on the frame. Answers
           WHERE the conditioning acts. It should light up on the bottle and not
           on the arm -- the arm's future path is given to the model as input,
           so friction has nothing to add there. If the arm lights up, the model
           is using mu as a generic nuisance code.

Usage (inside the container):
    python3 visualize_fric_conditioning.py <tracks.hdf5> --ckpt <both/best.ckpt> \
        --mode sweep --episode 7 --out /tmp/docker/sweep.mp4
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_gap_tracks import VideoOut, project           # noqa: E402

import h5py                                                   # noqa: E402

OBJECT_BODY = 1


class FricPredictor:
    """One checkpoint, queried at an arbitrary friction value.

    `mu` is an argument rather than read from the file on purpose: the whole
    point is to feed values the episode did NOT have.
    """

    def __init__(self, ckpt_path, device='cuda:0', infer_steps=None):
        import torch as th
        sys.path.insert(0, '/home/user/DyWA/dywa/src')
        from models.gap.common import FieldNorm
        from models.fric.track_model import FricTrackModel
        self.th = th
        self.dev = th.device(device)
        ck = th.load(ckpt_path, map_location='cpu')
        c = ck['cfg']
        self.cond = c.get('phys_cond', 'none')
        assert self.cond != 'none', (
            '%s is an UNCONDITIONED checkpoint -- there is no physics input to '
            'sweep. Point --ckpt at a phys_cond=both run.' % ckpt_path)
        m = FricTrackModel(trunk=ck['trunk'], d_model=c['d_model'],
                           action_mode=c.get('action_mode', 'realized_gain'),
                           obj_weight=c.get('obj_weight', 1.0),
                           phys_cond=self.cond,
                           phys_lo=c['phys_lo'], phys_hi=c['phys_hi'],
                           phys_log=c.get('phys_log', False),
                           dino_dim=max(int(c.get('dino_dim', 1024)), 1),
                           dropout=c.get('dropout', 0.1))
        m.normalizer.add('gt_3d_track',
                         FieldNorm(3).fit_isotropic_rms(ck['track_rms']))
        m.load_state_dict(ck['model'], strict=True)
        if ck['trunk'] == 'dit':
            m.num_inference_steps = infer_steps or 1
        self.model = m.to(self.dev).eval()
        self.H = m.H
        self.lo, self.hi = c['phys_lo'], c['phys_hi']
        self.log = bool(c.get('phys_log', False))
        # A ckpt trained on narrowed DINO must be fed narrowed DINO. If the file
        # still stores raw 1024-d features, project them here with the SAME
        # matrix training used -- same seed, same construction. A different
        # matrix would render a plausible-looking overlay from a model fed
        # garbage. Since 2026-09-03 the mass files that carry `image`
        # (tracks_m_a/b) are stored ALREADY narrowed to 64, so the width guard
        # below skips this path; projecting twice would destroy the features.
        self.dino_dim = int(c.get('dino_dim', 1024))
        self.dino_P = None
        if 0 < self.dino_dim < 1024:
            gen = th.Generator().manual_seed(int(c.get('split_seed', 0)))
            self.dino_P = th.randn(1024, self.dino_dim, generator=gen) \
                / float(np.sqrt(1024.0))
        self.channel = c.get('phys_channel', 'table')
        self.action_mode = c.get('action_mode', 'realized_gain')
        self.val_ids = set(int(x) for x in ck['val_ids'])
        self.paths = ck['paths']
        self.shuffled = bool(c.get('shuffle'))
        print('[pred] %s trunk, phys_cond=%s, action=%s, %s in [%.3f, %.3f] '
              '(%s)%s'
              % (ck['trunk'], self.cond, self.action_mode, self.channel,
                 self.lo, self.hi, 'log' if self.log else 'linear',
                 '  (SHUFFLED-LABEL CONTROL)' if self.shuffled else ''))

    def global_id(self, h5_path, episode):
        """Local episode index -> the id used by the checkpoint's split."""
        rp, off = os.path.realpath(h5_path), 0
        for p in self.paths:
            with h5py.File(p, 'r', locking=False) as fp:
                n = len(fp['episode_ends'])
            if os.path.realpath(p) == rp:
                return off + episode
            off += n
        # Not one of the checkpoint's files. That is legitimate for a STREAM
        # shard, whose episodes belong to neither split by construction -- the
        # streaming run fits each shard once and deletes it, so there is no
        # global id to map to. Return None and let the caller label it
        # explicitly rather than mislabel it 'val' (a silent lie about whether
        # the overlay flatters the model).
        return None

    def batch(self, f, t):
        th = self.th
        act = f['action'][t:t + self.H].astype(np.float32)
        if len(act) < self.H:
            return None
        d = f['dino_features'][t].astype(np.float32)
        dv = th.from_numpy(d.reshape(1, -1, d.shape[-1]))
        if self.dino_dim == 0:
            dv = th.zeros_like(dv[..., :1])
        elif self.dino_P is not None and dv.shape[-1] != self.dino_dim:
            dv = dv @ self.dino_P
        out = {
            'dino_features': dv.to(self.dev),
            'patch_centers': th.from_numpy(
                f['image_patch_centre_3d_position'][t].astype(np.float32)
            ).unsqueeze(0).to(self.dev),
            'agent_pos': th.from_numpy(
                f['robot_state'][t].astype(np.float32)).unsqueeze(0).to(self.dev),
            'action_seq': th.from_numpy(act).unsqueeze(0).to(self.dev),
        }
        if self.action_mode != 'realized':
            out['action_cmd_seq'] = th.from_numpy(
                f['action_cmd'][t:t + self.H].astype(np.float32)
            ).unsqueeze(0).to(self.dev)
        return out

    def predict_at(self, b, mus):
        """[K] friction values -> [K, N, H, 3] predictions.

        Batched as K copies of ONE frame so every mu sees bit-identical inputs;
        looping would also work but this makes the "only mu changed" claim
        structural rather than a promise.
        """
        th = self.th
        K = len(mus)
        rep = {k: v.expand(K, *v.shape[1:]).contiguous() for k, v in b.items()}
        rep['phys'] = th.tensor(mus, dtype=th.float32,
                                device=self.dev).reshape(K, 1)
        with th.no_grad():
            out = self.model.predict(rep, noise_seed=2000)
        return out.float().cpu().numpy()


def cam_of(f):
    return (np.asarray(f.attrs['T_cam'], np.float64), float(f.attrs['cam_fov']),
            int(f.attrs['img_size'][0]), int(f.attrs['img_size'][1]))


# What is actually being swept. Read from the ckpt/dataset, never assumed: this
# file used to hardcode "table friction" into every axis label and a
# blue=slippery/red=grippy legend, which on a mass sweep would have shipped a
# figure that was wrong in its words while right in its numbers.
CHANNEL_AXIS = {'table': 'table friction $\\mu$',
                'object': 'object friction $\\mu$',
                'mass': 'object mass (kg)'}
CHANNEL_SHORT = {'table': 'friction', 'object': 'friction', 'mass': 'mass'}


def grid_of(P, k):
    """K counterfactual values laid out the way the real grid was.

    A geometric grid queried on a linear sweep would put most of the K probes in
    the heavy half, and the "predicted displacement vs value" curve would look
    flat at the light end purely because it was barely sampled there.
    """
    return (np.geomspace(P.lo, P.hi, k) if getattr(P, 'log', False)
            else np.linspace(P.lo, P.hi, k))


def val_label(P, v):
    """Format one value of the swept variable with its own name and unit.

    The legend used to read `mu=0.14` regardless of channel, which on a mass
    sweep names the wrong quantity and omits the unit entirely.
    """
    ch = getattr(P, 'channel', '?')
    return ('mass=%.2f kg' % v) if ch == 'mass' else ('mu=%.2f' % v)


def axis_label(P):
    return CHANNEL_AXIS.get(getattr(P, 'channel', '?'),
                            getattr(P, 'channel', 'swept value'))


def short_name(P):
    return CHANNEL_SHORT.get(getattr(P, 'channel', '?'), 'the swept value')


def mu_color(k, K):
    """Low (cool) -> high (warm), as BGR for cv2. Ordered by the swept value."""
    a = k / max(K - 1, 1)
    return (int(255 * (1 - a)), int(60 + 60 * (1 - abs(2 * a - 1))), int(255 * a))


# ── mode: sweep ────────────────────────────────────────────────────────────
def mode_sweep(args, P):
    import cv2
    f = h5py.File(args.path, 'r', locking=False)
    ends = f['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    s, e = int(starts[args.episode]), int(ends[args.episode])
    T_cam, fov, IH, IW = cam_of(f)
    col = int(f.attrs['fric_col'])
    mu_true = float(f['phys_params'][s, col])
    gid = P.global_id(args.path, args.episode)
    if gid is None:
        split = 'OUTSIDE the ckpt split (stream shard: fit once, then deleted)'
    else:
        split = 'val' if gid in P.val_ids else 'TRAIN (overlay flatters the model)'
    mus = [float(x) for x in grid_of(P, args.n_mu)]
    print('[sweep] episode %d (%s), true mu=%.3f, sweeping %s'
          % (args.episode, split, mu_true, np.round(mus, 2).tolist()))

    U = args.upscale
    vid = VideoOut(args.out, IW * U, IH * U, args.fps)
    n_drawn = 0
    for t in range(s, e, args.stride):
        b = P.batch(f, t)
        if b is None:
            break
        preds = P.predict_at(b, mus)                       # [K, N, H, 3]
        anchors = f['image_patch_centre_3d_position'][t].astype(np.float64)
        owner = f['patch_owner_body'][t].astype(np.int64)
        valid = f['track_valid_mask_gt'][t].any(-1)
        sel = np.nonzero(valid & (owner == OBJECT_BODY))[0]
        img = f['image'][t]
        im = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (IW * U, IH * U), interpolation=cv2.INTER_NEAREST)

        if len(sel):
            # ground truth first, so the mu curves draw on top of it
            gt = f['gt_3d_track'][t].astype(np.float64)
            for i in sel:
                pts, _ = project(anchors[i][None] + np.concatenate(
                    [np.zeros((1, 3)), gt[i]], 0), T_cam, fov, IH, IW)
                pp = (pts * U).astype(np.int32)
                cv2.polylines(im, [pp], False, (255, 255, 255), 2, cv2.LINE_AA)
            for k, mu in enumerate(mus):
                c = mu_color(k, len(mus))
                for i in sel:
                    pts, _ = project(anchors[i][None] + np.concatenate(
                        [np.zeros((1, 3)), preds[k, i]], 0), T_cam, fov, IH, IW)
                    pp = (pts * U).astype(np.int32)
                    cv2.polylines(im, [pp], False, c, 1, cv2.LINE_AA)

        y = 16
        cv2.putText(im, 'white = ground truth (true %s)'
                    % val_label(P, mu_true),
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                    cv2.LINE_AA)
        for k, mu in enumerate(mus):
            y += 15
            tag = val_label(P, mu) + ('   <- true' if abs(mu - mu_true) ==
                                      min(abs(np.array(mus) - mu_true)) else '')
            cv2.putText(im, tag, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        mu_color(k, len(mus)), 1, cv2.LINE_AA)
        cv2.putText(im, 'same frame, same actions -- only the %s INPUT differs'
                    % short_name(P),
                    (8, IH * U - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (200, 200, 200), 1, cv2.LINE_AA)
        vid.write(im)
        n_drawn += 1
    vid.close()
    f.close()
    print('[sweep] wrote %s (%d frames)' % (args.out, n_drawn))


# ── mode: curve ────────────────────────────────────────────────────────────
def mode_curve(args, P):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mus = [float(x) for x in grid_of(P, args.n_mu)]
    disp = np.zeros(len(mus))
    n = 0
    gt_mu, gt_disp = [], []
    rng = np.random.default_rng(0)

    for path in P.paths:
        with h5py.File(path, 'r', locking=False) as f:
            ends = f['episode_ends'][:]
            starts = np.concatenate([[0], ends[:-1]])
            col = int(f.attrs['fric_col'])
            off = 0
            for p2 in P.paths:
                if os.path.realpath(p2) == os.path.realpath(path):
                    break
                with h5py.File(p2, 'r', locking=False) as fp:
                    off += len(fp['episode_ends'])
            for k in range(len(ends)):
                if (off + k) not in P.val_ids:
                    continue
                s, e = int(starts[k]), int(ends[k])
                cand = np.arange(s, max(s + 1, e - P.H))
                if len(cand) == 0:
                    continue
                for t in rng.choice(cand, size=min(args.per_ep, len(cand)),
                                    replace=False):
                    b = P.batch(f, int(t))
                    if b is None:
                        continue
                    owner = f['patch_owner_body'][int(t)].astype(np.int64)
                    valid = f['track_valid_mask_gt'][int(t)].any(-1)
                    sel = np.nonzero(valid & (owner == OBJECT_BODY))[0]
                    if len(sel) == 0:
                        continue
                    pr = P.predict_at(b, mus)              # [K, N, H, 3]
                    # endpoint displacement magnitude, averaged over object patches
                    disp += np.linalg.norm(pr[:, sel, -1, :], axis=-1).mean(1)
                    g = f['gt_3d_track'][int(t)][sel, -1, :]
                    gt_mu.append(float(f['phys_params'][int(t), col]))
                    gt_disp.append(float(np.linalg.norm(g, axis=-1).mean()))
                    n += 1
                    if n >= args.max_frames:
                        break
                if n >= args.max_frames:
                    break
        if n >= args.max_frames:
            break

    assert n, 'no val frames found'
    disp = 1e3 * disp / n
    gt_mu, gt_disp = np.asarray(gt_mu), 1e3 * np.asarray(gt_disp)
    # GT binned by the episode's real mu -- the reference the model's response
    # should resemble. Binned, not scattered: each episode has its own geometry,
    # which dominates any single point.
    edges = grid_of(P, 9)   # bin the GT the way the grid was laid out
    idx = np.clip(np.digitize(gt_mu, edges) - 1, 0, len(edges) - 2)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    gtb = np.array([gt_disp[idx == i].mean() if (idx == i).any() else np.nan
                    for i in range(len(ctr))])

    slope = np.polyfit(mus, disp, 1)[0]
    print('[curve] %d frames | predicted displacement %.1f -> %.1f mm over mu '
          '%.2f -> %.2f  (slope %+.2f mm per unit mu)'
          % (n, disp[0], disp[-1], mus[0], mus[-1], slope))
    print('[curve] ground truth, same range: %.1f -> %.1f mm'
          % (np.nanmean(gtb[:2]), np.nanmean(gtb[-2:])))

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    ax.plot(mus, disp, 'o-', color='#c1121f', lw=2, ms=5,
            label='model prediction (counterfactual mu)')
    ax.plot(ctr, gtb, 's--', color='#264653', lw=1.6, ms=4, alpha=0.85,
            label='ground truth (binned by real mu)')
    ax.set_xlabel('%s fed to the model' % axis_label(P))
    ax.set_ylabel('mean predicted object displacement over 16 steps (mm)')
    ax.set_title('Does the model move the bottle less when told %s is higher?\n'
                 'slope %+.2f mm per unit'
                 % (short_name(P), slope), fontsize=10)
    if getattr(P, 'log', False):
        ax.set_xscale('log')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    print('[curve] wrote %s' % args.out)


# ── mode: heatmap ──────────────────────────────────────────────────────────
def mode_heatmap(args, P):
    import cv2
    f = h5py.File(args.path, 'r', locking=False)
    ends = f['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    s, e = int(starts[args.episode]), int(ends[args.episode])
    T_cam, fov, IH, IW = cam_of(f)
    U = args.upscale
    vid = VideoOut(args.out, IW * U, IH * U, args.fps)
    tally = {}
    n_fr = 0
    for t in range(s, e, args.stride):
        b = P.batch(f, t)
        if b is None:
            break
        pr = P.predict_at(b, [P.lo, P.hi])                 # [2, N, H, 3]
        sens = np.linalg.norm(pr[0] - pr[1], axis=-1).mean(-1)   # [N] metres
        owner = f['patch_owner_body'][t].astype(np.int64)
        valid = f['track_valid_mask_gt'][t].any(-1)
        for o in np.unique(owner[valid]):
            a = tally.setdefault(int(o), [0.0, 0])
            m = valid & (owner == o)
            a[0] += float(sens[m].sum()); a[1] += int(m.sum())

        img = f['image'][t]
        im = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
        im = cv2.resize(im, (IW * U, IH * U), interpolation=cv2.INTER_NEAREST)
        anchors = f['image_patch_centre_3d_position'][t].astype(np.float64)
        pts, _ = project(anchors, T_cam, fov, IH, IW)
        hi = max(float(sens[valid].max()) if valid.any() else 1e-9, 1e-9)
        for i in np.nonzero(valid)[0]:
            a = float(np.clip(sens[i] / hi, 0, 1))
            c = (int(255 * (1 - a)), int(40), int(255 * a))
            cv2.circle(im, tuple((pts[i] * U).astype(np.int32)),
                       max(2, int(2 + 6 * a)), c, -1, cv2.LINE_AA)
        cv2.putText(im, 'sensitivity to mu: |pred(%.2f) - pred(%.2f)|  '
                    'red = high' % (P.lo, P.hi), (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        vid.write(im)
        n_fr += 1
    vid.close()
    print('[heatmap] wrote %s (%d frames)' % (args.out, n_fr))
    print('\n  mean |pred(mu_lo) - pred(mu_hi)| per body, mm '
          '(0 table, 1 OBJECT, 7-12 franka):')
    for o in sorted(tally):
        v, c = tally[o]
        print('    body %-3d  n=%-7d  %.2f mm' % (o, c, 1e3 * v / max(c, 1)))
    print('  The object should dominate. The arm\'s future path is an INPUT, so '
          'friction\n  has nothing to add there -- a hot arm means mu is being '
          'used as a nuisance code.')
    f.close()


def mode_stills(args, P):
    """Quantify and then show the mu fan, zoomed onto the object."""
    import cv2
    f = h5py.File(args.path, 'r', locking=False)
    ends = f['episode_ends'][:]
    starts = np.concatenate([[0], ends[:-1]])
    s, e = int(starts[args.episode]), int(ends[args.episode])
    T_cam, fov, IH, IW = cam_of(f)
    col = int(f.attrs['fric_col'])
    mu_true = float(f['phys_params'][s, col])
    mus = [float(x) for x in grid_of(P, args.n_mu)]
    print('[stills] episode %d, true mu=%.3f, mu = %s'
          % (args.episode, mu_true, np.round(mus, 2).tolist()))
    # Two gaps, and the comparison between them is the point: FAN is how much
    # the friction input moves the prediction, ERR is how far the prediction sits
    # from the truth. Both measured at the ENDPOINT so they are the same
    # quantity -- comparing an endpoint fan against a per-step error would
    # understate the error roughly 2x.
    print('%6s %7s %9s %9s %9s %9s %8s'
          % ('frame', 'n_obj', 'gt_mm', 'fan_mm', 'err_mm', 'fan_px',
             'fan/err'))

    rows = []
    for t in range(s, e, args.stride):
        b = P.batch(f, t)
        if b is None:
            break
        pr = P.predict_at(b, mus)                          # [K, N, H, 3]
        anchors = f['image_patch_centre_3d_position'][t].astype(np.float64)
        owner = f['patch_owner_body'][t].astype(np.int64)
        valid = f['track_valid_mask_gt'][t].any(-1)
        sel = np.nonzero(valid & (owner == OBJECT_BODY))[0]
        if len(sel) == 0:
            continue
        end = anchors[sel][None] + pr[:, sel, -1, :]       # [K, S, 3] world
        # spread = largest disagreement between any two mu, per patch
        fan_m = np.max(np.linalg.norm(end[:, None] - end[None], axis=-1), (0, 1))
        px, _ = project(end.reshape(-1, 3), T_cam, fov, IH, IW)
        px = px.reshape(len(mus), len(sel), 2)
        fan_px = np.max(np.linalg.norm(px[:, None] - px[None], axis=-1), (0, 1))
        gt_ep = f['gt_3d_track'][t][sel, -1, :].astype(np.float64)
        gt_m = np.linalg.norm(gt_ep, axis=-1).mean()
        # prediction at the mu this episode ACTUALLY had
        k_true = int(np.argmin(np.abs(np.asarray(mus) - mu_true)))
        err_m = np.linalg.norm(pr[k_true, sel, -1, :] - gt_ep, axis=-1).mean()
        rows.append((t - s, len(sel), 1e3 * gt_m, 1e3 * fan_m.mean(),
                     1e3 * err_m, fan_px.mean()))
        print('%6d %7d %9.1f %9.1f %9.1f %9.2f %8.2f'
              % (rows[-1][0], rows[-1][1], rows[-1][2], rows[-1][3],
                 rows[-1][4], rows[-1][5],
                 rows[-1][3] / max(rows[-1][4], 1e-9)))

        if args.png_every and ((t - s) % args.png_every == 0):
            img = f['image'][t]
            im = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
            # crop tightly around the object's patches, then upscale hard --
            # the whole point is that the effect is a few pixels wide.
            ac_px, _ = project(anchors[sel], T_cam, fov, IH, IW)
            allp = np.concatenate([ac_px, px.reshape(-1, 2)], 0)
            lo = np.maximum(allp.min(0) - args.pad, 0)
            hi = np.minimum(allp.max(0) + args.pad, [IW - 1, IH - 1])
            Z = args.zoom
            crop = im[int(lo[1]):int(hi[1]) + 1, int(lo[0]):int(hi[0]) + 1]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, None, fx=Z, fy=Z,
                              interpolation=cv2.INTER_NEAREST)
            gt = f['gt_3d_track'][t].astype(np.float64)
            for i in sel:
                q, _ = project(anchors[i][None] + np.concatenate(
                    [np.zeros((1, 3)), gt[i]], 0), T_cam, fov, IH, IW)
                cv2.polylines(crop, [((q - lo) * Z).astype(np.int32)], False,
                              (255, 255, 255), 2, cv2.LINE_AA)
            for k in range(len(mus)):
                c = mu_color(k, len(mus))
                for i_, i in enumerate(sel):
                    q, _ = project(anchors[i][None] + np.concatenate(
                        [np.zeros((1, 3)), pr[k, i]], 0), T_cam, fov, IH, IW)
                    cv2.polylines(crop, [((q - lo) * Z).astype(np.int32)], False,
                                  c, 1, cv2.LINE_AA)
            cv2.putText(crop, 'zoom x%d  fan %.1f mm = %.1f px (native)'
                        % (Z, 1e3 * fan_m.mean(), fan_px.mean()),
                        (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            out = '%s_f%03d.png' % (os.path.splitext(args.out)[0], t - s)
            cv2.imwrite(out, crop)

    a = np.asarray(rows, float)
    gtm, fan, err, px = a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean(), a[:, 5].mean()
    print('\n[stills] %d frames | gt endpoint %.1f mm | fan %.1f mm | error '
          '%.1f mm | fan %.2f px native (%.0f px at x%d)'
          % (len(a), gtm, fan, err, px, px * args.zoom, args.zoom))
    print('[stills] fan / error       = %.2f   <- how the mu spread compares '
          'with how wrong the model is' % (fan / max(err, 1e-9)))
    print('[stills] error / gt        = %.2f   <- relative error at the endpoint'
          % (err / max(gtm, 1e-9)))
    print('[stills] fan / gt          = %.2f   <- how much of the real motion '
          'friction accounts for' % (fan / max(gtm, 1e-9)))
    print('[stills] A fan much SMALLER than the error is expected, not broken: '
          'friction is one\n         small term in the object\'s motion and '
          'the error is dominated by everything else.')
    f.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--mode', choices=('sweep', 'curve', 'heatmap', 'stills'),
                    default='sweep')
    ap.add_argument('--png-every', type=int, default=0)
    ap.add_argument('--zoom', type=int, default=8)
    ap.add_argument('--pad', type=int, default=6)
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--n-mu', type=int, default=5)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--fps', type=int, default=10)
    ap.add_argument('--upscale', type=int, default=3)
    ap.add_argument('--per-ep', type=int, default=4,
                    help='curve: frames sampled per val episode')
    ap.add_argument('--max-frames', type=int, default=400,
                    help='curve: total frames')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--infer-steps', type=int, default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    P = FricPredictor(args.ckpt, args.device, args.infer_steps)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    {'sweep': mode_sweep, 'curve': mode_curve, 'heatmap': mode_heatmap,
     'stills': mode_stills}[args.mode](args, P)


if __name__ == '__main__':
    main()
