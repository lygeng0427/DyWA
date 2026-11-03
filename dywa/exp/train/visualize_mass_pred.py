#!/usr/bin/env python3
"""Overlay a MASS-conditioned model's predicted object tracks on the video.

Same picture as `visualize_gap_tracks.py --pred`: cyan = model, green->red ramp =
ground truth. That script builds a `models.gap.TrackModel`, which cannot load a
`FricTrackModel` checkpoint -- the fric model carries extra `phys_mlp` / FiLM /
`phys_pos` weights and needs a `phys` input that the gap batch never supplies.

So this file swaps in a Predictor that does both, and then calls the unmodified
visualizer. `visualize_gap_tracks.py` is part of the gap pipeline and is imported,
never edited -- the same rule the rest of the fric/mass work follows.

The mass fed to the model is the episode's TRUE value read from `phys_params`,
not a guess: this is the "what does the model actually predict" view, not the
counterfactual sweep (that is `visualize_fric_conditioning.py --mode sweep`).

Use `--only-bodies 1` -- the franka links carry ~40x more patches with much
larger displacements and completely bury the object.

    python3 visualize_mass_pred.py output/mass/tracks_m_a.hdf5 \
      --episode 7 --only-bodies 1 --upscale 3 \
      --pred output/mass/ckpt/transformer_both/best.ckpt \
      --out output/mass/pred_ep7.mp4
"""
import os
import sys

import numpy as np

sys.path.insert(0, '/home/user/DyWA/dywa/src')

import visualize_gap_tracks as V                                  # noqa: E402


class FricPredictor(V.Predictor):
    """`Predictor`, but backed by FricTrackModel and feeding it the true mass."""

    def __init__(self, ckpt_path, h5_path, episode, device='cuda:0',
                 infer_steps=None):
        import torch as th
        import h5py as _h5
        from models.gap.common import FieldNorm
        from models.fric.track_model import FricTrackModel
        self.th = th
        self.dev = th.device(device)

        ck = th.load(ckpt_path, map_location='cpu')
        c = ck['cfg']
        self.cond = c.get('phys_cond', 'none')
        m = FricTrackModel(trunk=ck['trunk'], d_model=c['d_model'],
                           action_mode=c.get('action_mode', 'realized_gain'),
                           obj_weight=c.get('obj_weight', 1.0),
                           phys_cond=self.cond,
                           phys_lo=c['phys_lo'], phys_hi=c['phys_hi'],
                           phys_log=c.get('phys_log', False))
        m.normalizer.add('gt_3d_track',
                         FieldNorm(3).fit_isotropic_rms(ck['track_rms']))
        # strict: a silent key mismatch would leave part of the net random and
        # the overlay would look plausible while meaning nothing.
        m.load_state_dict(ck['model'], strict=True)
        if ck['trunk'] == 'dit':
            m.num_inference_steps = infer_steps or 1
        self.model = m.to(self.dev).eval()
        self.trunk = ck['trunk']
        self.action_mode = c.get('action_mode', 'realized_gain')
        self.phys_col = int(c.get('phys_col', 0))
        self.cut = None

        # Global episode id: the dataset concatenates files in ckpt['paths']
        # order, so a local index must be offset by the preceding files' episode
        # counts or the reported split membership is wrong.
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
        print('  [pred] %s trunk, phys_cond=%s, action=%s, episode global id %d '
              '(%s split)' % (self.trunk, self.cond, self.action_mode, gid, split))
        if split == 'train':
            print('  [pred] WARNING: this episode was TRAINED on -- the overlay '
                  'flatters the model')

    def _batch(self, f, t):
        b = super()._batch(f, t)
        if b is None:
            return None
        if self.cond != 'none':
            mass = float(f['phys_params'][t, self.phys_col])
            b['phys'] = self.th.tensor([[mass]], dtype=self.th.float32,
                                       device=self.dev)
        return b


def main():
    V.Predictor = FricPredictor
    V.main()


if __name__ == '__main__':
    main()
