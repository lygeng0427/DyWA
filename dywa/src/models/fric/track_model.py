#!/usr/bin/env python3
"""`models.gap.TrackModel` + conditioning on a scalar physics parameter.

Subclass, not a copy. Both trunks already slice the patch tokens off
`self.n_prefix`, and the loss/normalizer/head paths are unchanged, so the whole
delta is `__init__` plus three overrides. That matters for the experiment this
exists for: `phys_cond='none'` and `phys_cond='both'` must differ in the
conditioning and in nothing else, or the A/B measures a refactor.

TWO CONDITIONING PATHS, and they are not redundant
--------------------------------------------------
  token  the friction embedding becomes an extra prefix token, so every patch
         can attend to it and weight it per patch. This is "in-context"
         conditioning, and it is what a diffusion trunk normally gets.
  film   the same embedding drives a per-layer scale/shift on the FFN branch of
         every encoder and decoder block (`common.EncoderLayer(film_dim=...)`,
         applied pre-residual with no post-FiLM norm precisely so the
         conditioning cannot be normalized away).

A token can be ignored by attention; FiLM cannot be routed around. Running
`both` by default means a null result is a statement about the DATA rather than
about one conditioning mechanism being weak, and `token` / `film` stay
selectable afterwards to attribute whatever gain shows up.

NORMALISATION USES THE KNOWN GRID, NOT DATASET STATISTICS. The held-out-mu run
trains on a set that is missing 8 of the 32 values, so dataset mean/std would
differ there and the conditioning input would mean something different than in
the runs it is compared against.
"""
import math
from typing import Optional

import torch as th
import torch.nn as nn

from models.gap.common import Encoder, EncoderLayer, sinusoidal_2d
from models.gap.track_model import TrackModel, GRID_H, GRID_W, N_PATCH

PHYS_CONDS = ('none', 'token', 'film', 'both')

# The gap model offers realized(7) / command(20) / both(27) / realized_gain(21).
# All of those except plain `realized` carry the 14 joint KP/KD dims, and those
# dims LEAK the swept variable: the teacher sees privileged phys_params and
# modulates stiffness with them (ridge -> log mass, split by episode: gains
# R2 +0.30, commanded dpos/dori -0.02, realized path -0.04). So a `command`
# ablation does not remove the leak, it only removes the realized path.
#
# `command_pose` is the missing arm: the policy's 6 commanded dpos/dori dims and
# nothing else. It is the one input that is BOTH causally upstream of contact
# (an intent, not an outcome) and leak-free, so it is the only setting where the
# model must genuinely infer the dynamics rather than read them off what already
# happened. Added here rather than in models/gap, which stays unedited.
CMD_POSE = 'command_pose'
ACTION_MODES = ('realized', 'command', 'both', 'realized_gain', CMD_POSE)


class FricTrackModel(TrackModel):
    def __init__(self, *args,
                 phys_cond: str = 'both',
                 # NO DEFAULTS. These used to default to friction's 0.1/1.5,
                 # which for a mass grid over [0.05, 1.0] would squash every
                 # conditioning input into roughly [-1.0, -0.29] -- two thirds of
                 # the dynamic range gone, with no error anywhere.
                 phys_lo: float = None,
                 phys_hi: float = None,
                 phys_log: bool = False,
                 action_mode: str = 'realized_gain',
                 # Width of the vector FiLM reads. None -> d_model, which is
                 # what `phys_mlp` emits and what every existing checkpoint was
                 # built with. A caller that conditions on a RAW vector (the
                 # latent subclass) passes its width here instead, so FiLM reads
                 # it directly and the embedding MLP disappears -- which is what
                 # the pick-bar reference does (`cond_dim=latent_dim`).
                 film_dim: Optional[int] = None,
                 encoder_depth: int = 2,
                 decoder_depth: int = 4,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 **kwargs):
        assert action_mode in ACTION_MODES, action_mode
        # The base class asserts its own four modes and sizes `action_encoder`
        # from them, so build as 'command' (same source tensor) and narrow the
        # encoder afterwards. Anything else would mean editing models/gap.
        self._cmd_pose = (action_mode == CMD_POSE)
        super().__init__(*args, encoder_depth=encoder_depth,
                         decoder_depth=decoder_depth, nhead=nhead,
                         dim_feedforward=dim_feedforward, dropout=dropout,
                         action_mode=('command' if self._cmd_pose
                                      else action_mode),
                         **kwargs)
        if self._cmd_pose:
            self.action_mode = CMD_POSE
            self.action_encoder = nn.Linear(6, self.d_model)
        assert phys_cond in PHYS_CONDS, phys_cond
        if phys_cond != 'none':
            assert phys_lo is not None and phys_hi is not None, (
                'phys_lo/phys_hi are required -- pass the grid endpoints from '
                'the dataset attrs (models.fric.dataset.read_grid)')
        phys_lo = 0.0 if phys_lo is None else phys_lo
        phys_hi = 1.0 if phys_hi is None else phys_hi
        # An unconditioned model has no phys input at all, so the map over it is
        # meaningless; carrying phys_log through would only let a placeholder
        # phys_lo=0 trip the log assert below.
        phys_log = phys_log and phys_cond != 'none'
        assert phys_hi > phys_lo, (phys_lo, phys_hi)
        assert not phys_log or phys_lo > 0, 'a log map needs phys_lo > 0'
        self.phys_cond = phys_cond
        self.phys_lo, self.phys_hi = float(phys_lo), float(phys_hi)
        self.phys_log = bool(phys_log)
        d = self.d_model

        if phys_cond != 'none':
            self.phys_mlp = nn.Sequential(
                nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        if phys_cond in ('token', 'both'):
            self.phys_pos = nn.Embedding(1, d)
            # [CLS, PHYS, state, H actions, patches]
            self.n_prefix = 3 + self.H
        if phys_cond in ('film', 'both'):
            # Rebuilt with the FiLM hook live. Same shapes and same init
            # distribution as the base class's, so `none` vs `film` is not also
            # a change of parameter count in the attention/FFN path.
            fd = d if film_dim is None else int(film_dim)
            self.film_dim = fd
            self.ctx = Encoder(
                lambda: EncoderLayer(d, nhead, dim_feedforward, dropout,
                                     film_dim=fd), encoder_depth)
            self.dec = Encoder(
                lambda: EncoderLayer(d, nhead, dim_feedforward, dropout,
                                     film_dim=fd), decoder_depth)

    # -- action tokens ------------------------------------------------------
    def action_features(self, batch):
        if self.action_mode == CMD_POSE:
            # dims 0:6 only -- 3 dpos, 3 dori. Clipped for the same reason the
            # base class clips: the policy emits an unsquashed Gaussian mean and
            # the simulator clamps before use, so values outside the box never
            # physically happened (48% of them are outside it).
            return batch['action_cmd_seq'][..., :6].clamp(-1.0, 1.0)
        return super().action_features(batch)

    def action_xyz(self, batch):
        """RoPE coordinate for each action token.

        MUST be overridden. The base class only integrates the commanded deltas
        when `action_mode == 'command'` exactly, and otherwise falls through to
        `action_seq[..., :3]` -- the REALIZED path. For `command_pose` that
        default would feed the realized trajectory in through the positional
        encoding, leaking back the very signal this arm exists to withhold,
        silently and with no shape error.
        """
        if self.action_mode == CMD_POSE:
            d = batch['action_cmd_seq'][..., :3].clamp(-1.0, 1.0) * self.max_pos
            return batch['agent_pos'][:, None, :3] + th.cumsum(d, dim=1)
        return super().action_xyz(batch)

    # -- conditioning -------------------------------------------------------
    def phys_vec(self, batch) -> Optional[th.Tensor]:
        """[B, D] embedding of the swept physics scalar, or None if unconditioned.

        Recomputed in each of the three call sites rather than cached on `self`:
        the MLP is negligible next to a 409-token encoder, and a stashed tensor
        would silently go stale between a `predict` and the DDIM loop that calls
        the decoder ten times.

        THE LOG MAP LIVES HERE AND NOWHERE ELSE. A mass grid is geometric,
        because mass enters through a ~ F/m, so equal RELATIVE steps are what the
        model should see evenly spaced. But `phys_params`, the HDF5 `fric_grid`
        attr, `bucketize` and `assert_phys_grid` all speak raw kg -- putting the
        log anywhere upstream would leave two components disagreeing about what
        `phys` means, and the failure mode is a silently compressed conditioning
        input rather than an exception.
        """
        if self.phys_cond == 'none':
            return None
        mu = batch['phys'].reshape(-1, 1).float()
        if self.phys_log:
            mu = mu.clamp_min(1e-8).log()
            lo, hi = math.log(self.phys_lo), math.log(self.phys_hi)
        else:
            lo, hi = self.phys_lo, self.phys_hi
        mu_n = 2.0 * (mu - lo) / (hi - lo) - 1.0
        return self.phys_mlp(mu_n)

    def _film(self, batch):
        return self.phys_vec(batch) if self.phys_cond in ('film', 'both') else None

    # -- context ------------------------------------------------------------
    def encode_context(self, batch):
        if self.phys_cond == 'none':
            return super().encode_context(batch)

        dino = batch['dino_features']
        B = dino.shape[0]
        if dino.dim() == 4:
            dino = dino.reshape(B, -1, dino.shape[-1])
        patches = self.dino_proj(dino)
        state = self.state_encoder(batch['agent_pos']).unsqueeze(1)
        if self.hist_k > 0:
            pc = batch['patch_centers']
            ph = batch['patch_centers_hist']
            prev = th.cat([pc.unsqueeze(1), ph[:, :-1]], dim=1)
            dv = (prev - ph).permute(0, 2, 1, 3).reshape(B, pc.shape[1], -1)
            patches = patches + self.vel_proj(dv)
            ap, ah = batch['agent_pos'], batch['agent_pos_hist']
            aprev = th.cat([ap.unsqueeze(1), ah[:, :-1]], dim=1)
            state = state + self.state_vel_proj(
                (aprev - ah).reshape(B, -1)).unsqueeze(1)
        actions = self.action_encoder(self.action_features(batch))

        cond = self.phys_vec(batch)
        parts = [self.cls.expand(B, -1, -1)]
        pos = [self.cls_pos.weight.unsqueeze(0).expand(B, -1, -1)]
        if self.phys_cond in ('token', 'both'):
            parts.append(cond.unsqueeze(1))
            pos.append(self.phys_pos.weight.unsqueeze(0).expand(B, -1, -1))
        parts += [state, actions, patches]
        pos += [self.state_pos.weight.unsqueeze(0).expand(B, -1, -1),
                self.action_pos.weight.unsqueeze(0).expand(B, -1, -1),
                sinusoidal_2d(GRID_H, GRID_W, self.d_model,
                              dino.device).unsqueeze(0).expand(B, -1, -1)]
        x = th.cat(parts, dim=1)
        pos = th.cat(pos, dim=1)

        mem = self.ctx(x, pos=pos,
                       **self._rope_obs(batch, B, x.shape[1]),
                       cond=self._film(batch))
        # The prefix width and the token layout are set in two places; if they
        # ever disagree every patch is scored against the wrong track and the
        # loss still goes down. Cheap to check, silent to get wrong.
        assert mem[:, self.n_prefix:].shape[1] == N_PATCH, (
            'prefix/token layout mismatch: %d patch tokens after n_prefix=%d'
            % (mem[:, self.n_prefix:].shape[1], self.n_prefix))
        return mem

    def _rope_obs(self, batch, B, n_tok):
        """As the base class, shifted by the extra PHYS token.

        Friction has no position, so like CLS the phys token is left unrotated.
        """
        if self.rope is None or self.phys_cond not in ('token', 'both'):
            return super()._rope_obs(batch, B, n_tok)
        pc = batch['patch_centers']
        H = self.H
        coords = pc.new_zeros(B, n_tok, 4)
        mask = th.zeros(B, n_tok, dtype=th.bool, device=pc.device)
        coords[:, 2:3, :3] = batch['agent_pos'][:, None, :3]     # state
        coords[:, 3:3 + H, :3] = self.action_xyz(batch)          # actions
        coords[:, 3:3 + H, 3] = th.arange(1, H + 1, device=pc.device,
                                          dtype=coords.dtype)
        coords[:, 3 + H:, :3] = pc                               # patches
        mask[:, 2:] = True                       # CLS and PHYS stay unrotated
        return dict(rope=self.rope, cos_sin=self.rope.build_cos_sin(coords),
                    rope_mask=mask)

    # -- trunks (identical to the base class except for `cond=`) ------------
    def _decode_transformer(self, mem, batch):
        B = mem.shape[0]
        x = mem[:, self.n_prefix:, :] + self.patch_query.unsqueeze(0)
        x = self.dec(x, **self._rope_patches(batch, B, x.shape[1]),
                     cond=self._film(batch))
        return self._emit(x, batch)

    def _decode_dit(self, mem, batch, noisy, t):
        from models.gap.common import timestep_embedding
        B, N = noisy.shape[0], noisy.shape[1]
        tok = self.track_embed(noisy.reshape(B, N, -1))
        fused = self.fuse(th.cat([mem[:, self.n_prefix:, :], tok], dim=-1))
        fused = fused + self.patch_query.unsqueeze(0)
        te = self.t_embed(timestep_embedding(t, 256)).unsqueeze(1)
        seq = th.cat([fused, te], dim=1)
        out = self.dec(seq, **self._rope_patches(batch, B, N + 1, n_extra=1),
                       cond=self._film(batch))
        return self._emit(out[:, :N, :], batch)
