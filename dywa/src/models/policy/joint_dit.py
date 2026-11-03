#!/usr/bin/env python3
"""GAPPolicy's joint action + 3D-track denoiser, de-bimanualised, with an
optional dynamics-belief conditioning input.

This is the OUTER model of the meta-TTT pipeline. The inner model
(`models.fric.LatentTrackModel`) fits a belief `q` from already-elapsed motion;
this one consumes `q` and denoises the policy's next action chunk and the
scene's next tracks together.

WHAT IS PORTED FROM GAP, AND WHAT IS NOT
----------------------------------------
Ported, because it is the architecture:
  * a context encoder over `[CLS, state, 391 patches]` and NOTHING ELSE,
  * per-patch fusion of the contextualised scene feature with the embedded
    NOISY track, which is why there is no cross-attention anywhere,
  * one self-attention stack over `[16 action, 391 fused track, 1 state,
    1 timestep]` = 409 tokens, denoised at ONE shared timestep,
  * 4-D RoPE whose action-token coordinate is tau-blended from the CURRENT
    NOISY chunk -- never ground truth, identical at train and inference, so
    there is no exposure-bias asymmetry.

Not ported:
  * the per-arm token split. DyWA has one 7-D EEF, so the 16-D state and
    2H action tokens collapse to 1 and H.
  * `diffusers`. There is none in this container, so DDIM is hand-rolled with a
    cosine alpha_bar and `prediction_type='sample'`, exactly as
    `models.gap.track_model.TrackModel` already does. `_cosine_alpha_bar` is
    imported from there rather than copied, so the two schedules cannot drift.

THE CONTEXT ENCODER CARRIES NO ACTION TOKENS, AND THAT IS THE WHOLE LEAK STORY.
`models.gap.TrackModel` feeds the next H actions as encoder tokens, because for
a passive track predictor the future action is a legitimate input -- and it is
worth ~50x (val 2.008 -> 0.038 without it). Here the action is what the model
must PRODUCE, so the only path by which it may enter is the noised action token
inside the DiT. `n_prefix = 2`, and `_probe_policy_smoke.py` asserts that
perturbing `action_cmd_seq` leaves the context memory bit-identical. A test is
cheaper than trusting the token layout.

CONDITIONING READS THE RAW VECTOR, WITH NO EMBEDDING MLP, as the pick-bar
reference does (`ttt_with_dyn_bar.py:268,344` pass `cond_dim=latent_dim` and
`:300,365` call the block with `q` itself). Measured at the real init, the
embedded route buys nothing that matters: d(gamma)/dq is within 13% of the
direct route, for 62x the parameters. `cond_dim` is therefore `z_dim` in
`latent` mode.

  'none'    no FiLM anywhere. The unconditioned baseline, and `film_dim=0`
            makes it bit-identical to a model that never had the hook.
  'mass'    the TRUE object mass, normalised on the known grid and lifted to
            `cond_dim` by a small MLP. This is the POSITIVE CONTROL -- the
            configuration that measured 6.4% on track prediction at 12K
            episodes -- not a deployable mode.
  'latent'  `batch['q']`, raw, `[B, cond_dim]`. The belief.

A NOTE ON THE SHARED TIMESTEP. GAP noises the action and the track at the same
`t` so the two modalities denoise in lockstep, and that is reproduced here. The
consequence is that "action clean, track noisy" is a state the model NEVER sees
in training, so a teacher-forced track evaluation is off-distribution. It is
still the closest comparable to an action-conditioned track model, but it is an
approximation and must be reported as one. Independent per-channel timesteps
would make it exact; that is a deliberate deviation from GAP and is not done
here.
"""
import math
from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn

from models.gap.common import (Encoder, EncoderLayer, FieldNorm, Normalizer,
                               RoPE4D, sinusoidal_2d, timestep_embedding)
from models.gap.track_model import TrackModel

N_PATCH = 391
GRID_H, GRID_W = 17, 23
# nvdr body ids in this scene: 0 table, 1 the pushed object, 7-12 franka links.
OBJECT_BODY = 1
COND_MODES = ('none', 'mass', 'latent')


class JointPolicyDiT(nn.Module):
    def __init__(self,
                 dino_dim: int = 64,
                 d_model: int = 512,
                 state_dim: int = 7,
                 action_dim: int = 20,
                 horizon: int = 16,
                 encoder_depth: int = 2,
                 decoder_depth: int = 4,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_train_timesteps: int = 100,
                 num_inference_steps: int = 10,
                 track_temporal_dim: int = 512,
                 clip_sample_range: float = 10.0,
                 obj_weight: float = 9.0,
                 movement_tau: float = 0.01,
                 track_loss_weight: float = 1.0,
                 cond: str = 'none',
                 cond_dim: int = 32,
                 phys_lo: float = None,
                 phys_hi: float = None,
                 phys_log: bool = True,
                 # `EncoderLayer` initialises FiLM with xavier_normal(gain=0.5),
                 # which is calibrated for a d_model-wide conditioning vector.
                 # A narrow one (cond_dim=32) emits gamma ~ 0.012 where a
                 # TRAINED, working path sits at 0.25-0.43 (measured on the
                 # gate's mass arm), so the belief cannot move the loss at all
                 # and the gradient that would grow the heads is itself tiny.
                 # Rescaling the init breaks that bootstrap. Note |q0| does NOT
                 # do this job: FiLM gives gamma = W q, so a CHANGE in the
                 # belief moves gamma by W dq -- it depends on the travel
                 # distance, not on where q sits (measured: |q0| 0.56 -> 11.1
                 # moved leverage 0.167% -> 0.248%).
                 film_init_scale: float = 1.0,
                 max_pos: float = 0.06):
        super().__init__()
        assert cond in COND_MODES, cond
        self.cond = cond
        self.cond_dim = int(cond_dim) if cond != 'none' else 0
        self.H = horizon
        self.d_model = d_model
        self.action_dim = action_dim
        self.max_pos = max_pos
        self.obj_weight = obj_weight
        self.movement_tau = movement_tau
        self.track_loss_weight = track_loss_weight
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.clip_sample_range = clip_sample_range

        film = self.cond_dim
        if cond == 'mass':
            # Required, not defaulted: friction's old 0.1/1.5 would squash a
            # mass grid over [0.1, 1.2] into roughly [-1.0, -0.29] with nothing
            # raising (models/fric/track_model.py says the same).
            assert phys_lo is not None and phys_hi is not None, \
                'phys_lo/phys_hi are required in cond=mass'
            self.phys_lo, self.phys_hi, self.phys_log = phys_lo, phys_hi, phys_log
            self.phys_mlp = nn.Sequential(
                nn.Linear(1, self.cond_dim), nn.SiLU(),
                nn.Linear(self.cond_dim, self.cond_dim))
        else:
            self.phys_lo, self.phys_hi, self.phys_log = phys_lo, phys_hi, phys_log

        # -- context encoder: [CLS, state, 391 patches] ---------------------
        self.dino_proj = nn.Linear(dino_dim, d_model)
        self.state_encoder = nn.Linear(state_dim, d_model)
        self.cls = nn.Parameter(th.randn(1, 1, d_model) * 0.02)
        self.cls_pos = nn.Embedding(1, d_model)
        self.state_pos = nn.Embedding(1, d_model)
        self.n_prefix = 2
        self.ctx = Encoder(
            lambda: EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                 film_dim=film), encoder_depth)

        # -- diffusion tokens ------------------------------------------------
        self.action_embed = nn.Sequential(
            nn.Linear(action_dim, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model))
        self.action_pos = nn.Embedding(horizon, d_model)
        # Flatten the H steps into one vector per patch, as GAP does: no
        # per-step weight sharing and no time embedding. Valid only because H
        # is fixed for every sample.
        self.track_embed = nn.Sequential(
            nn.Linear(horizon * 3, track_temporal_dim), nn.GELU(),
            nn.Linear(track_temporal_dim, d_model))
        self.fuse = nn.Linear(2 * d_model, d_model)
        self.patch_query = nn.Parameter(th.randn(N_PATCH, d_model) * 0.02)
        self.t_embed = nn.Sequential(
            nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.t_pos = nn.Embedding(1, d_model)
        self.dit = Encoder(
            lambda: EncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                 film_dim=film), decoder_depth)

        self.action_head = nn.Linear(d_model, action_dim)
        self.track_head = nn.Linear(d_model, horizon * 3)

        if cond != 'none' and film_init_scale != 1.0:
            with th.no_grad():
                for n_, p_ in self.named_parameters():
                    if '.film.' in n_:
                        p_.mul_(float(film_init_scale))
        self.film_init_scale = float(film_init_scale)

        self.rope = RoPE4D(d_model // nhead)
        self.normalizer = Normalizer()
        self.register_buffer('alpha_bar',
                             TrackModel._cosine_alpha_bar(num_train_timesteps),
                             persistent=False)

    # -- conditioning --------------------------------------------------------
    def cond_vec(self, batch) -> Optional[th.Tensor]:
        """[B, cond_dim], or None in 'none' mode.

        Recomputed per call rather than cached: the encoder and the DiT each
        need it, and a cached value would go stale the moment a caller swapped
        `q` between the two -- which is exactly what the inner loop does.
        """
        if self.cond == 'none':
            return None
        B = batch['patch_centers'].shape[0]
        if self.cond == 'latent':
            q = batch['q']
            assert q.shape == (B, self.cond_dim), (q.shape, B, self.cond_dim)
            return q
        p = batch['phys'].reshape(B, 1).float()
        lo, hi = self.phys_lo, self.phys_hi
        if self.phys_log:
            p = p.clamp(min=1e-6).log()
            lo, hi = math.log(lo), math.log(hi)
        return self.phys_mlp(2.0 * (p - lo) / (hi - lo) - 1.0)

    # -- context -------------------------------------------------------------
    def encode_context(self, batch) -> th.Tensor:
        """[B, 2 + 391, D].

        `batch['action_cmd_seq']` is NOT read here and must never be. The smoke
        probe asserts it.
        """
        dino = batch['dino_features']
        B = dino.shape[0]
        if dino.dim() == 4:
            dino = dino.reshape(B, -1, dino.shape[-1])
        patches = self.dino_proj(dino)
        state = self.state_encoder(batch['agent_pos']).unsqueeze(1)
        x = th.cat([self.cls.expand(B, -1, -1), state, patches], dim=1)
        pos = th.cat([
            self.cls_pos.weight.unsqueeze(0).expand(B, -1, -1),
            self.state_pos.weight.unsqueeze(0).expand(B, -1, -1),
            sinusoidal_2d(GRID_H, GRID_W, self.d_model,
                          x.device).unsqueeze(0).expand(B, -1, -1)], dim=1)

        pc = batch['patch_centers']
        coords = pc.new_zeros(B, x.shape[1], 4)
        mask = th.zeros(B, x.shape[1], dtype=th.bool, device=pc.device)
        coords[:, 1:2, :3] = batch['agent_pos'][:, None, :3]
        coords[:, 2:, :3] = pc
        mask[:, 1:] = True                       # CLS has no coordinate
        mem = self.ctx(x, pos=pos, rope=self.rope,
                       cos_sin=self.rope.build_cos_sin(coords),
                       rope_mask=mask, cond=self.cond_vec(batch))
        assert mem.shape[1] == self.n_prefix + N_PATCH, mem.shape
        return mem

    # -- the action tokens' RoPE coordinate ---------------------------------
    def action_xyz(self, noisy_a: th.Tensor, agent_pos: th.Tensor,
                   t: th.Tensor) -> th.Tensor:
        """[B, H, 3] world xyz per action token, tau-blended (GAP's a^tau).

        Interpolates from the FIXED current EEF position toward the position
        implied by the CURRENT NOISY chunk, weighted by how far denoising has
        come. It never touches ground truth and never touches a model output,
        so it is the identical computation at training and at inference.

        DyWA's command is a per-step DELTA in env-normalised units, not an
        absolute pose, so the position is integrated the way
        `TrackModel.action_xyz` does for `command` mode. Using the realized path
        instead would leak the very thing this model has to produce.
        """
        a = self.normalizer['action'].unnormalize(noisy_a) \
            if 'action' in self.normalizer else noisy_a
        d = a[..., :3].clamp(-1.0, 1.0) * self.max_pos
        est = agent_pos[:, None, :3] + th.cumsum(d, dim=1)       # [B, H, 3]
        tau = (1.0 - t.float() / self.num_train_timesteps).view(-1, 1, 1)
        g = agent_pos[:, None, :3].expand_as(est)
        return (1.0 - tau) * g + tau * est

    # -- one denoising step --------------------------------------------------
    def _decode(self, mem, batch, noisy_a, noisy_trk, t):
        """-> (action [B, H, A], track [B, N, H, 3]), both NORMALISED x0."""
        B = noisy_a.shape[0]
        H, N = self.H, N_PATCH
        cond = self.cond_vec(batch)

        act_tok = self.action_embed(noisy_a) \
            + self.action_pos.weight.unsqueeze(0)                # [B, H, D]
        trk_tok = self.track_embed(noisy_trk.reshape(B, N, -1))
        # The per-patch concat IS the conditioning on the scene -- this is why
        # GAP needs no cross-attention to a separate memory.
        fused = self.fuse(th.cat([mem[:, self.n_prefix:, :], trk_tok], dim=-1))
        fused = fused + self.patch_query.unsqueeze(0)            # [B, N, D]
        state_tok = mem[:, 1:2, :]                               # [B, 1, D]
        t_tok = self.t_embed(timestep_embedding(t, 256)).unsqueeze(1)

        seq = th.cat([act_tok, fused, state_tok, t_tok], dim=1)
        pos = th.cat([
            self.action_pos.weight.unsqueeze(0).expand(B, -1, -1),
            sinusoidal_2d(GRID_H, GRID_W, self.d_model,
                          seq.device).unsqueeze(0).expand(B, -1, -1),
            self.state_pos.weight.unsqueeze(0).expand(B, -1, -1),
            self.t_pos.weight.unsqueeze(0).expand(B, -1, -1)], dim=1)

        pc = batch['patch_centers']
        n_tok = seq.shape[1]
        coords = pc.new_zeros(B, n_tok, 4)
        mask = th.zeros(B, n_tok, dtype=th.bool, device=pc.device)
        coords[:, :H, :3] = self.action_xyz(noisy_a, batch['agent_pos'], t)
        coords[:, :H, 3] = th.arange(1, H + 1, device=pc.device,
                                     dtype=coords.dtype)
        coords[:, H:H + N, :3] = pc
        coords[:, H + N:H + N + 1, :3] = batch['agent_pos'][:, None, :3]
        mask[:, :H + N + 1] = True               # the timestep token stays off
        out = self.dit(seq, pos=pos, rope=self.rope,
                       cos_sin=self.rope.build_cos_sin(coords),
                       rope_mask=mask, cond=cond)

        a_out = self.action_head(out[:, :H, :])
        t_out = self.track_head(out[:, H:H + N, :]).reshape(B, N, H, 3)
        return a_out, t_out

    # -- weighting -----------------------------------------------------------
    def patch_weights(self, batch) -> th.Tensor:
        """[B, N] -- movement weighting x validity x obj_weight.

        Identical in form to `TrackModel.patch_weights` and for the identical
        reason: moving patches are ~90% franka links and ~10% object, both
        saturate the sigmoid, so without `obj_weight` ~90% of the auxiliary
        gradient goes to arm kinematics. Here that is worse than in the track
        model, because the arm's track is the action's OWN consequence and the
        model is already emitting that action.
        """
        gt = batch['gt_3d_track']
        disp = gt[:, :, -1, :].norm(dim=-1)
        w = th.sigmoid((5.0 / self.movement_tau) * (disp - self.movement_tau))
        w = w * batch['track_valid_mask'].any(dim=-1).float()
        if self.obj_weight != 1.0:
            w = w * th.where(batch['owner'] == OBJECT_BODY,
                             th.full_like(w, self.obj_weight),
                             th.ones_like(w))
        return w

    # -- loss ----------------------------------------------------------------
    def compute_loss(self, batch, noise_seed: Optional[int] = None,
                     per_group: bool = False):
        """`L_out = MSE(action) + lambda * weighted MSE(track)`.

        `noise_seed` pins `(t, eps)`. It is not optional bookkeeping here: the
        meta-gradient w.r.t. `q_K` is a single draw of a stochastic loss, and
        the effects being measured are ~1%. Any pre/post or A/B comparison on
        the same frames must pin it or the draw contributes its own difference.
        """
        mem = self.encode_context(batch)
        a_gt = self.normalizer['action'].normalize(batch['action_cmd_seq']) \
            if 'action' in self.normalizer else batch['action_cmd_seq']
        t_gt = self.normalizer['gt_3d_track'].normalize(batch['gt_3d_track']) \
            if 'gt_3d_track' in self.normalizer else batch['gt_3d_track']
        B = a_gt.shape[0]

        if noise_seed is None:
            t = th.randint(0, self.num_train_timesteps, (B,), device=a_gt.device)
            ea = th.randn_like(a_gt)
            et = th.randn_like(t_gt)
        else:
            g = th.Generator(device=a_gt.device)
            g.manual_seed(int(noise_seed))
            t = th.randint(0, self.num_train_timesteps, (B,),
                           device=a_gt.device, generator=g)
            ea = th.randn(a_gt.shape, device=a_gt.device, dtype=a_gt.dtype,
                          generator=g)
            et = th.randn(t_gt.shape, device=t_gt.device, dtype=t_gt.dtype,
                          generator=g)

        # ONE timestep for both modalities, so they sit at the same noise level.
        ab = self.alpha_bar[t]
        noisy_a = ab.sqrt().view(B, 1, 1) * a_gt + (1 - ab).sqrt().view(B, 1, 1) * ea
        noisy_t = ab.sqrt().view(B, 1, 1, 1) * t_gt \
            + (1 - ab).sqrt().view(B, 1, 1, 1) * et

        a_pred, t_pred = self._decode(mem, batch, noisy_a, noisy_t, t)

        a_loss = ((a_pred - a_gt) ** 2).mean()
        w = self.patch_weights(batch)
        t_err = ((t_pred - t_gt) ** 2).mean(dim=(-1, -2))          # [B, N]
        t_loss = (t_err * w).sum() / w.sum().clamp(min=1e-6)
        loss = a_loss + self.track_loss_weight * t_loss
        info = {'loss': float(loss.detach()),
                'action': float(a_loss.detach()),
                'track': float(t_loss.detach())}
        if per_group:
            # 0:3 commanded dpos (the PATH), 6: the 14 joint KP/KD (STIFFNESS).
            e = ((a_pred - a_gt) ** 2).detach()
            info['a_dpos'] = float(e[..., :3].mean())
            info['a_gain'] = float(e[..., 6:].mean())
        return loss, info

    # -- inference -----------------------------------------------------------
    @th.no_grad()
    def sample(self, batch, noise_seed: Optional[int] = None,
               pin_action: bool = False) -> Tuple[th.Tensor, th.Tensor]:
        """DDIM-sample both modalities in lockstep. -> (action, track) RAW.

        `pin_action=True` holds the action channel at the clean ground truth for
        every step, so only the track is sampled. That is the teacher-forced
        readout, and it is OFF-DISTRIBUTION under the shared timestep (see the
        module docstring) -- report it as an approximation, not as a number
        comparable to an action-conditioned track model without that caveat.
        """
        mem = self.encode_context(batch)
        B, N, H = batch['patch_centers'].shape[0], N_PATCH, self.H
        dev = mem.device
        if noise_seed is None:
            a = th.randn(B, H, self.action_dim, device=dev)
            x = th.randn(B, N, H, 3, device=dev)
        else:
            g = th.Generator(device=dev)
            g.manual_seed(int(noise_seed))
            a = th.randn(B, H, self.action_dim, device=dev, generator=g)
            x = th.randn(B, N, H, 3, device=dev, generator=g)
        if pin_action:
            a = self.normalizer['action'].normalize(batch['action_cmd_seq']) \
                if 'action' in self.normalizer else batch['action_cmd_seq']

        steps = th.linspace(self.num_train_timesteps - 1, 0,
                            self.num_inference_steps).long().to(dev)
        for i, t in enumerate(steps):
            a0, x0 = self._decode(mem, batch, a, x, t.expand(B))
            # clip_sample, as GAP.yaml's track scheduler sets. NOT optional:
            # with prediction_type='sample' the eps recovery divides by
            # sqrt(1-alpha_bar), which -> 0 at low noise, so an unclipped
            # outlier is amplified and compounds. Measured on the track-only
            # model without it: 10 DDIM steps gave 74.6 mm and 50 steps gave
            # 129.2 mm -- more steps made it WORSE, the signature of exactly
            # this.
            if self.clip_sample_range > 0:
                a0 = a0.clamp(-self.clip_sample_range, self.clip_sample_range)
                x0 = x0.clamp(-self.clip_sample_range, self.clip_sample_range)
            ab_t = self.alpha_bar[t]
            if i + 1 < len(steps):
                ab_p = self.alpha_bar[steps[i + 1]]
                den = (1 - ab_t).sqrt().clamp(min=1e-8)
                if not pin_action:
                    a = ab_p.sqrt() * a0 + (1 - ab_p).sqrt() * (
                        (a - ab_t.sqrt() * a0) / den)
                x = ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * (
                    (x - ab_t.sqrt() * x0) / den)
            else:
                if not pin_action:
                    a = a0
                x = x0
        if 'action' in self.normalizer:
            a = self.normalizer['action'].unnormalize(a)
        if 'gt_3d_track' in self.normalizer:
            x = self.normalizer['gt_3d_track'].unnormalize(x)
        return a, x
