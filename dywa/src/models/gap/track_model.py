#!/usr/bin/env python3
"""GAP-style 3D point-track prediction (single pinned friction config).

Tracks ONLY. GAP's own policy denoises actions and tracks jointly in one
self-attention stack; here the action branch is dropped entirely, so what is
left is the scene-motion model. The bimanual per-arm tokenization goes with it
-- DyWA has a single 7-D EEF pose.

Two trunks, identical in every other respect so a comparison is about the trunk
and nothing else:

  'transformer' -- deterministic regression. One forward pass, movement-weighted
                   MSE on the track. Cheap, and the honest baseline.
  'dit'         -- GAP's diffusion trunk: the noisy track is embedded per patch,
                   fused with that patch's scene feature, and denoised with the
                   diffusion timestep as its own token. DDIM is hand-rolled --
                   there is no `diffusers` in this container -- with a cosine
                   alpha_bar and prediction_type='sample', matching GAP.yaml.

ACTION CONDITIONING. The encoder sees `[CLS, state, H action tokens, 391 patch
tokens]`, one action token per step of the SAME horizon as the track. This is
not optional: the future motion of everything in the scene depends on where the
arm goes next, so without it the target is underdetermined and the best any
model can do is predict a conditional average (measured: val 2.008 -> 0.038).
The RoPE coordinate of an action token is that step's world xyz with
**t = the step index** -- what GAP's time axis was for, and dead weight without
action tokens.

`action_mode` picks WHAT goes in those tokens, because DyWA offers two very
different things and they are not interchangeable:

  'realized' -- the EEF pose actually reached at each step (7-D). A measured
                outcome: it already reflects IK error, control delay, noise and
                any deflection from contact. Strong signal, but unavailable at
                planning time, when only a command exists.
  'command'  -- the policy's own 20-D action: 3 dpos, 3 dori, 7 joint KP, 7
                joint KD. Causal and available before acting. The 14 gain dims
                have NO counterpart in the realized pose -- stiffness sets the
                contact FORCE, and pusher kinematics alone cannot express it.
  'both'     -- concatenated (27-D).
  'realized_gain' -- realized pose + ONLY the 14 gain dims (21-D). The
                commanded dpos/dori are largely redundant with the realized
                pose (both describe kinematics) and measurably HURT arm tracks
                in 'both'; the gains are the part with no kinematic
                counterpart. This keeps the signal and drops the noise.

Commanded and realized dpos correlate only 0.79/0.92/0.42 per axis, so these are
genuinely different inputs rather than two views of one.

NO LATENT BELIEF. Every episode here shares one pinned friction config, so
there is no hidden dynamics variable for a latent to infer -- it could only
memorise per-episode nuisance (initial pose, goal). It is removed rather than
disabled: the trunks are plain conditional regressors on (dino, patch centres,
EEF pose, action). `common.EncoderLayer` keeps its `film_dim` hook unused,
because the varying-friction experiment will need exactly that path back.
"""
from typing import Optional

import torch as th
import torch.nn as nn

from models.gap.common import (Encoder, EncoderLayer, Normalizer, RoPE4D,
                               sinusoidal_2d, timestep_embedding)

N_PATCH = 391
GRID_H, GRID_W = 17, 23
# nvdr body ids in this scene: 0 table, 1 the pushed object, 7-12 franka links.
OBJECT_BODY = 1


class TrackModel(nn.Module):
    def __init__(self,
                 trunk: str = 'transformer',
                 dino_dim: int = 1024,
                 d_model: int = 512,
                 state_dim: int = 7,
                 action_mode: str = 'realized',
                 action_cmd_dim: int = 20,
                 max_pos: float = 0.06,
                 track_horizon: int = 16,
                 encoder_depth: int = 2,
                 decoder_depth: int = 4,
                 nhead: int = 8,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 use_rope: bool = True,
                 num_train_timesteps: int = 100,
                 num_inference_steps: int = 10,
                 track_temporal_dim: int = 512,
                 clip_sample_range: float = 10.0,
                 obj_weight: float = 1.0,
                 hist_k: int = 0,
                 head: str = 'free',
                 n_body: int = 14,
                 movement_tau: float = 0.01):
        super().__init__()
        assert trunk in ('transformer', 'dit'), trunk
        assert action_mode in ('realized', 'command', 'both',
                               'realized_gain'), action_mode
        self.trunk = trunk
        self.action_mode = action_mode
        self.max_pos = max_pos
        self.H = track_horizon
        self.d_model = d_model
        self.movement_tau = movement_tau
        self.obj_weight = obj_weight
        self.hist_k = hist_k
        assert head in ('free', 'rigid'), head
        self.head = head
        self.n_group = n_body + 1        # +1 bucket for owner == -1
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.clip_sample_range = clip_sample_range

        # ---- observation tokens: [CLS, state, H actions, 391 patches] -------
        self.dino_proj = nn.Linear(dino_dim, d_model)
        self.state_encoder = nn.Linear(state_dim, d_model)
        # Action tokens are fed as a DELTA from the current EEF, not an absolute
        # pose: the same push executed elsewhere on the table should look like
        # the same action. Absolute geometry still reaches the model, through
        # the RoPE coordinate.
        n_gain = action_cmd_dim - 6          # 14: 7 joint KP + 7 joint KD
        act_in = {'realized': state_dim,
                  'command': action_cmd_dim,
                  'both': state_dim + action_cmd_dim,
                  'realized_gain': state_dim + n_gain}[action_mode]
        self.action_encoder = nn.Linear(act_in, d_model)
        self.cls = nn.Parameter(th.randn(1, 1, d_model) * 0.02)
        self.cls_pos = nn.Embedding(1, d_model)
        self.state_pos = nn.Embedding(1, d_model)
        self.action_pos = nn.Embedding(track_horizon, d_model)
        self.n_prefix = 2 + track_horizon        # CLS + state + action tokens

        self.ctx = Encoder(
            lambda: EncoderLayer(d_model, nhead, dim_feedforward, dropout),
            encoder_depth)
        self.dec = Encoder(
            lambda: EncoderLayer(d_model, nhead, dim_feedforward, dropout),
            decoder_depth)

        if hist_k > 0:
            # Same-pixel 3-D displacement between consecutive frames. Patch i is
            # the same PIXEL across frames, so pc[t]-pc[t-1] is scene flow at a
            # fixed pixel -- nonzero exactly where something moved. This is the
            # only route by which velocity reaches a single-frame model.
            self.vel_proj = nn.Linear(3 * hist_k, d_model)
            nn.init.zeros_(self.vel_proj.bias)
            self.state_vel_proj = nn.Linear(7 * hist_k, d_model)
            nn.init.zeros_(self.state_vel_proj.bias)

        self.patch_query = nn.Parameter(th.randn(GRID_H * GRID_W, d_model) * 0.02)
        if head == 'free':
            self.track_head = nn.Linear(d_model, track_horizon * 3)
        else:
            # Every patch of a rigid body moves by ONE shared transform --
            # compute_tracks generates the labels that way (anchor -> body local
            # frame -> replay through recorded body poses). So a body's whole
            # 391-patch track field has exactly 6 DoF per step. The free head
            # predicts 391*H*3 unconstrained numbers and cannot know that; this
            # one pools each body's patches, predicts a single SE(3) trajectory,
            # and expands it back, so all ~7 object patches vote on one estimate.
            #
            # NOTE for the 'dit' trunk: predict() clamps x0 to
            # +-clip_sample_range ELEMENTWISE, which is not a rigid operation.
            # It only activates beyond ~0.35 m of displacement (no real track
            # reaches that), but if you ever see non-rigid DiT output, this is
            # why. The transformer trunk is rigid exactly.
            self.track_head = nn.Linear(d_model, track_horizon * 6)
            # Zero-init so the initial motion is the IDENTITY: rotvec 0 -> R=I,
            # translation 0 -> displacement exactly 0. Without this a random
            # head emits metre-scale rigid motions at step 0 (measured loss 6.4
            # vs 0.37 for the free head) and training starts from far away.
            nn.init.zeros_(self.track_head.weight)
            nn.init.zeros_(self.track_head.bias)

        if trunk == 'dit':
            # Flatten the H steps into one vector per patch, as GAP does: no
            # per-step weight sharing and no time embedding. Only valid because
            # H is fixed for every sample.
            self.track_embed = nn.Sequential(
                nn.Linear(track_horizon * 3, track_temporal_dim), nn.GELU(),
                nn.Linear(track_temporal_dim, d_model))
            self.fuse = nn.Linear(2 * d_model, d_model)
            self.t_embed = nn.Sequential(nn.Linear(256, d_model), nn.SiLU(),
                                         nn.Linear(d_model, d_model))
            self.register_buffer('alpha_bar',
                                 self._cosine_alpha_bar(num_train_timesteps),
                                 persistent=False)

        self.rope = RoPE4D(d_model // nhead) if use_rope else None
        self.normalizer = Normalizer()

    # -- diffusion schedule (squaredcos_cap_v2, as GAP.yaml) ----------------
    @staticmethod
    def _cosine_alpha_bar(T: int) -> th.Tensor:
        s = 0.008
        t = th.arange(T + 1, dtype=th.float64) / T
        f = th.cos((t + s) / (1 + s) * 3.141592653589793 / 2) ** 2
        ab = (f / f[0]).clamp(1e-8, 1.0)
        return ab[1:].float()

    # -- context ------------------------------------------------------------
    def encode_context(self, batch):
        dino = batch['dino_features']
        B = dino.shape[0]
        if dino.dim() == 4:
            dino = dino.reshape(B, -1, dino.shape[-1])
        patches = self.dino_proj(dino)
        state = self.state_encoder(batch['agent_pos']).unsqueeze(1)
        if self.hist_k > 0:
            pc = batch['patch_centers']
            ph = batch['patch_centers_hist']            # [B, K, N, 3], newest 1st
            prev = th.cat([pc.unsqueeze(1), ph[:, :-1]], dim=1)
            dv = (prev - ph).permute(0, 2, 1, 3).reshape(B, pc.shape[1], -1)
            patches = patches + self.vel_proj(dv)
            ap = batch['agent_pos']
            ah = batch['agent_pos_hist']                # [B, K, 7]
            aprev = th.cat([ap.unsqueeze(1), ah[:, :-1]], dim=1)
            state = state + self.state_vel_proj(
                (aprev - ah).reshape(B, -1)).unsqueeze(1)

        actions = self.action_encoder(self.action_features(batch))  # [B, H, D]
        x = th.cat([self.cls.expand(B, -1, -1), state, actions, patches], dim=1)

        pos = th.cat([
            self.cls_pos.weight.unsqueeze(0).expand(B, -1, -1),
            self.state_pos.weight.unsqueeze(0).expand(B, -1, -1),
            self.action_pos.weight.unsqueeze(0).expand(B, -1, -1),
            sinusoidal_2d(GRID_H, GRID_W, self.d_model,
                          x.device).unsqueeze(0).expand(B, -1, -1)], dim=1)

        rope_args = self._rope_obs(batch, B, x.shape[1])
        return self.ctx(x, pos=pos, **rope_args)

    def action_features(self, batch):
        """[B, H, act_in] -- what the action tokens carry.

        The realized pose goes in as a DELTA from the current EEF (the same push
        elsewhere on the table should look like the same action); absolute
        geometry still reaches the model through the RoPE coordinate. The
        command is already normalised to ~[-1, 1] by the env, so it is clipped
        and passed through -- clipped because the policy emits an unsquashed
        Gaussian mean and the simulator clamps it before use, so anything
        outside the box never physically happened.
        """
        feats = []
        if self.action_mode in ('realized', 'both', 'realized_gain'):
            act = batch['action_seq']                   # [B, H, 7] absolute
            feats.append(th.cat([act[..., :3] - batch['agent_pos'][:, None, :3],
                                 act[..., 3:]], dim=-1))
        if self.action_mode in ('command', 'both'):
            feats.append(batch['action_cmd_seq'].clamp(-1.0, 1.0))
        elif self.action_mode == 'realized_gain':
            # dims 6: are the 7 joint KP + 7 joint KD; 0:6 (dpos, dori) are
            # deliberately excluded.
            feats.append(batch['action_cmd_seq'][..., 6:].clamp(-1.0, 1.0))
        return feats[0] if len(feats) == 1 else th.cat(feats, dim=-1)

    def action_xyz(self, batch):
        """[B, H, 3] world xyz for each action token's RoPE coordinate.

        In 'command' mode there is no realized path to read, so the INTENDED one
        is integrated from the commanded per-step deltas -- exactly what a
        planner would have. Using the realized path here instead would leak the
        very signal this arm is meant to do without.
        """
        if self.action_mode == 'command':
            d = batch['action_cmd_seq'][..., :3].clamp(-1.0, 1.0) * self.max_pos
            return batch['agent_pos'][:, None, :3] + th.cumsum(d, dim=1)
        return batch['action_seq'][..., :3]

    def _rope_obs(self, batch, B, n_tok):
        """4-D RoPE over the observation sequence.

        state token  -> current EEF xyz,      t = 0
        action token -> that step's EEF xyz,  t = 1..H   <- the time axis, live
        patch token  -> its own patch centre, t = 0
        CLS has no natural coordinate and is left unrotated.
        """
        if self.rope is None:
            return dict(rope=None, cos_sin=None, rope_mask=None)
        pc = batch['patch_centers']
        H = self.H
        coords = pc.new_zeros(B, n_tok, 4)
        mask = th.zeros(B, n_tok, dtype=th.bool, device=pc.device)
        coords[:, 1:2, :3] = batch['agent_pos'][:, None, :3]
        coords[:, 2:2 + H, :3] = self.action_xyz(batch)
        coords[:, 2:2 + H, 3] = th.arange(1, H + 1, device=pc.device,
                                          dtype=coords.dtype)
        coords[:, 2 + H:, :3] = pc
        mask[:, 1:] = True
        return dict(rope=self.rope, cos_sin=self.rope.build_cos_sin(coords),
                    rope_mask=mask)

    def _rope_patches(self, batch, B, n_tok, n_extra=0):
        """RoPE over patch tokens (+ optional trailing tokens with no
        coordinate, e.g. the diffusion timestep)."""
        if self.rope is None:
            return dict(rope=None, cos_sin=None, rope_mask=None)
        pc = batch['patch_centers']
        n_p = n_tok - n_extra
        coords = pc.new_zeros(B, n_tok, 4)
        coords[:, :n_p, :3] = pc
        mask = th.zeros(B, n_tok, dtype=th.bool, device=pc.device)
        mask[:, :n_p] = True
        return dict(rope=self.rope, cos_sin=self.rope.build_cos_sin(coords),
                    rope_mask=mask)

    # -- trunks -------------------------------------------------------------
    def _decode_transformer(self, mem, batch):
        B = mem.shape[0]
        x = mem[:, self.n_prefix:, :] + self.patch_query.unsqueeze(0)
        x = self.dec(x, **self._rope_patches(batch, B, x.shape[1]))
        return self._emit(x, batch)

    def _decode_dit(self, mem, batch, noisy, t):
        B, N = noisy.shape[0], noisy.shape[1]
        tok = self.track_embed(noisy.reshape(B, N, -1))
        # The per-patch concat IS the conditioning on the scene -- this is why
        # GAP needs no cross-attention to a separate memory.
        fused = self.fuse(th.cat([mem[:, self.n_prefix:, :], tok], dim=-1))
        fused = fused + self.patch_query.unsqueeze(0)
        te = self.t_embed(timestep_embedding(t, 256)).unsqueeze(1)
        seq = th.cat([fused, te], dim=1)
        out = self.dec(seq, **self._rope_patches(batch, B, N + 1, n_extra=1))
        return self._emit(out[:, :N, :], batch)

    # -- rigid expansion ------------------------------------------------------
    @staticmethod
    def _rotate(rv: th.Tensor, v: th.Tensor) -> th.Tensor:
        """Rotate v by the axis-angle rv, both [..., 3] -> [..., 3].

        Rodrigues in VECTOR form:
            v cos(t) + (k x v) sin(t) + k (k . v) (1 - cos(t))
        deliberately, rather than building R and doing R @ v. Any matmul here
        runs in TF32 on Ampere, whose ~1e-3 relative error breaks rigidity by up
        to 4 cm -- the same order as the error this head exists to reduce.
        Measured: matmul form 4.3e-2 m residual, 4.9e-4 with TF32 off, and this
        form needs no such global switch.
        """
        t = rv.norm(dim=-1, keepdim=True)
        k = rv / t.clamp(min=1e-8)
        c, s_ = t.cos(), t.sin()
        return (v * c + th.cross(k.expand_as(v), v, dim=-1) * s_
                + k * (k * v).sum(-1, keepdim=True) * (1.0 - c))

    def _rigid_tracks(self, x, batch):
        """Per-patch features -> one SE(3) trajectory per body -> per-patch track.

        Grouping uses `owner` (the nvdr body label). Segmentation is already a
        stated assumption of this benchmark -- DyWA's own student observes an
        object-segmented `partial_cloud` -- so this is not privileged state.
        owner == -1 (invalid patch) gets its own bucket; those patches are
        unsupervised anyway.
        """
        B, N, _ = x.shape
        G = self.n_group
        g = (batch['owner'] + 1).clamp(0, G - 1)                # [B, N]
        onehot = th.zeros(B, G, N, device=x.device, dtype=x.dtype)
        onehot.scatter_(1, g.unsqueeze(1), 1.0)
        cnt = onehot.sum(-1, keepdim=True).clamp(min=1.0)       # [B, G, 1]

        pooled = th.bmm(onehot, x) / cnt                        # [B, G, D]
        anchor = batch['patch_centers']                         # [B, N, 3]
        centre = th.bmm(onehot, anchor) / cnt                   # [B, G, 3]

        # The track is stored NORMALIZED (isotropic scale, zero offset), but a
        # rigid motion is only rigid in METRES: scaling a displacement field
        # while leaving the anchors alone is not a rigid transform. So build the
        # motion in metres, then normalise the result -- predict() undoes that
        # exactly and rigidity survives to world units.
        rms = 1.0
        if 'gt_3d_track' in self.normalizer:
            rms = float(1.0 / self.normalizer['gt_3d_track'].scale[0])
        se3 = self.track_head(pooled).reshape(B, G, self.H, 6)
        idx = g.unsqueeze(-1).unsqueeze(-1).expand(B, N, self.H, 6)
        se3 = th.gather(se3, 1, idx)                            # [B, N, H, 6]
        c = th.gather(centre, 1, g.unsqueeze(-1).expand(B, N, 3))

        rel = (anchor - c).unsqueeze(2).expand(-1, -1, self.H, -1)
        rot = self._rotate(se3[..., :3], rel)                   # [B, N, H, 3]
        trans = se3[..., 3:] * rms          # head emits ~unit scale -> metres
        # displacement from the anchor, in metres, matching the label convention
        raw = rot + c.unsqueeze(2) + trans - anchor.unsqueeze(2)
        return raw / rms if rms != 1.0 else raw

    def _emit(self, x, batch):
        B = x.shape[0]
        if self.head == 'free':
            return self.track_head(x).reshape(B, -1, self.H, 3)
        return self._rigid_tracks(x, batch)

    # -- weighting ----------------------------------------------------------
    def patch_weights(self, batch) -> th.Tensor:
        """[B, N] -- movement weighting times validity.

        Movement: static background patches carry no information about the push
        and would otherwise dominate by sheer count. Computed from the RAW
        metre-scale track so the 1 cm threshold means what it says regardless
        of normalization.

        Validity: an invalid patch's stored track is zero by CONVENTION, not
        measurement, so supervising it teaches the model to emit zeros for
        whatever the background happens to look like.

        `obj_weight` then multiplies the OBJECT's patches. It exists because the
        two together are badly unbalanced against the thing we care about:
        measured, moving patches are ~90% franka links and ~10% object, and both
        saturate the movement sigmoid, so ~90% of the gradient goes to arm
        kinematics -- which the model already predicts at 8% relative error
        because the realized EEF path is an input. The object sits at 50%.
        obj_weight ~9 equalises the two shares.
        """
        gt_raw = batch['gt_3d_track']
        disp = gt_raw[:, :, -1, :].norm(dim=-1)
        w = th.sigmoid((5.0 / self.movement_tau) * (disp - self.movement_tau))
        w = w * batch['track_valid_mask'].any(dim=-1).float()
        if self.obj_weight != 1.0:
            w = w * th.where(batch['owner'] == OBJECT_BODY,
                             th.full_like(w, self.obj_weight),
                             th.ones_like(w))
        return w

    # -- loss ---------------------------------------------------------------
    def compute_loss(self, batch, noise_seed: Optional[int] = None):
        """`noise_seed` pins the DiT's timestep and noise draw.

        Needed whenever the SAME batch is evaluated repeatedly and the
        differences between passes are the measurement (ablations, A/B of two
        checkpoints on identical frames). Left free, each pass draws a fresh
        (t, noise) and that variance can swamp the effect being measured.
        Ignored by the transformer trunk, which is deterministic already.
        """
        mem = self.encode_context(batch)
        gt = self.normalizer['gt_3d_track'].normalize(batch['gt_3d_track']) \
            if 'gt_3d_track' in self.normalizer else batch['gt_3d_track']

        if self.trunk == 'transformer':
            pred = self._decode_transformer(mem, batch)
        else:
            B = gt.shape[0]
            if noise_seed is None:
                t = th.randint(0, self.num_train_timesteps, (B,),
                               device=gt.device)
                noise = th.randn_like(gt)
            else:
                g = th.Generator(device=gt.device)
                g.manual_seed(int(noise_seed))
                t = th.randint(0, self.num_train_timesteps, (B,),
                               device=gt.device, generator=g)
                noise = th.randn(gt.shape, device=gt.device, dtype=gt.dtype,
                                 generator=g)
            ab = self.alpha_bar[t].view(B, 1, 1, 1)
            noisy = ab.sqrt() * gt + (1 - ab).sqrt() * noise
            pred = self._decode_dit(mem, batch, noisy, t)
        target = gt                            # prediction_type='sample'

        w = self.patch_weights(batch)
        err = ((pred - target) ** 2).mean(dim=(-1, -2))       # [B, N]
        loss = (err * w).sum() / w.sum().clamp(min=1e-6)
        return loss, {'loss': float(loss.detach()),
                      'move_w': float(w.mean().detach())}

    # -- inference ----------------------------------------------------------
    @th.no_grad()
    def predict(self, batch, noise_seed: Optional[int] = None):
        """-> track in RAW metres, [B, N, H, 3].

        `noise_seed` pins the DiT's starting noise. Set it when comparing runs
        over the same frames, or the DDIM draw contributes its own difference on
        top of whatever is being measured.
        """
        mem = self.encode_context(batch)
        if self.trunk == 'transformer':
            out = self._decode_transformer(mem, batch)
        else:
            B, N = batch['patch_centers'].shape[:2]
            if noise_seed is None:
                x = th.randn(B, N, self.H, 3, device=mem.device)
            else:
                g = th.Generator(device=mem.device)
                g.manual_seed(int(noise_seed))
                x = th.randn(B, N, self.H, 3, device=mem.device, generator=g)
            steps = th.linspace(self.num_train_timesteps - 1, 0,
                                self.num_inference_steps).long().to(mem.device)
            for i, t in enumerate(steps):
                x0 = self._decode_dit(mem, batch, x, t.expand(B))
                # clip_sample, as GAP.yaml's track_noise_scheduler sets
                # (range 10.0). Not optional: with prediction_type='sample' the
                # eps recovery below divides by sqrt(1-alpha_bar), which -> 0 at
                # low noise, so an unclipped x0 outlier is amplified and
                # compounds. Measured without it: 10 DDIM steps gave 74.6 mm and
                # 50 steps gave 129.2 mm -- more steps made it WORSE, which is
                # the signature of exactly this.
                if self.clip_sample_range > 0:
                    x0 = x0.clamp(-self.clip_sample_range,
                                  self.clip_sample_range)
                ab_t = self.alpha_bar[t]
                if i + 1 < len(steps):
                    ab_p = self.alpha_bar[steps[i + 1]]
                    eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt().clamp(min=1e-8)
                    x = ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps
                else:
                    x = x0
            out = x
        if 'gt_3d_track' in self.normalizer:
            out = self.normalizer['gt_3d_track'].unnormalize(out)
        return out
