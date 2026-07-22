#!/usr/bin/env python3
"""
Test-Time-Training (TTT) with a learned dynamics model, ported into DyWA.

This is a faithful port of the pick-bar TTT modules
(`ttt_with_dyn_bars.py`) adapted to the DyWA distillation setting:

  * A per-env latent belief `q` conditions both a policy (the DyWA student
    trunk, elsewhere) and a `DynamicsModel`.
  * `q` is gradient-adapted at inference by minimizing the prediction error of
    the `DynamicsModel` against `OutcomeEncoder(next_object_cloud)` over the
    interaction history (the TTT inner loop).

Design invariants (see plan):
  1. Attention is manual matmul/softmax so double-backward works under
     `create_graph=True` in the inner loop (fused SDPA has no double backward).
     These modules DO NOT reuse DyWA's PointEncoder/TokenEncoder — that keeps
     the MAML inner graph off PointNet2.
  2. `OutcomeEncoder` ends in `LayerNorm(latent_dim)` (anti-collapse guard).
  3. The clouds handed to `DynamicsModel`/`OutcomeEncoder` must be normalized in
     a *shared* frame (see `frame_of`/`to_frame`) so object motion survives —
     DyWA's per-cloud normalization would erase it.

The inner update is generalized to arbitrary leading "task" dims so the same
code serves offline meta-training (single episode: `q` is `[D]`) and
online/eval (`q` is `[num_env, D]`), preserving the bars per-task step scaling.
"""

from typing import Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from util.math_util import (compose_pose_tq, invert_pose_tq, apply_pose_tq,
                            rot6d_to_matrix)


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (match ttt_with_dyn_bars.py). Overridable via constructor args.
# ─────────────────────────────────────────────────────────────────────────────
LATENT_DIM = 64
TTT_ALPHA = 0.5
D_MODEL = 128
NHEAD = 4
GMM_LAYERS = 4          # kept for reference; policy trunk is DyWA's, not here
DYN_LAYERS = 3
OUTCOME_LAYERS = 2
# Off by default: the dyn/outcome nets are tiny, so gradient checkpointing buys
# almost no memory, and checkpointed double-backward (offline second-order) is
# exactly what trips the CUDACachingAllocator assert under expandable_segments.
USE_CHECKPOINT = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention blocks — manual matmul/softmax so double backward works.
# ─────────────────────────────────────────────────────────────────────────────
class FiLMTransformerBlock(nn.Module):
    """Pre-norm self-attention + FFN, with FiLM (on `cond`) applied to the FFN.

    Manual attention (not nn.MultiheadAttention) because fused SDPA kernels on
    CUDA do not implement double backward, which we need under create_graph.
    """

    def __init__(self, d_model, nhead, cond_dim, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # FiLM near-identity init: small gamma, zero beta. Non-zero so that
        # q-conditioning is live from step 0.
        self.film = nn.Linear(cond_dim, 2 * d_model)
        nn.init.xavier_normal_(self.film.weight, gain=0.5)
        nn.init.zeros_(self.film.bias)

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = th.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = th.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_dropout(self.attn_out(out))

    def _forward_impl(self, x, cond):
        h = self.ln1(x)
        x = x + self._attn(h)

        h = self.ln2(x)
        h = self.ffn(h)

        gb = self.film(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        x = x + h
        return x

    def forward(self, x, cond):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, cond, use_reentrant=False)
        return self._forward_impl(x, cond)


class SelfAttnBlock(nn.Module):
    """Manual pre-norm self-attention + FFN, no conditioning (double-backward safe)."""

    def __init__(self, d_model, nhead, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = th.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = th.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_dropout(self.attn_out(out))

    def _forward_impl(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. OutcomeEncoder — encodes the object's rigid MOTION (pcd, flow) → outcome latent.
#    Ported faithfully from ttt_with_dyn_doors.py: `flow` is the content channel
#    (V), `pcd` is positional (Q/K). Static points (zero flow) produce a constant
#    V, so a no-motion window yields a constant latent — the structural anti-collapse
#    guard that a cloud-only encoder lacks.
# ─────────────────────────────────────────────────────────────────────────────
class FlowContentAttnBlock(nn.Module):
    """Self-attention where Q/K see (flow+position) but V sees flow only.

    Manual matmul/softmax for double-backward safety under create_graph=True.
    """

    def __init__(self, d_model, nhead, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.scale = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        # Separate norms for the position-aware (Q/K) and content (V) streams.
        self.ln1_qk = nn.LayerNorm(d_model)
        self.ln1_v = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def _attn(self, x_qk, x_v):
        B, N, D = x_qk.shape
        q = self.q_proj(x_qk).view(B, N, self.nhead, self.d_head).transpose(1, 2)
        k = self.k_proj(x_qk).view(B, N, self.nhead, self.d_head).transpose(1, 2)
        v = self.v_proj(x_v).view(B, N, self.nhead, self.d_head).transpose(1, 2)
        scores = th.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out = th.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_dropout(self.attn_out(out))

    def _forward_impl(self, x_qk, x_v):
        # Residual stream tracks the content (V) path; the position stream only
        # routes attention. Pooled output is thus a function of V's (flow).
        h = self._attn(self.ln1_qk(x_qk), self.ln1_v(x_v))
        x = x_v + h
        x = x + self.ffn(self.ln2(x))
        return x

    def forward(self, x_qk, x_v):
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_impl, x_qk, x_v, use_reentrant=False)
        return self._forward_impl(x_qk, x_v)


class OutcomeEncoder(nn.Module):
    """`(pcd, flow) [B, N, 3]` (shared frame) → outcome latent `[B, latent_dim]`.

    `flow` is the per-point rigid displacement of the object over the K-window;
    `pcd` is the before-cloud. Called inside the inner loop with create_graph=True
    → double-backward safe. Trailing LayerNorm + V-from-flow guard prevent collapse.
    """

    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=OUTCOME_LAYERS,
                 latent_dim=LATENT_DIM, use_checkpoint=USE_CHECKPOINT):
        super().__init__()
        self.flow_embed = nn.Linear(3, d_model)     # content
        self.pcd_embed = nn.Linear(3, d_model)      # position
        self.first_block = FlowContentAttnBlock(
            d_model=d_model, nhead=nhead, use_checkpoint=use_checkpoint)
        self.rest_blocks = nn.ModuleList([
            SelfAttnBlock(d_model=d_model, nhead=nhead, use_checkpoint=use_checkpoint)
            for _ in range(max(0, num_layers - 1))
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, pcd, flow):
        pcd_after = pcd + flow
        flow_feat = self.flow_embed(flow)           # content → V
        pcd_feat = self.pcd_embed(pcd_after)        # position → Q/K
        x = self.first_block(flow_feat + pcd_feat, flow_feat)
        for blk in self.rest_blocks:
            x = blk(x)
        x = self.final_ln(x)
        return self.proj(x.mean(dim=1))


class FlowDecoder(nn.Module):
    """Per-point MLP `(pcd_point [3], z [latent_dim]) → flow_point [3]`.

    Outer-loop-only anti-collapse: forces `z` to be reconstructively complete
    w.r.t. the observed flow. A plain MLP (double-backward-safe, though only used
    in the single-backward outer pass today).
    """

    def __init__(self, latent_dim=LATENT_DIM, hidden=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 + latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, pcd, z):
        B, N, _ = pcd.shape
        z_b = z.unsqueeze(1).expand(B, N, -1)
        return self.net(th.cat([pcd, z_b], dim=-1))


# ─────────────────────────────────────────────────────────────────────────────
# 2b. TransformerPointEncoder — self-attention point encoder for the POLICY.
# ─────────────────────────────────────────────────────────────────────────────
class TransformerPointEncoder(nn.Module):
    """Self-attention transformer point encoder → `num_tokens` tokens.

    Drop-in replacement for the DyWA PointNet2 `PointEncoder` in the policy:
    same call `enc(cloud[B,N,3]) -> [B, num_tokens, embed_size]`.

    Uses the manual-attention `SelfAttnBlock` (same kernel as
    `FiLMTransformerBlock` but WITHOUT FiLM) — the latent belief `q` is injected
    downstream at the TokenDecoder, not here. `num_tokens` learnable *summary*
    tokens are prepended and self-attend over the point embeddings; their final
    states are the output tokens (DETR/Perceiver-style pooling to a fixed count).

    Per-cloud normalized (like PointNet2's pc_normalize); a `[center ‖ scale]`
    global embedding is added to the summary tokens so absolute location — which
    per-cloud normalization removes — is preserved.
    """

    def __init__(self, num_tokens, embed_size, num_layers=3, nhead=4,
                 use_checkpoint=False, eps=1e-6):
        super().__init__()
        self.num_tokens = num_tokens
        self.eps = eps
        self.embed = nn.Linear(3, embed_size)
        self.global_embed = nn.Linear(4, embed_size)          # [center(3) ‖ scale(1)]
        self.summary = nn.Parameter(th.randn(num_tokens, embed_size) * 0.02)
        self.blocks = nn.ModuleList([
            SelfAttnBlock(d_model=embed_size, nhead=nhead, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_size)

    def forward(self, cloud):
        B, N, _ = cloud.shape
        center = cloud.mean(dim=1, keepdim=True)                          # [B,1,3]
        scale = (cloud - center).norm(dim=-1).amax(dim=1, keepdim=True).clamp_min(self.eps)  # [B,1]
        cloud_n = (cloud - center) / scale.unsqueeze(-1)                  # [B,N,3]
        x = self.embed(cloud_n)                                           # [B,N,D]
        g = self.global_embed(th.cat([center.squeeze(1), scale], dim=-1))  # [B,D]
        summ = self.summary.unsqueeze(0).expand(B, -1, -1) + g.unsqueeze(1)  # [B,T,D]
        h = th.cat([summ, x], dim=1)                                      # [B,T+N,D]
        for blk in self.blocks:
            h = blk(h)
        return self.final_ln(h[:, :self.num_tokens])                     # [B,T,D]


# ─────────────────────────────────────────────────────────────────────────────
# 2b'. FiLMPolicyTrunk — doors-style belief-conditioned policy head.
#
#      Replaces DyWA's `TokenDecoder(film)` + `Aggregator` for the TTT student.
#      Motivation (measured, see plan REVISION 3): DyWA's `FilmBlock` computes
#      `out = scale*out + bias` with `scale` centered at 0 (not `1+gamma`), pushes
#      it through an `MLP_layer(norm='ln')` ending in LayerNorm, then adds a
#      q-INDEPENDENT `residual_conv(x)` highway. Net effect: ||dmu|| saturates at
#      ~1e-4 however far `q` moves, and ||dL/dq|| ~ 2.6e-5 vs ~31 for the output
#      projection. Since q0/dyn/outcome train ONLY through dL/dq, that froze them
#      all at init.
#
#      `FiLMTransformerBlock` (above) has none of those three problems: FiLM is
#      near-identity `(1+gamma)*h + beta` applied inside a pre-norm residual, with
#      no post-FiLM LayerNorm and no competing q-independent path. This trunk is
#      doors' `GMMHead` conditioning structure — FiLM blocks + a direct concat-q
#      head + a zero-init q-only residual — retargeted from per-point features to
#      DyWA's token sequence and to a single Gaussian instead of a GMM.
# ─────────────────────────────────────────────────────────────────────────────
class FiLMPolicyTrunk(nn.Module):
    """`(tokens [B,T,D], q [B,L])` → action distribution `[B, 2, action_size]`.

    `[...,0,:]` = mu, `[...,1,:]` = log_std (matching `belief_forward`'s contract).

    `q` reaches the output through three independent routes, mirroring doors:
      1. FiLM conditioning inside every block (shapes the representation),
      2. concatenation into the readout head (direct, live from step 0),
      3. a zero-init q-only residual on the logits (free to grow, harmless at init).

    Outer-loop only — the policy is never called inside the TTT inner loop — so
    `use_checkpoint` is left off and double-backward safety is not required here
    (the manual attention in `FiLMTransformerBlock` provides it regardless).
    """

    def __init__(self, embed_size: int, action_size: int, latent_dim: int,
                 num_layers: int = 4, nhead: int = 4, head_hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.action_size = action_size
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(d_model=embed_size, nhead=nhead,
                                 cond_dim=latent_dim, dropout=dropout,
                                 use_checkpoint=False)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(embed_size)

        # Direct concat-q readout (doors: `feats = cat([x, q_broadcast])`).
        self.head = nn.Sequential(
            nn.Linear(embed_size + latent_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 2 * action_size),
        )

        # q-only residual, zero-init (doors: `q_residual_weight`/`q_residual_mean`).
        self.q_res = nn.Linear(latent_dim, 2 * action_size)
        nn.init.zeros_(self.q_res.weight)
        nn.init.zeros_(self.q_res.bias)

    def forward(self, tokens: th.Tensor, q: th.Tensor) -> th.Tensor:
        x = tokens
        for blk in self.blocks:
            x = blk(x, q)
        x = self.final_ln(x)
        pooled = x.mean(dim=1)                              # [B, embed_size]
        out = self.head(th.cat([pooled, q], dim=-1))        # [B, 2*action_size]
        out = out + self.q_res(q)
        return out.reshape(*out.shape[:-1], 2, self.action_size)


# ─────────────────────────────────────────────────────────────────────────────
# 2c. MacroActionEncoder — summarizes a K-step action sequence → macro embedding.
#     Only built for `macro_action='seq'`; double-backward safe (manual attention),
#     because it is applied inside the outer meta-graph (its output feeds `dyn`
#     whose inner-loss gradient w.r.t. q is differentiated in the outer backward).
# ─────────────────────────────────────────────────────────────────────────────
class MacroActionEncoder(nn.Module):
    """`action_seq [B, K, action_raw_dim]` → macro embedding `[B, out_dim]`."""

    def __init__(self, action_raw_dim, out_dim, k_window, d_model=D_MODEL,
                 nhead=NHEAD, num_layers=2, use_checkpoint=False):
        super().__init__()
        self.embed = nn.Linear(action_raw_dim, d_model)
        self.pos = nn.Parameter(th.randn(k_window, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            SelfAttnBlock(d_model=d_model, nhead=nhead, use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, action_seq):
        B, K, _ = action_seq.shape
        x = self.embed(action_seq) + self.pos[:K].unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        return self.proj(x.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 3. DynamicsModel — per-point FiLM transformer, single latent head.
# ─────────────────────────────────────────────────────────────────────────────
class DynamicsModel(nn.Module):
    """`(q, macro_action, before_cloud)` → predicted outcome latent `[B, latent_dim]`.

    FiLM conditions on the belief `q` ALONE (as in `ttt_with_dyn_doors.py`): `q` is
    what the inner loop adapts, so the global modulation carries the belief and
    nothing else. The macro-action enters through the **per-point features**
    instead — it is a *global* action descriptor (`Δpos ‖ rotvec`, + gains/seq per
    mode), so it is broadcast to every point and concatenated to `cloud`.

    There is deliberately no `rel_pos = action_pos − cloud`: the macro's position
    part is a delta EE *displacement*, not an absolute contact point, so that
    subtraction (displacement − position) would be geometrically meaningless.
    (If a genuine contact anchor is wanted later, feed the EE *position* at the
    window start and use `ee_pos − cloud` — that needs a new input, so it's a
    follow-up, not done here.)

    `action_dim` is the macro-action width (mode-dependent: `ee_only`=6,
    `ee_mean_gains`=20, `seq`=macro_embed_dim).
    """

    def __init__(self, latent_dim=LATENT_DIM, action_dim=20, pos_dim=3,
                 d_model=D_MODEL, num_layers=DYN_LAYERS, nhead=NHEAD,
                 use_checkpoint=USE_CHECKPOINT):
        super().__init__()
        self.action_dim = action_dim
        self.pos_dim = pos_dim
        self.latent_dim = latent_dim

        # per-point input: [cloud(3) ‖ macro-action broadcast(action_dim)]
        self.embed = nn.Linear(3 + action_dim, d_model)
        # FiLM conditions on q alone (see class docstring).
        self.cond_dim = latent_dim
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(d_model=d_model, nhead=nhead, cond_dim=self.cond_dim,
                                 use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.latent_head = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, q, action, cloud):
        B, N, _ = cloud.shape
        act_bc = action.unsqueeze(1).expand(B, N, self.action_dim)   # [B, N, A]
        per_point = th.cat([cloud, act_bc], dim=-1)                  # [B, N, 3+A]

        x = self.embed(per_point)
        for blk in self.blocks:
            x = blk(x, q)                                # FiLM on q alone
        x = self.final_ln(x)
        return self.latent_head(x.mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Shared-frame normalization (invariant #3).
# ─────────────────────────────────────────────────────────────────────────────
def frame_of(cloud: th.Tensor, eps: float = 1e-6) -> Tuple[th.Tensor, th.Tensor]:
    """Compute a normalization frame (center, scale) from a reference cloud.

    cloud: `[*L, N, 3]` (`*L` = leading task/env dims).
    Returns center `[*L, 1, 3]`, scale `[*L, 1, 1]`.
    """
    center = cloud.mean(dim=-2, keepdim=True)                       # [*L, 1, 3]
    dist = (cloud - center).norm(dim=-1)                            # [*L, N]
    scale = dist.max(dim=-1, keepdim=True).values.clamp_min(eps)    # [*L, 1]
    return center, scale.unsqueeze(-1)                             # [*L,1,3],[*L,1,1]


def _align_cloud(frame: th.Tensor, ndim: int) -> th.Tensor:
    # Insert singleton dims *before* the trailing `[1, 3]` pair so the frame
    # broadcasts over any middle dims (e.g. a history axis H) of a cloud.
    while frame.dim() < ndim:
        frame = frame.unsqueeze(-3)
    return frame


def _align_pos(frame: th.Tensor, ndim: int) -> th.Tensor:
    # Insert singleton dims *before* the trailing feature dim, for point tensors.
    while frame.dim() < ndim:
        frame = frame.unsqueeze(-2)
    return frame


def to_frame(cloud: th.Tensor, center: th.Tensor, scale: th.Tensor) -> th.Tensor:
    """Apply a shared frame to a cloud `[*L, ..., N, 3]` (any middle dims OK)."""
    c = _align_cloud(center, cloud.dim())
    s = _align_cloud(scale, cloud.dim())
    return (cloud - c) / s


def to_frame_pos(pos: th.Tensor, center: th.Tensor, scale: th.Tensor) -> th.Tensor:
    """Apply a shared frame to a position `[*L, ..., 3]` (e.g. an action pos)."""
    c = _align_pos(center.squeeze(-2), pos.dim())   # [*L,3] → broadcastable
    s = _align_pos(scale.squeeze(-2), pos.dim())    # [*L,1] → broadcastable
    return (pos - c) / s


def to_frame_vec(vec: th.Tensor, scale: th.Tensor) -> th.Tensor:
    """Apply a shared frame to a displacement vector `[*L, ..., 3]` (flow / net EE
    Δpos) — **scale only**: a displacement is translation-invariant, so unlike a
    position it must NOT be re-centered. The frame is translate+isotropic-scale
    (no rotation), so vector directions are preserved."""
    s = _align_pos(scale.squeeze(-2), vec.dim())
    return vec / s


def matrix_to_rotvec(R: th.Tensor) -> th.Tensor:
    """Rotation matrix `[*,3,3]` → axis-angle (rotation vector) `[*,3]`."""
    cos = (((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5).clamp(-1.0, 1.0)
    ang = th.acos(cos)                                            # [*]
    ax = th.stack([R[..., 2, 1] - R[..., 1, 2],
                   R[..., 0, 2] - R[..., 2, 0],
                   R[..., 1, 0] - R[..., 0, 1]], dim=-1)          # = 2·sin(ang)·axis
    two_sin = (2.0 * th.sin(ang)).unsqueeze(-1)
    rotvec = ax / two_sin.clamp_min(1e-8) * ang.unsqueeze(-1)
    # small-angle limit: ax ≈ 2·ang·axis  ⇒  rotvec ≈ ax/2
    small = (ang.abs() < 1e-4).unsqueeze(-1)
    return th.where(small, 0.5 * ax, rotvec)


def ee_pose_delta(hand_s: th.Tensor, hand_sK: th.Tensor) -> th.Tensor:
    """Net EE motion over a window → `[*, 6]` = `[Δpos(3) ‖ rotvec(R_rel)(3)]`.

    `hand_s`/`hand_sK` are DyWA `hand_state` (`pose6d`: `pos(3) ‖ rot6d(6)`, 9-dim).
    The rotation part is the *relative* rotation `R_rel = R_{s+K} · R_sᵀ` — NOT a
    raw difference of the two rot6d encodings, which would depend on the absolute
    starting orientation and so would not be a true net motion.

    Δpos is returned in world units; the caller scale-normalizes it into the shared
    frame (`to_frame_vec`). The rotvec is frame-invariant (the frame is
    translate + isotropic-scale, no rotation).

    NaN-safety: reset/pad frames store an all-zero `hand_state`, whose zero-norm
    rot6d column makes `rot6d_to_matrix` divide by zero → NaN. Any window touching
    such a frame is zeroed (treated as "no motion") so a stray invalid frame cannot
    poison the loss. (~1% of DyWA windows; the object flow for the same window is
    unaffected since it comes from `object_pose`, not `hand_state`.)
    """
    dpos = hand_sK[..., :3] - hand_s[..., :3]
    R_s = rot6d_to_matrix(hand_s[..., 3:9])
    R_k = rot6d_to_matrix(hand_sK[..., 3:9])
    R_rel = R_k @ R_s.transpose(-1, -2)
    out = th.cat([dpos, matrix_to_rotvec(R_rel)], dim=-1)
    bad = ~th.isfinite(out).all(dim=-1, keepdim=True)
    return th.where(bad, th.zeros_like(out), out)


def rigid_flow(cloud: th.Tensor, pose_s: th.Tensor, pose_sK: th.Tensor) -> th.Tensor:
    """Per-point rigid flow of the object over a window (world frame, unnormalized).

    cloud   : `[*L, N, 3]`  object points at step s
    pose_s  : `[*L, 7]`     object pose `(pos ‖ quat_xyzw)` at step s   (privileged)
    pose_sK : `[*L, 7]`     object pose at step s+K                     (privileged)
    Returns `flow [*L, N, 3] = T·p − p`, `T = pose_sK ∘ pose_s⁻¹` (SE(3)).

    Computed OUTSIDE the autograd graph — flow is a constant target/input, so the
    quaternion math need not be double-backward safe.
    """
    T = compose_pose_tq(pose_sK, invert_pose_tq(pose_s))     # [*L, 7]
    moved = apply_pose_tq(T.unsqueeze(-2), cloud)            # [*L, N, 3]
    return moved - cloud


# ─────────────────────────────────────────────────────────────────────────────
# 5. TTT inner update — latent MSE against the OutcomeEncoder target.
# ─────────────────────────────────────────────────────────────────────────────
def ttt_inner_update(q: th.Tensor,
                     dyn: DynamicsModel,
                     outcome_enc: OutcomeEncoder,
                     hist_clouds: th.Tensor,
                     hist_actions: th.Tensor,
                     hist_flow: th.Tensor,
                     ttt_alpha: float = TTT_ALPHA,
                     create_graph: bool = True,
                     detach_q: bool = False
                     ) -> Tuple[th.Tensor, th.Tensor]:
    """One inner TTT step on `q` using latent MSE over the history.

    Shapes (arbitrary leading "task" dims `*L` — () for one episode, `(E,)` for
    E parallel envs):
        q            : [*L, D]
        hist_clouds  : [*L, H, N, 3]   (shared-frame normalized, before-cloud `cloud_s`)
        hist_actions : [*L, H, A]      (macro-action over the window)
        hist_flow    : [*L, H, N, 3]   (shared-frame-normalized rigid flow over the window)

    Target latent = `OutcomeEncoder(cloud_s, flow)`; prediction = `dyn(q, macro, cloud_s)`.

    `create_graph=True` enables the outer meta-gradient (offline second order).
    `detach_q=True` differentiates the inner loss against a detached copy of q,
    so q0 is updated only through the linear `q_k = q - α·grad` term
    (first-order, for the online path).

    Returns `(q_k, inner_loss_per_task)`.
    """
    L = q.shape[:-1]
    D = q.shape[-1]
    H, N = hist_clouds.shape[-3], hist_clouds.shape[-2]
    A = hist_actions.shape[-1]

    q_diff = q.detach().requires_grad_(True) if detach_q else q
    q_exp = q_diff.unsqueeze(-2).expand(*L, H, D)            # [*L, H, D]

    q_flat = q_exp.reshape(-1, D)                            # [M, D],  M = prod(L)*H
    cl_flat = hist_clouds.reshape(-1, N, 3)
    ac_flat = hist_actions.reshape(-1, A)
    fl_flat = hist_flow.reshape(-1, N, 3)

    z_pred = dyn(q_flat, ac_flat, cl_flat)                  # [M, D]
    z_act = outcome_enc(cl_flat, fl_flat)                  # [M, D]  (pcd=cloud_s, flow)

    z_pred = z_pred.reshape(*L, H, -1)
    z_act = z_act.reshape(*L, H, -1)

    inner_per = ((z_pred - z_act) ** 2).mean(dim=(-1, -2))  # [*L]
    inner_loss = inner_per.sum()

    q_grad = th.autograd.grad(inner_loss, q_diff, create_graph=create_graph)[0]
    q_grad = th.nan_to_num(q_grad, nan=0.0, posinf=0.0, neginf=0.0)

    # Conditional per-task normalisation: cap only when large, preserving
    # magnitude when the inner-loss gradient is weak (bars behaviour, batched).
    norm = q_grad.norm(dim=-1, keepdim=True) + 1e-8
    q_grad = th.where(norm > 1.0, q_grad / norm, q_grad)

    q_k = q - ttt_alpha * q_grad
    return q_k, inner_per.detach()


# ─────────────────────────────────────────────────────────────────────────────
# 6. GMM isotropic NLL loss — kept for reference / optional policy heads.
#    (DyWA's outer loss is GaussianKLDivLoss; this is not used by default.)
# ─────────────────────────────────────────────────────────────────────────────
class GMMLoss(nn.Module):
    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, weights, means, gt_action):
        gt_exp = gt_action.unsqueeze(1)
        sq_dist = ((gt_exp - means) ** 2).sum(dim=-1)
        exponent = th.log(weights + 1e-8) - sq_dist / (2 * self.sigma ** 2)
        return -th.logsumexp(exponent, dim=1).mean()
