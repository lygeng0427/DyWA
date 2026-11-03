#!/usr/bin/env python3
"""Shared pieces for the GAP-style track models: normalizer, RoPE, attention.

Ported from third_party/GAP with the changes torch 1.11 / py3.8 forces:

  * LinearNormalizer -- GAP's version stores stats in an `nn.ParameterDict`
    keyed to plain tensors, which torch <=1.11 rejects outright
    ("cannot assign 'torch.FloatTensor' as parameter"). Rewritten on
    register_buffer, which also keeps the stats inside state_dict so a
    checkpoint carries its own normalization.
  * attention -- F.scaled_dot_product_attention needs torch >=2.0; written out
    explicitly here.
  * RoPE4D.apply -> RoPE4D.rotate, because `apply` shadows nn.Module.apply and
    would break on any model.apply(init_fn) recursion.
"""
import math
import os
from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# Recompute encoder-layer activations in backward instead of storing them.
# OFF by default, so every existing run is byte-identical. Meta-training turns it
# on (`GAP_CHECKPOINT=1`): there the support forward is retained through a DOUBLE
# backward, attention scores dominate memory at 409^2 x 8 heads x 6 layers, and
# the meta-batch -- i.e. the outer supervision per step -- is what that memory
# buys. `use_reentrant=False` is required: the reentrant implementation does not
# support double backward, and torch 1.11 is the first version to offer it.
_CKPT = os.environ.get('GAP_CHECKPOINT', '0') == '1'


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------
class FieldNorm(nn.Module):
    """Affine x' = scale * x + offset, with the stats living in buffers."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('scale', th.ones(dim))
        self.register_buffer('offset', th.zeros(dim))

    def _apply_affine(self, x, scale, offset):
        d = scale.shape[0]
        shape = x.shape
        y = x.reshape(-1, d) * scale + offset
        return y.reshape(shape)

    def normalize(self, x):
        return self._apply_affine(x, self.scale, self.offset)

    def unnormalize(self, x):
        return self._apply_affine(x, 1.0 / self.scale, -self.offset / self.scale)

    @th.no_grad()
    def fit_limits(self, data: th.Tensor, eps: float = 1e-4):
        """Map [min, max] -> [-1, 1] per channel (GAP's 'limits' mode)."""
        d = data.reshape(-1, data.shape[-1]).float()
        lo, hi = d.min(0).values, d.max(0).values
        rng = hi - lo
        dead = rng < eps
        rng = th.where(dead, th.full_like(rng, 2.0), rng)
        scale = 2.0 / rng
        offset = -1.0 - scale * lo
        offset = th.where(dead, -lo, offset)             # constant dims -> 0
        self.scale.copy_(scale)
        self.offset.copy_(offset)
        return self

    @th.no_grad()
    def fit_standard(self, data: th.Tensor, eps: float = 1e-4):
        """Map each channel to zero mean / unit std.

        For a DIFFUSION target the scale is not cosmetic: x_t = sqrt(ab)*x0 +
        sqrt(1-ab)*eps mixes the target with UNIT-variance gaussian noise, so a
        target whose std is s has signal-to-noise s*sqrt(ab/(1-ab)) and is
        invisible over most of the schedule when s << 1. fit_limits sets the
        scale from min/max, which is right for a bounded, roughly uniform
        quantity but collapses a heavy-tailed one: the commanded action's
        per-dim std came out 0.265-0.386 (3x too small) because 36% of raw
        values already sit outside [-1, 1] and the limits are set by outliers.
        """
        d = data.reshape(-1, data.shape[-1]).float()
        mu, sd = d.mean(0), d.std(0)
        dead = sd < eps
        sd = th.where(dead, th.ones_like(sd), sd)
        self.scale.copy_(1.0 / sd)
        self.offset.copy_(-mu / sd)
        return self

    @th.no_grad()
    def fit_isotropic_rms(self, rms: float):
        """Single global scale, zero offset -- for 3-D displacements.

        Per-axis min/max scaling would distort displacement DIRECTIONS, and a
        nonzero offset would stop a zero displacement mapping to zero. Both
        matter for tracks, so this is deliberately not fit_limits.
        """
        rms = max(float(rms), 1e-4)
        self.scale.fill_(1.0 / rms)
        self.offset.zero_()
        return self


class Normalizer(nn.Module):
    """A dict of FieldNorms; unknown keys pass through untouched."""

    def __init__(self):
        super().__init__()
        self.fields = nn.ModuleDict()

    def add(self, key: str, field: FieldNorm):
        self.fields[key] = field
        return self

    def __contains__(self, key):
        return key in self.fields

    def __getitem__(self, key) -> FieldNorm:
        return self.fields[key]

    def normalize(self, d: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
        return {k: (self.fields[k].normalize(v) if k in self.fields else v)
                for k, v in d.items()}


# ---------------------------------------------------------------------------
# 4-D RoPE (world x, y, z in metres + a step axis)
# ---------------------------------------------------------------------------
def _rotate_half(x: th.Tensor) -> th.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return th.stack((-x2, x1), dim=-1).flatten(-2)


class RoPE4D(nn.Module):
    """Ported from GAP. NOTE: the rotation entry point is `rotate`, not
    `apply` -- nn.Module.apply exists and recursing into a child that shadows
    it raises TypeError."""

    def __init__(self, head_dim: int, spatial_base: float = 100.0,
                 time_base: float = 100.0, spatial_scale: float = 0.5,
                 time_scale: float = 16.0):
        super().__init__()
        assert head_dim % 8 == 0, f'head_dim must be divisible by 8, got {head_dim}'
        per = head_dim // 4
        per -= per % 2
        dims = [per, per, per, head_dim - 3 * per]
        assert all(d > 0 and d % 2 == 0 for d in dims), f'bad axis split {dims}'
        self.axis_dims = dims
        self.spatial_scale = spatial_scale
        self.time_scale = time_scale
        for i, (name, base) in enumerate(zip(
                ['x', 'y', 'z', 't'],
                [spatial_base, spatial_base, spatial_base, time_base])):
            d = dims[i]
            inv = 1.0 / (base ** (th.arange(0, d, 2).float() / d))
            self.register_buffer(f'inv_freq_{name}', inv, persistent=False)

    def build_cos_sin(self, coords: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """coords [B, N, 4] -> cos, sin each [B, N, head_dim]."""
        assert coords.shape[-1] == 4, f'expected 4-D coords, got {coords.shape}'
        scales = coords.new_tensor([self.spatial_scale, self.spatial_scale,
                                    self.spatial_scale, self.time_scale])
        normed = coords / scales
        cs, sn = [], []
        for i, name in enumerate(['x', 'y', 'z', 't']):
            inv = getattr(self, f'inv_freq_{name}')
            ph = normed[..., i:i + 1] * inv
            ph = ph.repeat_interleave(2, dim=-1)
            cs.append(ph.cos()); sn.append(ph.sin())
        return th.cat(cs, -1), th.cat(sn, -1)

    def rotate(self, x, cos, sin, rope_mask: Optional[th.Tensor] = None):
        """x [B, h, N, hd]; cos/sin [B, N, hd]; rope_mask [B, N] (False = leave
        the token unrotated -- for tokens with no natural coordinate)."""
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        out = x * cos + _rotate_half(x) * sin
        if rope_mask is None:
            return out
        return th.where(rope_mask.unsqueeze(1).unsqueeze(-1), out, x)


# ---------------------------------------------------------------------------
# attention / transformer blocks
# ---------------------------------------------------------------------------
class RoPEAttention(nn.Module):
    """Multi-head self-attention, explicit softmax (no SDPA on torch 1.11)."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, query, key, value, rope=None, cos_sin=None,
                rope_mask=None):
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))
        if rope is not None:
            cos, sin = cos_sin
            q = rope.rotate(q, cos, sin, rope_mask=rope_mask)
            k = rope.rotate(k, cos, sin, rope_mask=rope_mask)
        att = th.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = th.softmax(att, dim=-1)
        if self.dropout_p > 0.0 and self.training:
            att = F.dropout(att, p=self.dropout_p)
        out = th.matmul(att, v)
        B, _, N, _ = out.shape
        out = out.transpose(1, 2).reshape(B, N, self.nhead * self.head_dim)
        return self.out_proj(out)


class EncoderLayer(nn.Module):
    """Pre-norm self-attention + FFN. Position encodings are added to Q and K
    only, never to V (GAP's convention).

    `film_dim > 0` adds FiLM on the FFN branch, applied before the residual
    add and with no post-FiLM norm -- so the conditioning cannot be normalized
    away. This mirrors ttt.py's FiLMTransformerBlock, which exists because
    DyWA's own FilmBlock routes around the conditioning through a
    q-independent highway and measurably freezes the belief.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, film_dim: int = 0):
        super().__init__()
        self.self_attn = RoPEAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.film = None
        if film_dim > 0:
            self.film = nn.Linear(film_dim, 2 * d_model)
            nn.init.xavier_normal_(self.film.weight, gain=0.5)
            nn.init.zeros_(self.film.bias)

    def _fwd(self, x, pos, rope, cos_sin, rope_mask, cond):
        h = self.ln1(x)
        hq = h if pos is None else h + pos
        x = x + self.drop(self.self_attn(hq, hq, h, rope=rope,
                                        cos_sin=cos_sin, rope_mask=rope_mask))
        h = self.ln2(x)
        h = self.linear2(self.drop(F.gelu(self.linear1(h))))
        if self.film is not None and cond is not None:
            gamma, beta = self.film(cond).chunk(2, dim=-1)
            h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)
        return x + self.drop(h)

    def forward(self, x, pos=None, rope=None, cos_sin=None, rope_mask=None,
                cond=None):
        # Gated on grad being enabled, NOT on self.training: the meta inner loop
        # runs the support forward in eval() (to match test time) and that is
        # exactly the pass whose graph is retained, so a `self.training` gate
        # would skip checkpointing precisely where it is needed. Positional args
        # only -- torch 1.11's checkpoint rejects stray keyword arguments.
        if _CKPT and th.is_grad_enabled():
            return checkpoint(self._fwd, x, pos, rope, cos_sin, rope_mask,
                              cond, use_reentrant=False)
        return self._fwd(x, pos, rope, cos_sin, rope_mask, cond)


class Encoder(nn.Module):
    def __init__(self, layer_fn, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([layer_fn() for _ in range(num_layers)])

    def forward(self, x, pos=None, rope=None, cos_sin=None, rope_mask=None,
                cond=None):
        for lyr in self.layers:
            x = lyr(x, pos=pos, rope=rope, cos_sin=cos_sin,
                    rope_mask=rope_mask, cond=cond)
        return x


def sinusoidal_2d(h: int, w: int, dim: int, device, temperature: float = 1e4):
    """GAP's 2-D grid position encoding for the patch tokens."""
    gy, gx = th.meshgrid(th.arange(h, device=device, dtype=th.float32),
                         th.arange(w, device=device, dtype=th.float32))
    assert dim % 4 == 0
    omega = th.arange(dim // 4, device=device, dtype=th.float32)
    omega = 1.0 / (temperature ** (omega / (dim / 4.0)))
    y = gy.flatten()[:, None] * omega[None, :]
    x = gx.flatten()[:, None] * omega[None, :]
    return th.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


def timestep_embedding(t: th.Tensor, dim: int = 256) -> th.Tensor:
    half = dim // 2
    freqs = th.exp(-math.log(1e4) * th.arange(half, device=t.device,
                                              dtype=th.float32) / half)
    a = t.float()[:, None] * freqs[None, :]
    return th.cat([a.cos(), a.sin()], dim=-1)
