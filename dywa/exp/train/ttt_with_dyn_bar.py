"""
TTT-with-dynamics for the pick-bar task — v3 architecture.

This is the v1 pick-bar pipeline (`ttt_with_dyn_bar.py`) with **only the
model architecture** swapped for the one in `ttt_doors_model.py`: manual-attention
FiLM transformers. The data pipeline (`BarDataset`) and the meta-training loop
structure are kept from v1.

Why this architecture:

  • Manual matmul/softmax attention (FiLMTransformerBlock / SelfAttnBlock) so
    double backward works under create_graph=True in the TTT inner loop.
  • OutcomeEncoder ends in LayerNorm(latent_dim) — there is no constant-output
    minimum, so the encoder cannot collapse.
  • GMMHead is a FiLM-conditioned per-point transformer with concat-q heads and
    zero-init q-residual paths; FiLM uses non-zero init so q-conditioning is
    live from epoch 0.
  • DynamicsModel is a per-point FiLM transformer over the actual before-pcd
    (not a pooled embedding), with the same non-zero FiLM init.

Adaptations vs `ttt_doors_model.py`:

  • No per-point flow is available in the pick-bar dataset, so OutcomeEncoder is
    a plain manual-attention transformer over the post-attempt point cloud.
  • The action is a 3-D grasp position (no direction sub-vector).
  • The separate scene encoder (`PointCloudEncoder`) is eliminated.
  • Inner-loop step uses TTT_ALPHA = 0.1 and a fixed sigma = 0.1, matching v1.
"""

import os
# PyTorch 2.1.x: `expandable_segments:True` (set globally in this env's
# PYTORCH_CUDA_ALLOC_CONF) triggers an internal CUDACachingAllocator assert
# during gradient-checkpointed double backward. Disable it for this process —
# this must run before torch is imported anywhere.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
import wandb

# Reuse v1's dataset unchanged — it already yields
# (eval_pcd, eval_action, hist_obs, hist_actions, hist_outcomes).
from ttt_with_dyn_bar import BarDataset


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
LATENT_DIM   = 64
ACTION_DIM   = 3       # 3D grasp position (world coords)
TTT_ALPHA    = 0.1     # inner-loop step size
N_PTS        = 1200    # fixed point cloud size (subsample / oversample to this)

# Backbone width / depth
D_MODEL        = 128
NHEAD          = 4
GMM_LAYERS     = 4
DYN_LAYERS     = 3
OUTCOME_LAYERS = 2
K_COMPONENTS   = 4

# Gradient checkpointing on transformer blocks — recomputes activations during
# backward instead of storing them. Verified bit-exact with double backward.
USE_CHECKPOINT = True

# TTT inner-update meta-learning semantics (see ttt_inner_update docstring).
#   False (default) = v1-style: differentiate the inner loss w.r.t. q0 directly,
#                     so q0 receives full second-order MAML curvature.
#   True            = doors-style: differentiate w.r.t. a detached copy, so q0
#                     is updated only through the linear q_k term.
DETACH_Q = False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Attention blocks — manual matmul/softmax so double backward works.
# ─────────────────────────────────────────────────────────────────────────────
class FiLMTransformerBlock(nn.Module):
    """Pre-norm self-attention + FFN, with FiLM on the FFN output.

    Attention is implemented manually (matmul + softmax) rather than via
    nn.MultiheadAttention because the latter dispatches to fused SDPA kernels
    on CUDA that do NOT implement double backward — and we need double backward
    when create_graph=True in the inner loop.
    """
    def __init__(self, d_model, nhead, cond_dim, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model        = d_model
        self.nhead          = nhead
        self.d_head         = d_model // nhead
        self.scale          = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        # Pre-norm self-attention (manual)
        self.ln1          = nn.LayerNorm(d_model)
        self.qkv          = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out     = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout  = nn.Dropout(dropout)

        # Pre-norm FFN
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

        # FiLM near-identity init: small γ, zero β
        self.film = nn.Linear(cond_dim, 2 * d_model)
        nn.init.xavier_normal_(self.film.weight, gain=0.5)
        nn.init.zeros_(self.film.bias)

    def _attn(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(B, N, 3, self.nhead, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_dropout(attn)
        out    = torch.matmul(attn, v)
        out    = out.transpose(1, 2).contiguous().view(B, N, D)
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
    """Manual pre-norm self-attention + FFN, no FiLM conditioning.

    Same manual attention kernel as FiLMTransformerBlock (double-backward safe).
    Used by the OutcomeEncoder, where there is no conditioning input.
    """
    def __init__(self, d_model, nhead, dropout=0.1, use_checkpoint=False):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.d_model        = d_model
        self.nhead          = nhead
        self.d_head         = d_model // nhead
        self.scale          = self.d_head ** -0.5
        self.use_checkpoint = use_checkpoint

        self.ln1          = nn.LayerNorm(d_model)
        self.qkv          = nn.Linear(d_model, 3 * d_model, bias=True)
        self.attn_out     = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout  = nn.Dropout(dropout)

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

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn   = F.softmax(scores, dim=-1)
        attn   = self.attn_dropout(attn)
        out    = torch.matmul(attn, v)
        out    = out.transpose(1, 2).contiguous().view(B, N, D)
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
# 2. OutcomeEncoder — encodes the post-attempt point cloud (no-flow variant).
# ─────────────────────────────────────────────────────────────────────────────
class OutcomeEncoder(nn.Module):
    """pcd_after [B, N, 3] → outcome latent [B, latent_dim].

    Plain manual-attention transformer over the bar PCD after an attempt.
    The trailing LayerNorm prevents constant-output collapse.

    Called inside the inner loop with create_graph=True, so it must support
    double backward — hence SelfAttnBlock's manual attention.
    """
    def __init__(self, d_model=D_MODEL, nhead=NHEAD, num_layers=OUTCOME_LAYERS,
                 latent_dim=LATENT_DIM, use_checkpoint=USE_CHECKPOINT):
        super().__init__()
        self.embed  = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList([
            SelfAttnBlock(d_model=d_model, nhead=nhead,
                          use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, pcd_after):
        x = self.embed(pcd_after)
        for blk in self.blocks:
            x = blk(x)
        x = self.final_ln(x)
        pooled = x.mean(dim=1)                             # [B, d_model]
        return self.proj(pooled)                           # [B, latent_dim]


# ─────────────────────────────────────────────────────────────────────────────
# 3. GMM head — FiLM-conditioned per-point transformer (action_dim = 3).
# ─────────────────────────────────────────────────────────────────────────────
class GMMHead(nn.Module):
    """Per-point GMM head: each of the N points proposes K components → N*K total.

    eval_pcd goes through a FiLM-conditioned transformer (conditioned on q),
    then the per-point features are concatenated with a broadcast q for the
    weight/mean heads. A zero-init q-only residual path lets q nudge the output
    directly. The means are predicted as offsets from the eval points.
    """
    def __init__(self,
                 latent_dim=LATENT_DIM, k=K_COMPONENTS, action_dim=ACTION_DIM,
                 d_model=D_MODEL, num_layers=GMM_LAYERS, nhead=NHEAD,
                 use_checkpoint=USE_CHECKPOINT):
        super().__init__()
        self.k          = k
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.embed = nn.Linear(3, d_model)
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(d_model=d_model, nhead=nhead,
                                 cond_dim=latent_dim,
                                 use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Direct concat-q path
        self.weight_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, k),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(d_model + latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, k * action_dim),
        )

        # q-only residual path, zero-init.
        self.q_residual_weight = nn.Linear(latent_dim, k)
        self.q_residual_mean   = nn.Linear(latent_dim, k * action_dim)
        nn.init.zeros_(self.q_residual_weight.weight)
        nn.init.zeros_(self.q_residual_weight.bias)
        nn.init.zeros_(self.q_residual_mean.weight)
        nn.init.zeros_(self.q_residual_mean.bias)

    def forward(self, q, eval_pcd):
        B, N, _ = eval_pcd.shape

        x = self.embed(eval_pcd)
        for blk in self.blocks:
            x = blk(x, q)
        x = self.final_ln(x)

        q_broadcast = q.unsqueeze(1).expand(B, N, -1)
        feats = torch.cat([x, q_broadcast], dim=-1)

        w_logits = self.weight_head(feats)
        m_raw    = self.mean_head(feats).view(B, N, self.k, self.action_dim)

        # Zero-init residual paths
        w_res = self.q_residual_weight(q).unsqueeze(1)
        m_res = self.q_residual_mean(q).view(B, 1, self.k, self.action_dim)
        w_logits = w_logits + w_res
        m_raw    = m_raw    + m_res

        weights = F.softmax(w_logits.view(B, -1), dim=1)
        # Means are offsets from the eval points (position only, action_dim=3).
        means   = (eval_pcd.unsqueeze(2) + m_raw).view(B, -1, self.action_dim)
        return weights, means


# ─────────────────────────────────────────────────────────────────────────────
# 4. Dynamics — per-point FiLM transformer, single latent head (action_dim = 3).
# ─────────────────────────────────────────────────────────────────────────────
class DynamicsModel(nn.Module):
    """Per-point transformer dynamics with a single latent output head.

    Predicts the action-outcome latent given (q, action, before-pcd); the target
    is OutcomeEncoder(after-pcd). FiLM conditioning is on q alone — the action
    enters the trunk via the per-point input features (relative position to the
    grasp point). With a 3-D action there is no direction, so the per-point
    feature is just [pcd ‖ (action_pos - pcd)].
    """
    def __init__(self,
                 latent_dim=LATENT_DIM, action_dim=ACTION_DIM,
                 d_model=D_MODEL, num_layers=DYN_LAYERS, nhead=NHEAD,
                 use_checkpoint=USE_CHECKPOINT):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # per-point input: [pcd(3) ‖ rel_pos(3)] = 6 dims (no action direction)
        self.embed = nn.Linear(6, d_model)
        self.blocks = nn.ModuleList([
            FiLMTransformerBlock(d_model=d_model, nhead=nhead,
                                 cond_dim=latent_dim,
                                 use_checkpoint=use_checkpoint)
            for _ in range(num_layers)
        ])
        self.final_ln = nn.LayerNorm(d_model)

        # Pooled outcome-latent head — target is OutcomeEncoder(pcd_after).
        self.latent_head = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def _backbone(self, q, action, pcd):
        """Shared transformer trunk: returns per-point features [B, N, d_model]."""
        action_pos = action[:, :3].unsqueeze(1)            # [B, 1, 3]
        rel_pos    = action_pos - pcd                      # [B, N, 3]
        per_point  = torch.cat([pcd, rel_pos], dim=-1)     # [B, N, 6]

        x = self.embed(per_point)
        for blk in self.blocks:
            x = blk(x, q)
        x = self.final_ln(x)
        return x

    def forward(self, q, action, pcd):
        """Returns outcome_latent [B, latent_dim]."""
        feats  = self._backbone(q, action, pcd)
        latent = self.latent_head(feats.mean(dim=1))
        return latent


# ─────────────────────────────────────────────────────────────────────────────
# 5. Loss — isotropic GMM NLL, position only (fixed sigma).
# ─────────────────────────────────────────────────────────────────────────────
class GMMLoss(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, weights, means, gt_action):
        gt_exp   = gt_action.unsqueeze(1)
        sq_dist  = ((gt_exp - means) ** 2).sum(dim=-1)
        exponent = torch.log(weights + 1e-8) - sq_dist / (2 * self.sigma ** 2)
        return -torch.logsumexp(exponent, dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6. TTT inner update — latent MSE against the OutcomeEncoder target.
# ─────────────────────────────────────────────────────────────────────────────
def ttt_inner_update(q_var, dyn, outcome_enc,
                     hist_pcds, hist_actions, hist_outcomes,
                     create_graph=True, detach_q=DETACH_Q, retain_graph=None):
    """One inner step on q using latent MSE over the history.

      z_preds    = dyn(q, hist_actions, hist_pcds)     # before-pcd → outcome latent
      z_actuals  = outcome_enc(hist_outcomes)          # after-pcd  → outcome latent
      inner_loss = MSE(z_preds, z_actuals)

    `detach_q` controls how q0 receives the outer meta-gradient:
      • False (default, v1-style): the inner loss is differentiated w.r.t.
        q_var directly, so q0 receives full second-order MAML curvature.
      • True (doors-style): the inner gradient is taken against a detached
        copy of q_var, so q0 is updated only by the outer loss flowing
        through the linear `q_k = q_var - α·q_grad` term.

    `retain_graph=None` reproduces torch's default (retain iff create_graph),
    i.e. every pre-existing caller is unchanged.  `ttt_nograd_bar_v3.py` passes
    True so that, with create_graph=False, `inner_loss` can still be backwarded
    afterwards as a direct auxiliary objective.

    Returns (q_k, inner_loss).
    """
    H = hist_pcds.shape[0]
    q_diff = q_var.detach().requires_grad_(True) if detach_q else q_var
    q_exp  = q_diff.unsqueeze(0).expand(H, -1)

    z_preds   = dyn(q_exp, hist_actions, hist_pcds)        # [H, latent_dim]
    z_actuals = outcome_enc(hist_outcomes)                 # [H, latent_dim]
    inner_loss = F.mse_loss(z_preds, z_actuals)

    q_grad = torch.autograd.grad(inner_loss, q_diff, create_graph=create_graph,
                                 retain_graph=retain_graph)[0]
    q_grad = torch.nan_to_num(q_grad, nan=0.0, posinf=0.0, neginf=0.0)
    # Conditional normalisation: cap only when large, preserving magnitude when
    # the inner-loss gradient is weak.
    q_grad_norm = q_grad.norm() + 1e-8
    if q_grad_norm > 1.0:
        q_grad = q_grad / q_grad_norm

    q_k = q_var - TTT_ALPHA * q_grad
    return q_k, inner_loss


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_metrics(outcome_enc, gmm, dyn, q0, dataset, device, sigma=0.1):
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    outcome_enc.eval(); gmm.eval(); dyn.eval()
    loss_fn = GMMLoss(sigma=sigma)

    pre_losses, post_losses = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        eval_pcd, eval_action, hist_obs, hist_actions, hist_outcomes = batch
        eval_pcd      = eval_pcd.float().to(device)
        eval_action   = eval_action.float().to(device)
        hist_obs      = hist_obs.squeeze(0).float().to(device)
        hist_actions  = hist_actions.squeeze(0).float().to(device)
        hist_outcomes = hist_outcomes.squeeze(0).float().to(device)

        with torch.no_grad():
            w_pre, m_pre = gmm(q0.unsqueeze(0), eval_pcd)
            pre_losses.append(loss_fn(w_pre, m_pre, eval_action).item())

        if hist_obs.shape[0] > 0:
            with torch.enable_grad():
                q_post, _ = ttt_inner_update(q0, dyn, outcome_enc,
                                             hist_obs, hist_actions, hist_outcomes,
                                             create_graph=False)
            q_post = torch.nan_to_num(q_post.detach(), nan=0.0)
        else:
            q_post = q0.clone()

        with torch.no_grad():
            w_post, m_post = gmm(q_post.unsqueeze(0), eval_pcd)
            post_losses.append(loss_fn(w_post, m_post, eval_action).item())

    outcome_enc.train(); gmm.train(); dyn.train()
    return float(np.mean(pre_losses)), float(np.mean(post_losses))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Meta-training
# ─────────────────────────────────────────────────────────────────────────────
def run_meta_training(train_set, val_set,
                      epochs=100, outer_lr=1e-3,
                      effective_batch_size=32,
                      end_dropout=0.3, sigma=0.1,
                      save_dir='./checkpoints/ttt_dyn_bar_v3',
                      project_name='meta_q_learning',
                      save_every=None, track_gap=False,
                      run_name='ttt-dynamics-bar-v3'):
    """Meta-train the TTT stack.

    `save_every` / `track_gap` / `run_name` are opt-in and default to the
    original behaviour, so existing callers are unaffected:
      save_every : also write `epoch{N}_ttt_bar.pt` every N epochs.
      track_gap  : also track `best_gap_ttt_bar.pt`, the checkpoint maximising
                   the TTT adaptation gain (val_pre - val_post).
    """
    wandb.init(
        project=project_name, name=run_name, entity='r-pad',
        config={
            'epochs': epochs, 'outer_lr': outer_lr,
            'effective_batch_size': effective_batch_size,
            'history_dropout': end_dropout, 'sigma': sigma,
            'latent_dim': LATENT_DIM, 'ttt_alpha': TTT_ALPHA,
            'action_dim': ACTION_DIM, 'n_pts': N_PTS,
            'd_model': D_MODEL, 'nhead': NHEAD,
            'gmm_layers': GMM_LAYERS, 'dyn_layers': DYN_LAYERS,
            'outcome_layers': OUTCOME_LAYERS, 'k_components': K_COMPONENTS,
            'use_checkpoint': USE_CHECKPOINT,
            'augment_prob': train_set.augment_prob,
        }
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_dir, exist_ok=True)

    outcome_enc = OutcomeEncoder().to(device)
    gmm         = GMMHead().to(device)
    dyn         = DynamicsModel().to(device)
    q0          = nn.Parameter(torch.randn(LATENT_DIM, device=device) * 0.1)

    optimizer = optim.Adam(
        list(outcome_enc.parameters()) +
        list(gmm.parameters()) + list(dyn.parameters()) + [q0],
        lr=outer_lr
    )
    loss_fn    = GMMLoss(sigma=sigma)
    dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)

    print(f"Starting meta-training on {device} "
          f"(effective batch {effective_batch_size}, "
          f"augment_prob={train_set.augment_prob})...")
    print(f"  OutcomeEnc params: {sum(p.numel() for p in outcome_enc.parameters())/1e6:.2f}M")
    print(f"  GMM        params: {sum(p.numel() for p in gmm.parameters())/1e6:.2f}M")
    print(f"  DYN        params: {sum(p.numel() for p in dyn.parameters())/1e6:.2f}M")

    best_post_loss = float('inf')
    best_gap       = float('-inf')
    optimizer.zero_grad()

    for epoch in range(epochs):
        outcome_enc.train(); gmm.train(); dyn.train()
        total_loss         = 0.0
        current_batch_loss = 0.0
        actual_batch_i     = 0

        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader),
                              desc=f'Epoch {epoch + 1}'):
            eval_pcd, eval_action, hist_obs, hist_actions, hist_outcomes = batch
            eval_pcd      = eval_pcd.float().to(device)
            eval_action   = eval_action.float().to(device)
            hist_obs      = hist_obs.squeeze(0).float().to(device)
            hist_actions  = hist_actions.squeeze(0).float().to(device)
            hist_outcomes = hist_outcomes.squeeze(0).float().to(device)

            use_history = (np.random.rand() >= end_dropout) and (hist_obs.shape[0] > 0)

            # ── INNER LOOP ────────────────────────────────────────────────
            if use_history:
                q_k, _ = ttt_inner_update(q0, dyn, outcome_enc,
                                          hist_obs, hist_actions, hist_outcomes,
                                          create_graph=True)
                q_k = torch.nan_to_num(q_k, nan=0.0)
            else:
                q_k = q0

            # ── OUTER LOOP ────────────────────────────────────────────────
            weights, means = gmm(q_k.unsqueeze(0), eval_pcd)
            outer_loss     = loss_fn(weights, means, eval_action)

            (outer_loss / effective_batch_size).backward()
            current_batch_loss += outer_loss.item()

            if (i + 1) % effective_batch_size == 0:
                nn.utils.clip_grad_norm_(
                    list(outcome_enc.parameters()) +
                    list(gmm.parameters()) + list(dyn.parameters()) + [q0],
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                total_loss += current_batch_loss

                if actual_batch_i % 50 == 0:
                    wandb.log({'batch_loss': current_batch_loss / effective_batch_size})

                current_batch_loss = 0.0
                actual_batch_i    += 1

        if len(dataloader) % effective_batch_size != 0:
            nn.utils.clip_grad_norm_(
                list(outcome_enc.parameters()) +
                list(gmm.parameters()) + list(dyn.parameters()) + [q0],
                max_norm=1.0
            )
            optimizer.step()
            optimizer.zero_grad()
            total_loss += current_batch_loss

        avg_train_loss = total_loss / max(len(dataloader), 1)

        print(f'Epoch {epoch + 1} | Evaluating...')
        val_pre,   val_post   = evaluate_metrics(outcome_enc, gmm, dyn, q0,
                                                  val_set,   device, sigma)

        print(f'  Train NLL : {avg_train_loss:.4f}')
        print(f'  Val   NLL : pre={val_pre:.4f}  post={val_post:.4f}')

        wandb.log({
            'epoch':          epoch + 1,
            'train_loss':     avg_train_loss,
            'val_pre_nll':    val_pre,
            'val_post_nll':   val_post,
        })

        ckpt = {
            'epoch':       epoch + 1,
            'outcome_enc': outcome_enc.state_dict(),
            'gmm':         gmm.state_dict(),
            'dyn':         dyn.state_dict(),
            'q0':          q0.detach().cpu(),
            'optimizer':   optimizer.state_dict(),
        }
        if val_post < best_post_loss:
            best_post_loss = val_post
            torch.save(ckpt, os.path.join(save_dir, 'best_ttt_bar.pt'))
            print(f'  [*] Best saved (post NLL: {best_post_loss:.4f})')

        if track_gap:
            gap = val_pre - val_post
            if gap > best_gap:
                best_gap = gap
                torch.save(ckpt, os.path.join(save_dir, 'best_gap_ttt_bar.pt'))
                print(f'  [*] Best-gap saved (val_pre - val_post: {best_gap:.4f})')

        if save_every and (epoch + 1) % save_every == 0:
            torch.save(ckpt, os.path.join(save_dir, f'epoch{epoch + 1}_ttt_bar.pt'))
            print(f'  [*] Periodic checkpoint saved (epoch {epoch + 1})')

        torch.save(ckpt, os.path.join(save_dir, 'latest_ttt_bar.pt'))

    print('Training complete.')
    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATA_PATH = '/home/liyuan/ng_doors/dataset_pick-bar-v2.pkl'

    train_set, val_set = BarDataset.load_split(
        DATA_PATH,
        train_ratio=0.9,
        n_pts=N_PTS,
        augment_prob=0.1,
    )
    print(f"Train: {len(train_set)} samples, Val: {len(val_set)} samples")

    run_meta_training(
        train_set, val_set,
        epochs=2000,
        outer_lr=1e-3,
        effective_batch_size=64,
        end_dropout=0.3,
        sigma=0.1,
        save_dir='./checkpoints/ttt_dyn_bar_v3',
        project_name='meta_q_learning',
    )
