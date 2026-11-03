#!/usr/bin/env python3
"""Second-order meta-training of a dynamics belief, with a POLICY outer loss.

    METAP_EVAL_DATA=... METAP_VAL_IDS_FROM=... \
    bash dywa/exp/scripts/train_meta_policy.sh 0

Two models, no shared weights:

  INNER  `LatentTrackModel` (trunk=transformer, direct FiLM on the raw latent).
         Its object-only track error on already-elapsed windows defines
         dL/dq; `meta_ttt.adapt` takes K unit-normalised steps.
  OUTER  `JointPolicyDiT`, conditioned on the adapted belief, denoising the
         action chunk and the scene tracks together.

The outer loss is differentiated THROUGH the inner update, so the gradient
reaches `q0`, `log_alpha`, the track model AND the policy. `detach_q=True`
drops that to FOMAML for the ablation.

WHY THE INNER TARGET IS GROUND TRUTH, AND WHY THAT MATTERS. The pick-bar
reference's inner target is `OutcomeEncoder(pcd_after)` -- a LEARNED function
supervised only by the meta-gradient, which is why it needs a trailing
LayerNorm to have no constant-output minimum and still showed partial collapse.
Here the target is the measured GT track. There is no collapse mode, no
anti-collapse trick and no random-init-z control to run, and the inner loss is
in millimetres. What is given up is that MAML can no longer reshape what
"mismatch" MEANS, only the predictor -- less expressive, and less degenerate.

WHAT STILL HAS TO BE WATCHED, because nothing here raises:

  * the inner model receives ONLY the meta-gradient, so nothing forces it to
    remain a track predictor. The degenerate case is a near-constant dL/dq,
    giving q_k ~ q0 + const -- a fixed bias the policy absorbs, which is the
    inert-step failure this program already hit once. `METAP_AUX` adds the
    inner model's own supervised loss as a direct term; it is ON by default.
  * the outer loss is stochastic in `(t, eps)`, so every pre/post comparison
    pins the seed (`meta_policy.policy_err`).
  * support and query come from ONE trajectory 16 frames apart, so the belief
    can encode pose rather than physics. `METAP_SHUFFLE_Q=1` at validation is
    the control.

The shard window is `train_stream_tracks.ShardWindow`, imported, so the
collector contract stays the one `_probe_stream_equivalence.py` verified; its
knobs are the `STREAM_*` ones. Everything owned by this file is `METAP_*`.
"""
import os

# Must precede any torch import (meta_train_ttt.py:30-33).
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import gc
import math
import sys
import time

import numpy as np
import torch as th
import torch.nn as nn
import wandb

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.fric.dataset import FricTrackDataset, read_grid, count_episodes
from models.fric.latent_model import LatentTrackModel
from models.policy.joint_dit import JointPolicyDiT

import meta_ttt as M
import meta_policy as MP
import train_stream_tracks as S
from train_meta_track import MetaEpisodeSampler, meta_collate, flat
from train_fric_track import fit_track_norm, PINNED_ALL, PHYS_NAMES

EVAL_DATA = os.environ.get('METAP_EVAL_DATA', '')
VAL_IDS_FROM = os.environ.get('METAP_VAL_IDS_FROM', '')
INIT_POLICY = os.environ.get('METAP_INIT_POLICY', '')
INIT_INNER = os.environ.get('METAP_INIT_INNER', '')
# Continue BOTH models from a previous metapolicy checkpoint. `INIT_POLICY`
# only seeds the outer model from a gate run and leaves the inner one random,
# which is right for a first launch and wrong for a second phase -- the belief
# and the policy are co-adapted, so restarting one of them discards the pairing.
INIT_FROM = os.environ.get('METAP_INIT_FROM', '')
SAVE_DIR = os.environ.get('METAP_SAVE_DIR',
                          '/home/user/DyWA/output/mass/ckpt_metapolicy')
Z_DIM = int(os.environ.get('METAP_Z_DIM', 32))
D_MODEL = int(os.environ.get('METAP_D_MODEL', 512))
DROPOUT = float(os.environ.get('METAP_DROPOUT', 0.1))
OBJ_WEIGHT = float(os.environ.get('METAP_OBJ_WEIGHT', 9.0))
LAMBDA = float(os.environ.get('METAP_LAMBDA', 1.0))
ACTION_MODE = os.environ.get('METAP_ACTION_MODE', 'command')
Q0_STD = float(os.environ.get('METAP_Q0_STD', 0.1))
ALPHA0 = float(os.environ.get('METAP_ALPHA0', 0.89))
# FiLM's xavier_normal(gain=0.5) init is calibrated for a d_model-wide
# conditioning vector; a raw z_dim-wide one emits gamma ~0.012 where a TRAINED
# working path sits at 0.25-0.43 (measured on the gate's mass arm). At that
# scale the belief moves the loss ~0.03% -- a hundredfold under the validation
# noise -- so `gain` is arithmetic on noise and the gradient that would grow the
# heads is itself negligible. Measured leverage at alpha=0.89:
#   scale  1 -> 0.17%   5 -> 2.10%   10 -> 7.85%   20 -> 28.4%
# 10 clears the ~1-2% block noise with margin and still leaves the heads 3x
# below the trained target, so they grow rather than start saturated.
# `_probe_film_scale.py` re-measures this for any new (z_dim, d_model).
FILM_SCALE = float(os.environ.get('METAP_FILM_SCALE', 10.0))
# METAP_ALPHA_SET overwrites the inner step size AFTER any warm-start load (the
# checkpoint's log_alpha would otherwise win, since it rides in state_dict);
# METAP_ALPHA_FREEZE keeps it there by excluding it from the optimizer.
#
# Both exist because dL/dalpha ~ 0 on the flat surface near q0, so the outer
# loss has no signal with which to GROW the step -- the r-series drove alpha
# 0.1035 -> 0.0544 while the measured optimum was ~5x the trained value. Use
# `_probe_belief_leverage.py` to pick a value: it reports how far the belief
# must travel before the loss moves more than the validation noise.
ALPHA_SET = float(os.environ.get('METAP_ALPHA_SET', 0.0))
ALPHA_FREEZE = os.environ.get('METAP_ALPHA_FREEZE', '0') == '1'
H = int(os.environ.get('METAP_H', 16))
K_INNER = int(os.environ.get('METAP_K', 1))
M_MAX = int(os.environ.get('METAP_M_MAX', 8))
EVERY = int(os.environ.get('METAP_EVERY', 16))
N_QUERY = int(os.environ.get('METAP_QUERY', 4))
HIST_DROP = float(os.environ.get('METAP_HIST_DROP', 0.3))
DETACH_Q = os.environ.get('METAP_DETACH_Q', '0') == '1'
AUX = float(os.environ.get('METAP_AUX', 1.0))
# MEASURED FAILURE THIS FIXES. With AUX alone the belief carried NO mass: the
# causally fitted q scored OOF R2 -0.591 against a shuffled null of -0.281, and
# even a non-causal 60-step whole-episode CEILING fit scored -0.443. The cause
# is AUX itself -- `meta_ttt.adapt` returns the loss from BEFORE the final step,
# so at K=1 it supervises the inner model at q0, a CONSTANT. That trains it to
# be the mass-marginal predictor, i.e. to ignore q, and the only pressure for q
# to mean anything was the second-order term (13.3% of the trunk gradient).
#
# MASS_AUX additionally supervises the inner model at `phys_embed(true mass)`,
# so its conditioning slot is forced to carry mass and the loss surface in q
# acquires a mass-aligned minimum for the inner loop to descend toward. True
# mass is used at TRAINING only -- exactly the privileged-information setup the
# teacher/student pipeline already runs on -- and never at test time, where q
# is fitted from motion alone.
#
# Q0_ANCHOR pulls the prior to `phys_embed(grid midpoint)`, so "blind" means
# "assume the mean mass" and the inner gradient starts on the embedding
# manifold rather than at an arbitrary point off it.
# Restrict the swept variable to the TEACHER's own DR. Outside [0.1, 0.5] the
# teacher degrades (success 0.84 -> 0.60, timeouts 11% -> 23%) and the measured
# "mass helps" is largely failure-detection: +5.3% out of range against +0.17%
# in. In range the effect is real but small and SLOW -- it needs ~10 passes to
# surface (+1.04% pooled at 300k steps, invisible at 60k).
#
# `PolicyShardWindow` reads the filter from train_policy's module scope, so the
# env must be set BEFORE that import. Explicit rather than a duplicated filter,
# which would be free to drift.
MASS_LO = float(os.environ.get('METAP_MASS_LO', 0.0))
MASS_HI = float(os.environ.get('METAP_MASS_HI', 0.0))
if MASS_LO:
    os.environ['POLICY_MASS_LO'] = str(MASS_LO)
if MASS_HI:
    os.environ['POLICY_MASS_HI'] = str(MASS_HI)

MASS_AUX = float(os.environ.get('METAP_MASS_AUX', 1.0))
Q0_ANCHOR = float(os.environ.get('METAP_Q0_ANCHOR', 1.0))
BATCH_EP = int(os.environ.get('METAP_BATCH_EP', 2))
TOTAL_STEPS = int(os.environ.get('METAP_TOTAL_STEPS', 400000))
BLOCK_STEPS = int(os.environ.get('METAP_BLOCK_STEPS', 5000))
LR = float(os.environ.get('METAP_LR', 1e-4))
WARMUP = int(os.environ.get('METAP_WARMUP', 4000))
WD = float(os.environ.get('METAP_WD', 1e-6))
SEED = int(os.environ.get('METAP_SEED', 0))
WORKERS = int(os.environ.get('METAP_WORKERS', 6))
DEVICE = os.environ.get('METAP_DEVICE', 'cuda:0')
VAL_EPS = int(os.environ.get('METAP_VAL_EPS', 24))
VAL_SEED = int(os.environ.get('METAP_VAL_SEED', 9000))
SHUFFLE_Q = os.environ.get('METAP_SHUFFLE_Q', '0') == '1'
LOG_EVERY = int(os.environ.get('METAP_LOG_EVERY', 50))
TAG = os.environ.get('METAP_TAG', '')
WANDB_PROJECT = os.environ.get('METAP_WANDB_PROJECT', 'dywa-meta-policy')
WANDB_MODE = os.environ.get('METAP_WANDB_MODE', 'online')


# Imported AFTER the POLICY_MASS_* env is set above: train_policy reads its
# filter at module load.
from train_policy import PolicyShardWindow, fit_action_norm      # noqa: E402


def norm_phys(p, lo, hi, log):
    """Physical scalar -> [-1, 1] on the KNOWN grid, matching
    `JointPolicyDiT.cond_vec`'s mass branch exactly. Grid endpoints, never
    dataset statistics: a held-out-mass run would otherwise normalise
    differently from the run it is compared against."""
    if log:
        p = p.clamp(min=1e-6).log()
        lo, hi = math.log(lo), math.log(hi)
    return 2.0 * (p - lo) / (hi - lo) - 1.0


def lr_at(step):
    """Warmup then constant -- a stream run must be resumable, and OneCycle
    needs `total_steps` up front and raises past it."""
    return LR * min(1.0, float(step + 1) / max(WARMUP, 1))


def load_policy_warm_start(policy, path):
    """Load a `train_policy.py` checkpoint into the conditioned policy.

    The gate's `none` arm has NO FiLM parameters, so this is necessarily
    `strict=False` -- and `strict=False` is exactly how a trunk mismatch loads
    nothing at all and leaves a random model training happily (the TTT section
    of CLAUDE.md records that failure). So assert the shape of the mismatch:
    every missing key must be a FiLM head, and nothing may be unexpected.
    """
    src = th.load(path, map_location='cpu')
    r = policy.load_state_dict(src['model'], strict=False)
    bad = [k for k in r.missing_keys if '.film.' not in k]
    assert not bad, 'warm start is missing non-FiLM weights: %s' % bad[:6]
    assert not r.unexpected_keys, r.unexpected_keys[:6]
    print('[init] policy warm-started from %s (cond=%s, step %s, val %s); '
          '%d FiLM tensors left at init'
          % (path, src['cfg']['cond'], src.get('gstep'),
             src.get('val', {}).get('action'), len(r.missing_keys)))
    return float(src['track_rms'])


def validate(inner, policy, val_eps, dev, phys_embed=None, pn=None):
    out = MP.evaluate_policy(inner, policy, val_eps, H, K_INNER, M_MAX, EVERY,
                             seed=VAL_SEED)
    out.update(MP.diagnostics(inner, policy, val_eps, H, K_INNER, M_MAX, EVERY,
                              phys_embed=phys_embed, phys_norm=pn,
                              seed=VAL_SEED))
    return out


def main():
    eval_paths = [p for p in EVAL_DATA.split(',') if p and os.path.exists(p)]
    assert eval_paths, 'METAP_EVAL_DATA is required (the frozen eval corpus)'
    assert VAL_IDS_FROM and os.path.exists(VAL_IDS_FROM), \
        'METAP_VAL_IDS_FROM is required'
    th.manual_seed(SEED)
    np.random.seed(SEED)
    dev = th.device(DEVICE)

    g = read_grid(eval_paths)
    col, grid = g['col'], g['grid']
    expect_pinned = {k: v for k, v in PINNED_ALL.items() if k != PHYS_NAMES[col]}
    print('[phys] swept=%s (col %d, %s) grid %.3f..%.3f'
          % (g['channel'], col, g['spacing'], g['lo'], g['hi']))

    LO = MASS_LO or g['lo']
    HI = MASS_HI or g['hi']
    ref = th.load(VAL_IDS_FROM, map_location='cpu')
    va_ids = np.sort(np.asarray([int(x) for x in ref['val_ids']], np.int64))
    if MASS_LO or MASS_HI:
        import h5py as _h5
        off, pm = 0, {}
        for q_ in eval_paths:
            with _h5.File(q_, 'r', locking=False) as f_:
                e_ = f_['episode_ends'][:]
                s_ = np.concatenate([[0], e_[:-1]])
                v_ = f_['phys_params'][:, col][s_]
            for i_, m_ in enumerate(v_):
                pm[off + i_] = float(m_)
            off += len(v_)
        n0 = len(va_ids)
        va_ids = np.asarray([e for e in va_ids if LO <= pm[int(e)] <= HI],
                            np.int64)
        print('[range] swept variable restricted to [%.3f, %.3f]: val %d -> %d'
              % (LO, HI, n0, len(va_ids)))
    assert va_ids.max() < count_episodes(eval_paths)
    assert [os.path.realpath(x) for x in ref['paths']] == \
           [os.path.realpath(x) for x in eval_paths[:len(ref['paths'])]], \
        'reference ckpt paths are not a prefix of METAP_EVAL_DATA'
    print('[data] val ids REUSED from %s: %d episodes'
          % (VAL_IDS_FROM, len(va_ids)))

    # Norms are fitted from the same fixed draw of the val corpus the gate used,
    # so the meta run's loss is on the same scale as the gate's. A warm-started
    # policy overrides track_rms from its own checkpoint, because FieldNorm's
    # buffers are PERSISTENT and ride inside state_dict -- the weights and the
    # recorded scalar must come from the same place.
    ds_norm = FricTrackDataset(eval_paths, phys_col=col,
                               episodes=[int(e) for e in va_ids])
    trk_norm, rms = fit_track_norm(ds_norm)
    act_norm = fit_action_norm(ds_norm)

    cfg_want = {'z_dim': Z_DIM, 'd_model': D_MODEL, 'dino_dim': S.DINO_DIM,
                'H': H}
    inner = LatentTrackModel(
        trunk='transformer', d_model=D_MODEL, action_mode=ACTION_MODE,
        obj_weight=OBJ_WEIGHT, dropout=DROPOUT, dino_dim=S.DINO_DIM,
        z_dim=Z_DIM, direct_film=True, film_init_scale=FILM_SCALE,
        q0_init_std=Q0_STD, alpha_init=(ALPHA_SET or ALPHA0),
        phys_lo=LO, phys_hi=HI, phys_log=(g['spacing'] == 'log'))
    inner.normalizer.add('gt_3d_track', FieldNorm(3).fit_isotropic_rms(rms))
    if INIT_INNER:
        src = th.load(INIT_INNER, map_location='cpu')
        inner.load_state_dict(src['model'], strict=True)
        rms = float(src['track_rms'])
        print('[init] inner warm-started from %s' % INIT_INNER)
    inner = inner.to(dev)

    policy = JointPolicyDiT(
        dino_dim=S.DINO_DIM, d_model=D_MODEL, dropout=DROPOUT,
        obj_weight=OBJ_WEIGHT, track_loss_weight=LAMBDA,
        cond='latent', cond_dim=Z_DIM, film_init_scale=FILM_SCALE,
        phys_lo=LO, phys_hi=HI, phys_log=(g['spacing'] == 'log'))
    policy.normalizer.add('gt_3d_track', FieldNorm(3).fit_isotropic_rms(rms))
    policy.normalizer.add('action', act_norm)
    if INIT_FROM:
        assert not (INIT_POLICY or INIT_INNER), \
            'METAP_INIT_FROM supersedes INIT_POLICY/INIT_INNER; set one'
        src = th.load(INIT_FROM, map_location='cpu')
        for k in ('z_dim', 'd_model', 'dino_dim', 'H'):
            assert src['cfg'][k] == cfg_want[k], (
                'continuation checkpoint has %s=%r, this run wants %r'
                % (k, src['cfg'][k], cfg_want[k]))
        inner.load_state_dict(src['inner'], strict=True)
        policy.load_state_dict(src['policy'], strict=True)
        rms = float(src['track_rms'])
        print('[init] CONTINUING both models from %s (step %s, val post %s)'
              % (INIT_FROM, src.get('gstep'),
                 src.get('val', {}).get('post')))
    elif INIT_POLICY:
        rms = load_policy_warm_start(policy, INIT_POLICY)
    policy = policy.to(dev)
    if ALPHA_SET > 0.0:
        with th.no_grad():
            inner.log_alpha.fill_(math.log(math.expm1(ALPHA_SET)))
        print('[init] alpha OVERRIDDEN to %.4f%s'
              % (float(inner.alpha()), ' and FROZEN' if ALPHA_FREEZE else ''))
    MP.assert_compatible(inner, policy)

    live = float(policy.normalizer['gt_3d_track'].scale.reshape(-1)[0])
    assert abs(live - 1.0 / rms) < 1e-6, (live, 1.0 / rms)

    # Same trap as track_rms, one field over: FieldNorm.scale/offset are
    # PERSISTENT buffers, so every load_state_dict above silently restored the
    # CHECKPOINT's action normaliser over the fit made here. A checkpoint
    # written before the action target was standardised carries a ~3x smaller
    # scale AND an output head trained to emit values on that scale -- loading
    # it is not a warm start, it is a different objective. Fail loudly.
    la = policy.normalizer['action'].scale.detach().cpu()
    if not th.allclose(la, act_norm.scale, rtol=0.05, atol=1e-6):
        raise SystemExit(
            '[norm] FATAL: the loaded checkpoint\'s action normaliser '
            'disagrees with the fit on this data.\n'
            '       loaded scale %.4f..%.4f   fitted scale %.4f..%.4f\n'
            '       median ratio loaded/fitted = %.3f\n'
            '       A ratio near 1/3 means a pre-standardisation checkpoint '
            '(fit_limits). Its policy weights are stale -- train the policy '
            'from scratch instead of warm-starting it.'
            % (la.min(), la.max(), act_norm.scale.min(), act_norm.scale.max(),
               float((la / act_norm.scale).median())))
    print('[norm] action normaliser VERIFIED standard: scale %.4f..%.4f'
          % (la.min(), la.max()))
    n_in = sum(p.numel() for p in inner.parameters())
    n_po = sum(p.numel() for p in policy.parameters())
    print('[model] inner %d + policy %d params | z_dim=%d K=%d m_max=%d '
          'every=%d query=%d alpha=%.4f'
          % (n_in, n_po, Z_DIM, K_INNER, M_MAX, EVERY, N_QUERY,
             float(inner.alpha())))
    print('[norm] track_rms = %.8f  FROZEN' % rms)

    win = PolicyShardWindow(S.STREAM_DIR, S.WINDOW_SHARDS, col, grid,
                            expect_pinned, eval_paths)
    win.wait_for_min()
    win.refresh()
    print('[stream] %d shards resident, %d frames'
          % (len(win.resident), win.n_frames))

    sel = va_ids[np.linspace(0, len(va_ids) - 1,
                             min(VAL_EPS, len(va_ids))).astype(int)]
    val_sets = [FricTrackDataset(eval_paths, phys_col=col, episodes=[int(e)])
                for e in sel]

    run = 'metapolicy_z%d_k%d%s' % (Z_DIM, K_INNER, ('_' + TAG) if TAG else '')
    out_dir = os.path.join(SAVE_DIR, run)
    os.makedirs(out_dir, exist_ok=True)
    # Maps the true mass into the SAME space the belief lives in, so the inner
    # model's conditioning slot means one thing and q can occupy it.
    phys_embed = nn.Sequential(nn.Linear(1, Z_DIM), nn.SiLU(),
                               nn.Linear(Z_DIM, Z_DIM)).to(dev)
    params = [q for n_, q in inner.named_parameters()
              if not (ALPHA_FREEZE and n_ == 'log_alpha')] \
        + list(policy.parameters()) + list(phys_embed.parameters())
    opt = th.optim.AdamW(params, lr=LR, weight_decay=WD)
    wandb.init(project=WANDB_PROJECT, name=run, mode=WANDB_MODE, dir=out_dir,
               config=dict(z_dim=Z_DIM, k_inner=K_INNER, m_max=M_MAX,
                           every=EVERY, n_query=N_QUERY, H=H, lam=LAMBDA,
                           hist_drop=HIST_DROP, detach_q=DETACH_Q, aux=AUX,
                           batch_ep=BATCH_EP, lr=LR, seed=SEED, wd=WD,
                           track_rms=rms, total_steps=TOTAL_STEPS,
                           film_scale=FILM_SCALE, alpha0=(ALPHA_SET or ALPHA0),
                           mass_aux=MASS_AUX, q0_anchor=Q0_ANCHOR,
                           init_policy=INIT_POLICY, init_inner=INIT_INNER,
                           init_from=INIT_FROM, act_norm='standard',
                           params_inner=n_in, params_policy=n_po))
    print('[wandb] %s' % getattr(wandb.run, 'url', '(offline)'))

    cfg = dict(z_dim=Z_DIM, d_model=D_MODEL, dino_dim=S.DINO_DIM,
               action_mode=ACTION_MODE, obj_weight=OBJ_WEIGHT, lam=LAMBDA,
               H=H, k_inner=K_INNER, m_max=M_MAX, every=EVERY,
               q0_init_std=Q0_STD, alpha_init=(ALPHA_SET or ALPHA0),
               direct_film=True, film_init_scale=FILM_SCALE,
               phys_lo=LO, phys_hi=HI, mass_lo=MASS_LO, mass_hi=MASS_HI,
               phys_log=(g['spacing'] == 'log'), phys_col=col, seed=SEED,
               act_norm='standard')

    step, best = 0, float('inf')
    while step < TOTAL_STEPS:
        added, evicted = win.refresh()
        gc.collect()
        dss = [d for _, _, d in win.resident]
        sampler = MetaEpisodeSampler(dss, H, M_MAX, EVERY, N_QUERY,
                                     BLOCK_STEPS * BATCH_EP, SEED + step)
        dl = th.utils.data.DataLoader(
            sampler, batch_size=BATCH_EP, shuffle=False, num_workers=WORKERS,
            collate_fn=meta_collate, pin_memory=True, drop_last=True,
            persistent_workers=False)
        print('[stream] step %d: %d shards (+%d/-%d), %d frames, %d episodes'
              % (step, len(win.resident), added, evicted, win.n_frames,
                 sampler.n_ep), flush=True)

        acc = {'loss': 0.0, 'action': 0.0, 'track': 0.0, 'inner_mm': 0.0,
               'mass_mm': 0.0, 'aux_share': 0.0}
        nacc, t0 = 0, time.time()
        inner.train()
        policy.train()
        for b in dl:
            if step >= TOTAL_STEPS or nacc >= BLOCK_STEPS:
                break
            for gp in opt.param_groups:
                gp['lr'] = lr_at(step)
            sup = flat({k: v.to(dev, non_blocking=True)
                        for k, v in b['sup'].items()})
            qry = flat({k: v.to(dev, non_blocking=True)
                        for k, v in b['qry'].items()})
            mask_full = b['sup_mask'].to(dev)
            mask = mask_full
            if HIST_DROP > 0.0:
                drop = th.rand(mask.shape[0], device=dev) < HIST_DROP
                mask = mask & ~drop.unsqueeze(1)
            # The mass auxiliary uses the UNDROPPED mask: history dropout is
            # about the TTT path, and teaching q's meaning should not be
            # skipped on 30% of episodes.
            inner.eval()
            sup['ep_mask'] = mask_full
            ph = b['phys'].to(dev).reshape(-1, 1)
            q_true = phys_embed(norm_phys(ph, LO, HI, g['spacing'] == 'log'))
            mass_loss = M.object_loss_per_ep(inner, sup, q_true).mean()
            q_mid = phys_embed(th.zeros(1, 1, device=dev)).reshape(-1)
            anchor = ((inner.q0 - q_mid.detach()) ** 2).sum()
            inner.train()

            sup['ep_mask'] = mask

            q = inner.q0.unsqueeze(0).expand(mask.shape[0], -1)
            # eval() for the inner forward so dropout cannot make the SAME
            # support give two different beliefs -- test time has no dropout,
            # and this is the pass whose graph is retained.
            inner.eval()
            qk, inner_loss = M.adapt(inner, sup, q, K_INNER,
                                     create_graph=True, detach_q=DETACH_Q)
            inner.train()
            qry['q'] = qk.repeat_interleave(N_QUERY, dim=0)
            loss, info = policy.compute_loss(qry)
            # Keeps the inner model an actual track predictor. Without it its
            # only gradient is the meta path, which does not require the inner
            # loss to mean anything -- the degenerate optimum is a constant
            # dL/dq, i.e. no adaptation at all.
            total = loss + AUX * inner_loss + MASS_AUX * mass_loss \
                + Q0_ANCHOR * anchor

            opt.zero_grad(set_to_none=True)
            total.backward()
            th.nn.utils.clip_grad_norm_(
                list(inner.parameters()) + list(policy.parameters())
                + list(phys_embed.parameters()), 1.0)
            opt.step()

            acc['loss'] += info['loss']
            acc['action'] += info['action']
            acc['track'] += info['track']
            acc['inner_mm'] += 1000.0 * float(inner_loss.detach())
            acc['mass_mm'] += 1000.0 * float(mass_loss.detach())
            # The aux term is raw METRES and the outer loss is normalised, so
            # their ratio drifts as the outer loss falls. Log the share rather
            # than leaving AUX=1.0 on faith.
            acc['aux_share'] += AUX * float(inner_loss.detach()) / max(
                float(total.detach()), 1e-9)
            nacc += 1
            step += 1
            if nacc % LOG_EVERY == 0:
                wandb.log({'train/%s' % k: v / nacc for k, v in acc.items()},
                          step=step)
                wandb.log({'train/lr': lr_at(step),
                           'train/alpha': float(inner.alpha()),
                           'window/frames': win.n_frames}, step=step)

        del dl
        gc.collect()
        th.cuda.empty_cache()

        eps = []
        for d_ in val_sets:
            data, t_rel, sc = M.cache_episode(d_, dev)
            eps.append((data, t_rel, sc))
        va = validate(inner, policy, eps, dev, phys_embed,
                      lambda x: norm_phys(x, LO, HI, g['spacing'] == 'log'))
        del eps
        th.cuda.empty_cache()

        print('  step %d | inner %.2f mass %.2f mm | pre %.5f post %.5f '
              'gain %+.2f%% (dpos %+.2f%% gains %+.2f%%) ceil %+.2f%% '
              'shuf %+.2f%% rec %.2f | q->mass R2 %.3f (null %.3f) lev %.2f%% '
              '| %.0f s'
              % (step, acc['inner_mm'] / max(nacc, 1),
                 acc['mass_mm'] / max(nacc, 1), va['pre'], va['post'],
                 va['gain_pct'], va['gain_dpos_pct'], va['gain_gain_pct'],
                 va['ceil_pct'], va['shuf_pct'], va.get('recovered', float('nan')),
                 va.get('belief/mass_r2', float('nan')),
                 va.get('belief/mass_r2_null', float('nan')),
                 va.get('leverage/at_alpha', float('nan')),
                 time.time() - t0), flush=True)
        wandb.log({('val/%s' % k if '/' not in k else k): v
                   for k, v in va.items() if isinstance(v, (int, float))},
                  step=step)

        ck = {'inner': inner.state_dict(), 'policy': policy.state_dict(),
              'phys_embed': phys_embed.state_dict(),
              'cfg': cfg, 'track_rms': rms, 'val_ids': [int(x) for x in va_ids],
              'paths': eval_paths, 'gstep': step, 'val': va}
        th.save(ck, os.path.join(out_dir, 'last.ckpt'))
        if va['post'] < best:
            best = va['post']
            th.save(ck, os.path.join(out_dir, 'best.ckpt'))
            print('  [*] best val post %.5f (gain %+.5f)' % (best, va['gain']))

    wandb.finish()


if __name__ == '__main__':
    sys.exit(main())
