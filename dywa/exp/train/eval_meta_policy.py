#!/usr/bin/env python3
"""Evaluate a `train_meta_policy.py` checkpoint on the pinned val episodes.

    METAP_CKPT=.../best.ckpt bash dywa/exp/scripts/eval_meta_policy.sh [GPU]

Four columns, because a single post-adaptation number cannot be interpreted on
its own:

  blind   the belief stays at the learned prior q0. No adaptation at all.
  ttt     CAUSAL: at the anchor the belief has descended the inner loss only on
          windows with `s + H <= a` -- what a robot could have known. Before the
          first window elapses this is bit-identical to `blind`, which is the
          built-in causality check rather than an approximation.
  ceil    the belief fitted on the WHOLE episode, future included. Not a result
          -- it is the upper bound on what any causal fit could reach, and the
          ratio ttt/ceil is the only honest statement of how much of the
          available signal the causal procedure actually gets.
  shuf    the PREVIOUS episode's belief. The adaptation machinery runs
          identically but the belief is about the wrong trajectory. A model
          genuinely using the belief is clearly WORSE here than blind; one
          ignoring it lands on blind, and then nothing else in the arm means
          anything.

The adaptation itself is `meta_policy.walk_episode_policy`, the same function
the trainer validates with, so this script cannot drift from the procedure that
was trained. It reads K / m_max / every from the checkpoint's own cfg -- which
is why `train_meta_policy.py` writes THIS run's values there rather than
copying a warm-start source's.

THE ACTION TERM IS THE RESULT. The track term is carried beside it for context
and is ~87% arm, where the arm's track is the action's own consequence.
"""
import os
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import numpy as np
import torch as th

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.fric.dataset import FricTrackDataset, read_grid
from models.fric.latent_model import LatentTrackModel
from models.policy.joint_dit import JointPolicyDiT

import meta_ttt as M
import meta_policy as MP

CKPT = os.environ.get('METAP_CKPT', '')
EVAL = os.environ.get('METAP_EVAL_DATA', '')
DEVICE = os.environ.get('METAP_DEVICE', 'cuda:0')
N_EP = int(os.environ.get('METAP_EVAL_EPS', 0))        # 0 -> all
K_EVAL = os.environ.get('METAP_K_EVAL', '')            # '' -> the ckpt's K
M_EVAL = int(os.environ.get('METAP_M_MAX_EVAL', 0))    # 0 -> the ckpt's m_max
VAL_SEED = int(os.environ.get('METAP_VAL_SEED', 9000))
CEIL_STEPS = int(os.environ.get('METAP_CEIL_STEPS', 60))
OUT = os.environ.get('METAP_EVAL_OUT', '')


def build(dev):
    """-> (inner, policy, ckpt, cfg). Importable, so probes score the same
    model the eval does."""
    assert CKPT and os.path.exists(CKPT), 'METAP_CKPT=%r' % CKPT
    ck = th.load(CKPT, map_location='cpu')
    c = ck['cfg']
    rms = float(ck['track_rms'])
    inner = LatentTrackModel(
        trunk='transformer', d_model=c['d_model'], action_mode=c['action_mode'],
        obj_weight=c['obj_weight'], dropout=0.0, dino_dim=c['dino_dim'],
        z_dim=c['z_dim'], direct_film=c.get('direct_film', False),
        film_init_scale=c.get('film_init_scale', 1.0),
        q0_init_std=c['q0_init_std'], alpha_init=c['alpha_init'],
        phys_lo=c['phys_lo'], phys_hi=c['phys_hi'], phys_log=c['phys_log'])
    inner.normalizer.add('gt_3d_track', FieldNorm(3).fit_isotropic_rms(rms))
    inner.load_state_dict(ck['inner'], strict=True)

    policy = JointPolicyDiT(
        dino_dim=c['dino_dim'], d_model=c['d_model'], dropout=0.0,
        obj_weight=c['obj_weight'], track_loss_weight=c['lam'],
        cond='latent', cond_dim=c['z_dim'],
        film_init_scale=c.get('film_init_scale', 1.0),
        phys_lo=c['phys_lo'], phys_hi=c['phys_hi'], phys_log=c['phys_log'])
    policy.normalizer.add('gt_3d_track', FieldNorm(3).fit_isotropic_rms(rms))
    policy.normalizer.add('action', FieldNorm(policy.action_dim))
    policy.load_state_dict(ck['policy'], strict=True)
    return inner.to(dev).eval(), policy.to(dev).eval(), ck, c


def ceiling_q(inner, data, sc, q0, n_rows=16, steps=60):
    """Non-causal fit over the whole episode -- the upper bound, never a result."""
    rows = np.nonzero(sc.cpu().numpy())[0]
    sel = rows[np.unique(np.linspace(0, len(rows) - 1,
                                     min(n_rows, len(rows))).astype(int))]
    sup = M.pack_support([M.take(data, sel)], len(sel))
    with th.enable_grad():
        qf, _ = M.adapt(inner, sup, q0.clone(), steps, create_graph=False)
    return qf.detach()


def main():
    dev = th.device(DEVICE)
    inner, policy, ck, c = build(dev)
    MP.assert_compatible(inner, policy)
    H, m_max = c['H'], (M_EVAL or c['m_max'])
    every = c['every']
    ks = [int(x) for x in K_EVAL.split(',') if x] or [int(c['k_inner'])]
    paths = [p for p in EVAL.split(',') if p and os.path.exists(p)] \
        or [p for p in ck['paths'] if os.path.exists(p)]
    g = read_grid(paths)
    print('[ckpt] %s  step %s' % (CKPT, ck.get('gstep')))
    print('[cfg ] z_dim=%d K=%s m_max=%d every=%d  (trained K=%d m_max=%d)'
          % (c['z_dim'], ks, m_max, every, c['k_inner'], c['m_max']))

    ids = [int(x) for x in ck['val_ids']]
    if N_EP:
        ids = [ids[i] for i in np.linspace(0, len(ids) - 1, N_EP).astype(int)]
    q0 = inner.q0.detach().unsqueeze(0)

    cols = ['blind'] + ['ttt_k%d' % k for k in ks] + ['ceil', 'shuf']
    acc = {c_: 0.0 for c_ in cols}
    acc_t = {c_: 0.0 for c_ in cols}
    n = 0
    per_ep = {c_: [] for c_ in cols}
    prev_q = None
    done = 0
    for e in ids:
        ds = FricTrackDataset(paths, phys_col=c['phys_col'], episodes=[int(e)])
        data, t_rel, sc = M.cache_episode(ds, dev)
        if not bool(sc.any()):
            del data
            continue
        ep = {}
        for k in ks:
            pa, qa, pt, qt, nn_, qs, _pd, _qd, _pg, _qg = MP.walk_episode_policy(
                inner, policy, data, t_rel, sc, H, q0, k, m_max, every,
                seed=VAL_SEED, collect=True)
            if k == ks[0]:
                n += nn_
                ep['blind'] = pa / nn_
                ep['_pa'], ep['_pt'] = pa, pt
                last_q = qs[-1][1] if qs else None
            ep['ttt_k%d' % k] = qa / nn_
            ep['_qa%d' % k], ep['_qt%d' % k] = qa, qt

        # ceiling and shuffled share the blind frame population
        rows_all = np.nonzero(sc.cpu().numpy())[0]
        groups = {}
        for i in rows_all:
            groups.setdefault(M.anchor_row(t_rel, int(i), every), []).append(int(i))
        if prev_q is None:
            # No previous belief to borrow yet: `shuf` would fall back to q0,
            # i.e. to BLIND, which is exactly the contamination that makes a
            # belief-ignoring model pass this control. Drop the episode from
            # every column instead, so all four stay on one frame population.
            prev_q = last_q
            n -= nn_
            del data
            th.cuda.empty_cache()
            continue
        qc = ceiling_q(inner, data, sc, q0, steps=CEIL_STEPS)
        # `collect=True` hands back numpy, so lift it before use.
        qs_ = th.as_tensor(prev_q, device=dev).reshape(1, -1)
        ca = sa = ct = st = 0.0
        for a, idx in groups.items():
            qb = M.take(data, np.asarray(idx, np.int64))
            sd = VAL_SEED + int(a) * 1009
            x, xt, _, _, _ = MP.policy_err(policy, qb, qc, sd)
            y, yt, _, _, _ = MP.policy_err(policy, qb, qs_, sd)
            ca += x; ct += xt; sa += y; st += yt
        acc['blind'] += ep['_pa']; acc_t['blind'] += ep['_pt']
        for k in ks:
            acc['ttt_k%d' % k] += ep['_qa%d' % k]
            acc_t['ttt_k%d' % k] += ep['_qt%d' % k]
        acc['ceil'] += ca; acc_t['ceil'] += ct
        acc['shuf'] += sa; acc_t['shuf'] += st
        ep['ceil'] = ca / nn_
        ep['shuf'] = sa / nn_
        for c_ in cols:
            per_ep[c_].append(ep[c_])
        prev_q = last_q
        del data
        th.cuda.empty_cache()
        done += 1
        if done % 25 == 0:
            print('  .. %d/%d episodes' % (done, len(ids)), flush=True)

    print('\n=== action MSE (normalised command units), %d frames, %d episodes ==='
          % (n, done))
    base = acc['blind'] / max(n, 1)
    print('%-10s %10s %10s %10s' % ('column', 'action', 'vs blind', 'track'))
    for c_ in cols:
        v = acc[c_] / max(n, 1)
        print('%-10s %10.5f %+10.5f %10.5f'
              % (c_, v, base - v, acc_t[c_] / max(n, 1)))

    bl = np.asarray(per_ep['blind'])
    for c_ in cols[1:]:
        d = bl - np.asarray(per_ep[c_])
        print('per-episode gain %-8s %+8.5f +- %.5f   win %d/%d'
              % (c_, d.mean(), d.std(ddof=1) / np.sqrt(len(d)),
                 int((d > 0).sum()), len(d)))

    if OUT:
        np.savez(OUT, **{('ep_%s' % k): np.asarray(v)
                         for k, v in per_ep.items()}, ids=np.asarray(ids[:done]))
        print('\n[saved] %s' % OUT)


if __name__ == '__main__':
    sys.exit(main())
