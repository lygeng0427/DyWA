#!/usr/bin/env python3
"""Shared core for meta-learned TTT with a POLICY as the outer model.

Two models, no shared weights. The inner one is
`models.fric.LatentTrackModel` -- a track Transformer whose object-only
prediction error on already-elapsed windows defines a gradient in the belief.
The outer one is `models.policy.JointPolicyDiT`, which consumes that belief and
denoises the action chunk and the scene tracks together. The belief is the only
thing that passes between them.

Everything about the inner half is `meta_ttt`, imported and not reimplemented:
the causal support rule, the unit-normalised step, `adapt`. This module adds
only the two things that change when the outer loss stops being a track loss:

  `policy_err`           per-frame ACTION error at a PINNED noise draw
  `walk_episode_policy`  causal pre/post over one episode

ONE FUNCTION, TWO CALLERS, still. `walk_episode_policy` is what the trainer's
validation and `eval_meta_policy.py` both call, so a train/test divergence in
the adaptation procedure would require editing one function rather than
agreeing across two call sites. That is the discipline `meta_ttt`'s docstring
exists to enforce and the exact failure that cost the previous program.

THE NOISE MUST BE PINNED, and it is not bookkeeping. `JointPolicyDiT.compute_loss`
draws `(t, eps)`, so the same frames scored twice differ by the draw. The effect
being measured here is `pre - post`, which has been ~1% in every arm of this
program, so an unpinned draw would be larger than the signal. `policy_err`
therefore keys the seed to `(caller seed, chunk offset)` and `walk_episode_policy`
keys it to the ANCHOR, so the pre and the post of a given group see the identical
noise and the difference is attributable to `q` alone.

READ THE ACTION TERM. The total loss is the action MSE plus a track term that is
~87% arm, and the arm's track is the action's own consequence -- predicting it
from an action the model is simultaneously emitting is close to a lookup. This
project has inverted a conclusion twice by reading an aggregate that was ~90%
arm, so `policy_err` returns the ACTION term only and the track term is carried
beside it for context.
"""
from typing import Dict, List, Tuple

import numpy as np
import torch as th

import meta_ttt as M


def assert_compatible(inner, policy):
    """The belief is the only interface, so its width must agree exactly.

    A mismatch would be caught by `cond_vec`'s shape assert on the first forward
    anyway, but failing here names the two models instead of a tensor shape.
    """
    assert policy.cond == 'latent', (
        'the outer model must be cond=latent; got %r' % policy.cond)
    assert policy.cond_dim == inner.z_dim, (
        'policy cond_dim=%d but inner z_dim=%d' % (policy.cond_dim, inner.z_dim))
    assert policy.H == inner.H, (policy.H, inner.H)


@th.no_grad()
def policy_err(policy, batch: Dict[str, th.Tensor], q: th.Tensor, seed: int,
               chunk: int = 8) -> Tuple[float, float, int, float, float]:
    """`(action, track, n, dpos, gains)` sums under belief `q`.

    The dpos/gains split is not cosmetic. Measured on this corpus, the teacher
    handles mass IN its DR purely by impedance: conditioning on true mass buys
    +0.59% on the 14 joint KP/KD dims and -0.07% on the 3 commanded dpos dims.
    A single action number would let a gains-only effect read as "the belief
    helps the policy move better", which it does not.

    Sums rather than means so a caller can pool across groups and episodes
    without weighting by hand. `compute_loss`'s action term is already a mean
    over the batch, so `mean * n` is the exact frame sum.
    """
    n = batch['agent_pos'].shape[0]
    a_tot = t_tot = d_tot = g_tot = 0.0
    for s in range(0, n, chunk):
        b = {k: v[s:s + chunk] for k, v in batch.items()}
        nb = b['agent_pos'].shape[0]
        b['q'] = q.reshape(1, -1).expand(nb, -1)
        _, info = policy.compute_loss(b, noise_seed=seed + s,
                                      per_group=True)
        a_tot += info['action'] * nb
        t_tot += info['track'] * nb
        d_tot += info['a_dpos'] * nb
        g_tot += info['a_gain'] * nb
    return a_tot, t_tot, n, d_tot, g_tot


def walk_episode_policy(inner, policy, data, t_rel, scorable, H, q0,
                        k_steps: int, m_max: int, every: int,
                        seed: int = 0, collect: bool = False):
    """Causal TTT across one episode, scored on the POLICY.

    Mirrors `meta_ttt.walk_episode` exactly -- frames are grouped by anchor, the
    belief is fitted ONCE per group from that anchor's elapsed windows, and
    every frame in the group is scored with it. Groups whose support is empty
    (`t < H`) are scored at `q0` in BOTH columns, so pre and post agree there to
    the digit. That is a built-in causality check, not an approximation.

    Returns `(pre_action, post_action, pre_track, post_track, n, qs)`.
    """
    rows_all = np.nonzero(scorable.cpu().numpy())[0]
    groups: Dict[int, List[int]] = {}
    for i in rows_all:
        groups.setdefault(M.anchor_row(t_rel, int(i), every), []).append(int(i))

    pre_a = post_a = pre_t = post_t = 0.0
    pre_d = post_d = pre_g = post_g = 0.0
    n = 0
    qs = []
    for a, idx in groups.items():
        rows = M.support(t_rel, a, H, m_max)
        qb = M.take(data, np.asarray(idx, np.int64))
        # Keyed to the ANCHOR, so the pre and post of this group see identical
        # noise and the difference is `q` and nothing else.
        sd = int(seed) + int(a) * 1009
        a0, t0, n0, d0, g0 = policy_err(policy, qb, q0, sd)
        pre_a += a0
        pre_t += t0
        pre_d += d0
        pre_g += g0
        n += n0
        if len(rows) == 0:
            post_a += a0
            post_t += t0
            post_d += d0
            post_g += g0
            continue
        sup = M.pack_support([M.take(data, rows)], len(rows))
        with th.enable_grad():
            qk, _ = M.adapt(inner, sup, q0.clone(), k_steps, create_graph=False)
        a1, t1, _, d1, g1 = policy_err(policy, qb, qk, sd)
        post_a += a1
        post_t += t1
        post_d += d1
        post_g += g1
        if collect:
            qs.append((a, qk[0].detach().cpu().numpy()))
    return (pre_a, post_a, pre_t, post_t, n, qs, pre_d, post_d, pre_g, post_g)


def _oof_r2(X, y, lam=1.0, folds=5):
    """Out-of-fold ridge R2. In-sample R2 with a 32-d regressor on ~32 points is
    near 1 by construction and means nothing."""
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    n = len(y)
    idx = np.arange(n)
    pred = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        if len(tr) < 3 or len(te) == 0:
            continue
        A = np.c_[X[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[tr])
        pred[te] = np.c_[X[te], np.ones(len(te))] @ w
    return 1.0 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


def evaluate_policy(inner, policy, episodes, H, k_steps, m_max, every,
                    seed: int = 0, ceil_steps: int = 40):
    """Four columns on one episode set, plus the action split.

      pre   the belief stays at the learned prior q0 -- no adaptation.
      post  CAUSAL TTT. Before the first window elapses it is bit-identical to
            pre, which is the built-in causality check.
      ceil  the belief fitted NON-CAUSALLY on the whole episode. Never a
            result -- it is the upper bound on what any causal fit could reach,
            and `post/ceil` is the only honest statement of how much of the
            available signal the causal procedure actually gets. r3 reached 65%.
      shuf  the PREVIOUS episode's belief. The machinery runs identically but
            the belief is about the wrong trajectory, so a model genuinely
            using it must be clearly WORSE here than pre. The first episode is
            DROPPED from every column rather than defaulted, because defaulting
            puts a causal number in the shuffled slot -- contamination in the
            direction that makes a belief-ignoring model look fine.
    """
    inner.eval()
    policy.eval()
    q0 = inner.q0.detach().unsqueeze(0)
    A = {k: 0.0 for k in ('pre', 'post', 'ceil', 'shuf',
                          'pre_d', 'post_d', 'pre_g', 'post_g', 'pre_t', 'post_t')}
    n = 0
    prev = None
    for data, t_rel, sc in episodes:
        if not bool(sc.any()):
            continue
        pa, qa, pt, qt, nn_, qs, pd, qd, pg, qg = walk_episode_policy(
            inner, policy, data, t_rel, sc, H, q0, k_steps, m_max, every,
            seed=seed, collect=True)
        last = qs[-1][1] if qs else None
        if prev is None:
            prev = last
            continue                      # no belief to borrow yet: drop it
        rows = np.nonzero(sc.cpu().numpy())[0]
        groups: Dict[int, List[int]] = {}
        for i in rows:
            groups.setdefault(M.anchor_row(t_rel, int(i), every), []).append(int(i))
        sel = rows[np.unique(np.linspace(0, len(rows) - 1,
                                         min(16, len(rows))).astype(int))]
        sup = M.pack_support([M.take(data, sel)], len(sel))
        with th.enable_grad():
            qc, _ = M.adapt(inner, sup, q0.clone(), ceil_steps, create_graph=False)
        qp = th.as_tensor(prev, device=q0.device).reshape(1, -1)
        ca = sa = 0.0
        for a_, idx in groups.items():
            qb = M.take(data, np.asarray(idx, np.int64))
            sd = int(seed) + int(a_) * 1009
            ca += policy_err(policy, qb, qc, sd)[0]
            sa += policy_err(policy, qb, qp, sd)[0]
        for k, v in (('pre', pa), ('post', qa), ('ceil', ca), ('shuf', sa),
                     ('pre_d', pd), ('post_d', qd), ('pre_g', pg),
                     ('post_g', qg), ('pre_t', pt), ('post_t', qt)):
            A[k] += v
        n += nn_
        prev = last
    inner.train()
    policy.train()
    d = max(n, 1)
    o = {k: v / d for k, v in A.items()}
    o['gain'] = o['pre'] - o['post']
    o['gain_pct'] = 100.0 * o['gain'] / max(o['pre'], 1e-12)
    o['ceil_pct'] = 100.0 * (o['pre'] - o['ceil']) / max(o['pre'], 1e-12)
    o['shuf_pct'] = 100.0 * (o['pre'] - o['shuf']) / max(o['pre'], 1e-12)
    # How much of the AVAILABLE signal the causal fit gets. A bare gain of
    # +0.3% is uninterpretable; "+0.3% of an available +1.0%" is not.
    o['recovered'] = (o['gain'] / (o['pre'] - o['ceil'])
                      if o['pre'] - o['ceil'] > 1e-9 else float('nan'))
    o['gain_dpos_pct'] = 100.0 * (o['pre_d'] - o['post_d']) / max(o['pre_d'], 1e-12)
    o['gain_gain_pct'] = 100.0 * (o['pre_g'] - o['post_g']) / max(o['pre_g'], 1e-12)
    o['n'] = n
    return o


def diagnostics(inner, policy, episodes, H, k_steps, m_max, every,
                phys_embed=None, phys_norm=None, seed: int = 0, n_dir: int = 3):
    """The metrics that would have caught every failure this pipeline has had.

      belief/mass_r2        does the CAUSAL belief carry the hidden variable?
                            Against its own shuffled null, because chance R2
                            with a 32-d regressor on ~30 episodes is NOT 0.
                            This is the metric whose absence cost four blocks.
      belief/travel         |q_K - q0|; must equal K*alpha, lower only where
                            support was empty.
      belief/dir_spread     spread of the displacement DIRECTIONS. All of the
                            information is in direction (the step is
                            unit-normalised), so ~0 means every episode got the
                            same belief, i.e. collapse.
      leverage/at_alpha     relative change in the policy's action loss for a
                            random perturbation of size K*alpha. If this is
                            under the block-to-block noise of val, the reported
                            gain is arithmetic on noise no matter how long the
                            run goes.
      leverage/gamma        mean FiLM gamma scale. A TRAINED, working path on
                            this corpus sits at 0.25-0.43.
      inner/mass_gain       does the inner model's own loss improve when
                            conditioned on the TRUE mass instead of q0? If ~0
                            its conditioning slot is inert and the belief
                            cannot mean anything, whatever the policy does.
    """
    inner.eval()
    policy.eval()
    dev = inner.q0.device
    q0 = inner.q0.detach().unsqueeze(0)
    out = {}
    Q, mass, sups, qry = [], [], [], []
    for data, t_rel, sc in episodes:
        if not bool(sc.any()):
            continue
        rows = np.nonzero(sc.cpu().numpy())[0]
        anc = M.anchor_row(t_rel, int(rows[-1]), every)
        r = M.support(t_rel, anc, H, m_max)
        if len(r) == 0:
            continue
        sup = M.pack_support([M.take(data, r)], len(r))
        with th.enable_grad():
            qk, _ = M.adapt(inner, sup, q0.clone(), k_steps, create_graph=False)
        Q.append(qk[0].detach().cpu().numpy())
        mass.append(float(data['phys'][0, 0]))
        if len(sups) < 6:
            sups.append(sup)
            qry.append(M.take(data, rows[:16]))
    if len(Q) >= 10:
        Qa = np.asarray(Q)
        d = Qa - q0.cpu().numpy()
        y = np.log(np.asarray(mass))
        rng = np.random.default_rng(0)
        out['belief/mass_r2'] = _oof_r2(d, y)
        out['belief/mass_r2_null'] = float(np.mean(
            [_oof_r2(d, rng.permutation(y)) for _ in range(10)]))
        out['belief/travel'] = float(np.linalg.norm(d, axis=1).mean())
        out['belief/dir_spread'] = float(np.std(
            d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)))

    def perr(q):
        t = nn_ = 0.0
        for i, b in enumerate(qry):
            e, _, nb, _, _ = policy_err(policy, b, q, 7000 + i)
            t += e
            nn_ += nb
        return t / max(nn_, 1)

    if qry:
        base = perr(q0)
        mag = float(inner.alpha()) * k_steps
        th.manual_seed(seed)
        rel = []
        for _ in range(n_dir):
            u = th.randn_like(q0)
            u = u / u.norm()
            for sg in (1.0, -1.0):
                rel.append(abs(perr(q0 + sg * mag * u) - base) / max(base, 1e-12))
        out['leverage/at_alpha'] = 100.0 * float(np.mean(rel))
    g = [float((w @ q0.reshape(-1)).std())
         for nm, w in policy.named_parameters() if nm.endswith('film.weight')]
    if g:
        out['leverage/gamma'] = float(np.mean(g))

    if sups and phys_embed is not None and phys_norm is not None:
        with th.enable_grad():
            b0 = float(np.mean([float(M.object_loss_per_ep(inner, sp, q0).mean())
                                for sp in sups]))
            bt = []
            for sp, (data, _, _) in zip(sups, episodes[:len(sups)]):
                qt = phys_embed(phys_norm(data['phys'][:1].reshape(1, 1)))
                bt.append(float(M.object_loss_per_ep(inner, sp, qt).mean()))
            bt = float(np.mean(bt))
        out['inner/mass_gain'] = 100.0 * (b0 - bt) / max(b0, 1e-12)
        out['inner/mm_at_q0'] = 1000.0 * b0
    inner.train()
    policy.train()
    return out
