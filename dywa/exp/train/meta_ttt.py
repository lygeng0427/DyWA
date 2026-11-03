#!/usr/bin/env python3
"""Second-order meta-learned test-time training of a latent, for track models.

ONE FUNCTION, THREE CALLERS. `ttt_inner_update` is written once and called by
meta-training with `create_graph=True`, and by `evaluate_metrics` / the eval
script with `create_graph=False` -- exactly as the pick-bar reference's trainer,
`evaluate_metrics` and `ttt_adapt` do. Train-vs-test agreement is then checkable
by reading one function instead of diffing two call sites, which is the failure
mode that cost the previous program: it adapted on `compute_loss` at training
and on an object-only loss at evaluation, and nothing raised.

THE CAUSALITY RULE, and it is the whole correctness story. At query frame `t`, a
window that starts at `s` may be used as evidence iff `s + H <= t` -- only then
has its H-step track already fully elapsed and been observed. `support()` derives
its own window list from `t`; no caller hands one in, because a caller that
computed it slightly wrong would leak the future with nothing raising. Two
further rules that are easy to violate without noticing:

  * The subsample must not depend on the episode's total length `T`. `T` is
    future information at time `t`.
  * It must be deterministic given `(episode, t)`, or results are not
    reproducible across runs.

THE INNER LOSS IS OBJECT-ONLY AND IN RAW METRES, deliberately not
`TrackModel.compute_loss`. That applies `patch_weights` with `obj_weight=9`,
which equalises object and arm at roughly 50/50, and it is MSE in NORMALISED
units. The arm's track is nearly determined by the commanded action the model is
already given, so including it roughly doubles the noise exactly at low support
-- which is where all of the remaining headroom sits.

An inner/outer loss DIFFERENCE is not a defect here. Under MAML it is what the
outer loop trains the model to exploit -- the reference's inner is a latent MSE
while its outer is a GMM NLL. What must match is inner-at-train vs inner-at-test,
and that is guaranteed structurally by the one-function rule above.

THE PER-EPISODE DENOMINATOR is what makes the inner loss the same function a
single-episode TTT computes -- `TrackModel.compute_loss` and the older
`ttt_common.object_loss` divide by a batch-global `w.sum()`, so under them an
episode's loss depends on which others shared its batch. It is also the only
version that stays correct if the step ever stops being unit-normalised, and the
only one whose logged value is a real per-episode millimetre number.

But do NOT claim it is what makes batched meta-training equal single-episode
TTT. It is not -- `unit_step` is. A batch-global denominator is a positive
constant w.r.t. `q`, so normalising the gradient cancels it exactly and `q_k` is
unchanged. Measured (`_probe_meta_batcheq.py`): batched vs single-episode `q_k`
agree to 1.05e-5, which is float32 gradient noise (1.3e-7) amplified by the
1/|g| in the normaliser (min nonzero |dL/dq| = 8.2e-4). What DOES break the
equivalence is misaligning which episode's latent conditions which frames --
`q.repeat(m, 1)` instead of `q.repeat_interleave(m, 0)` -- and that moves `q_k`
by 7.5e-2, 7105x larger. That is the bug to guard against, and the probe's
negative control is written for it.

Nothing here imports `ttt_common`; `support`/`cache_episode` are copied from it
(pure data plumbing, no design commitments) so this pipeline is self-contained.
`eval_fric_track.subsets` IS imported -- it is the canonical `mv:object`
definition and every number reported must be on that population.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from eval_fric_track import subsets

# Keys a cached episode carries. `dino_features` dominates the footprint; the
# tracks dominate for long episodes (284 frames x 391 x 16 x 3 x 4B = 213 MB),
# which is still comfortable on a 24 GB card and removes h5py from the loop.
CACHE_KEYS = ('dino_features', 'patch_centers', 'agent_pos', 'action_seq',
              'action_cmd_seq', 'owner', 'gt_3d_track', 'track_valid_mask',
              'phys')

# Keys a model forward actually reads. `phys` is carried for diagnostics (the
# mass-decode probe, the same-mass swap control) but LatentTrackModel never
# reads it -- that is the point of the whole exercise.
MODEL_KEYS = tuple(k for k in CACHE_KEYS if k != 'phys')


# ---------------------------------------------------------------------------
# 1. Episode caching and the causal support rule
# ---------------------------------------------------------------------------
def cache_episode(ds, dev):
    """Load every frame of a single-episode dataset onto `dev` once.

    Returns `(data, t_rel, scorable)`:
      data      dict of [T, ...] tensors
      t_rel     [T] frame index WITHIN the episode -- what `s + H <= t` needs.
                A global row index would silently let a window from a previous
                episode count as evidence.
      scorable  [T] bool, frames with at least one MOVING object patch. This is
                `eval_fric_track.subsets`'s `mv:object`, so the population is
                identical to every number this is compared against.
    """
    base = ds.ds if hasattr(ds, 'ds') else ds
    t_rel = np.asarray([t - st for _, _, t, _, st in base.index], np.int64)
    assert np.all(np.diff(t_rel) > 0), 'frames not in increasing order'
    items = [ds[i] for i in range(len(ds))]
    data = {k: th.stack([it[k] for it in items]).to(dev) for k in CACHE_KEYS}
    m = subsets({k: data[k] for k in ('gt_3d_track', 'track_valid_mask', 'owner')})
    return data, t_rel, m['mv:object'].any(-1)


def support(t_rel, i, H, m_max, scorable=None):
    """Row indices of the elapsed, usable windows for query row `i`.

    Causal by construction (`t_rel[s] + H <= t_rel[i]`). Subsampled EVENLY over
    the available prefix when it exceeds `m_max` (so the oldest and the most
    recent are always kept), which is deterministic given `(episode, t)` and
    independent of the episode's future length.

    `scorable=None` -- NO pre-filter on "did the object move in this window" --
    is the setting both the trainer and the evaluator use, and they must agree.
    The filter is attractive (a static window carries no dynamics signal) but
    computing it needs `gt_3d_track`/`track_valid_mask`/`owner` for the WHOLE
    episode, while the training sampler loads only the ~16 frames it is about to
    use. Precomputing it across 1.6M frames means reading the tracks, which is
    ~120 GB of HDF5. Using it at eval and not at training would be precisely the
    train/test asymmetry this module exists to prevent.

    Dropping it is close to free and degrades gracefully. A static window
    contributes 0 to BOTH the numerator and the denominator of
    `object_loss_per_ep`, so it is a wasted support slot, not noise -- and only
    ~6% of indexed frames are static. When EVERY support window is static the
    loss is identically 0, its gradient is 0, and `q_k == q0`: no evidence
    yields the prior, which is the right answer and is reached automatically.
    """
    if scorable is None:
        avail = np.nonzero(t_rel + H <= t_rel[i])[0]
    else:
        ok = np.nonzero(scorable.cpu().numpy())[0] if th.is_tensor(scorable) \
            else np.nonzero(scorable)[0]
        avail = ok[t_rel[ok] + H <= t_rel[i]]
    if len(avail) <= m_max:
        return avail
    return avail[np.unique(np.linspace(0, len(avail) - 1,
                                       m_max).astype(np.int64))]


def anchor_row(t_rel, i, every):
    """Row index of the TTT anchor for query row `i`.

    Test time cannot afford to refit the latent at every frame -- episodes here
    run 19-301 frames (mean 124), against the reference's 1-13 attempts, and
    consecutive frames differ by one window out of dozens. So the latent is
    refitted every `every` frames and the one in force at `t` was fitted at
    `a = every * floor(t / every)`.

    Meta-training calls this too, so the cadence is part of the procedure being
    meta-learned rather than a test-time approximation of it. `every <= 1`
    recovers per-frame refitting.

    Strictly more causal than querying at `i`: the anchor is at or before `i`,
    so `t_rel[s] + H <= t_rel[a] <= t_rel[i]` still holds.
    """
    if every <= 1:
        return int(i)
    a = (t_rel[i] // every) * every
    j = int(np.searchsorted(t_rel, a, side='right')) - 1
    return max(j, 0)


def take(data, rows):
    return {k: v[rows] for k, v in data.items() if k in MODEL_KEYS}


def pack_support(per_ep: Sequence[Dict[str, th.Tensor]], m_max: int):
    """Stack per-episode support frames into one episode-major batch.

    Returns `(sup, ep_mask)` where `sup` has `n_ep * m_max` rows, row
    `e * m_max + j` belonging to episode `e`, and `ep_mask` is `[n_ep, m_max]`.
    Short episodes are padded by repeating their first row; the mask zeroes
    those out of the loss.

    Padding rather than ragged batching keeps the reduction a plain reshape --
    no `index_add`, no segment ids -- which is what makes `object_loss_per_ep`
    trivially safe under double backward.
    """
    n_ep = len(per_ep)
    mask = th.zeros(n_ep, m_max, dtype=th.bool)
    rows: List[Dict[str, th.Tensor]] = []
    for e, d in enumerate(per_ep):
        m = len(next(iter(d.values())))
        assert 0 < m <= m_max, (m, m_max)
        mask[e, :m] = True
        idx = th.arange(m_max).clamp(max=m - 1)
        rows.append({k: v[idx] for k, v in d.items()})
    sup = {k: th.cat([r[k] for r in rows], 0) for k in rows[0]}
    dev = next(iter(sup.values())).device
    sup['ep_mask'] = mask.to(dev)
    return sup


# ---------------------------------------------------------------------------
# 2. The inner loss and the inner update
# ---------------------------------------------------------------------------
def object_loss_per_ep(model, sup, q):
    """[n_ep] mean per-patch-step L2 over MOVING OBJECT patches, in RAW METRES.

    `sup` is episode-major with `sup['ep_mask']` of shape `[n_ep, m_max]`; `q`
    is `[n_ep, z_dim]` and is broadcast to that episode's own support rows, so
    episode `e`'s loss depends on `q[e]` and nothing else. That is what lets a
    single `autograd.grad(losses.sum(), q)` return every episode's own gradient
    in one backward.

    `predict()` cannot be used -- it is `@th.no_grad()` -- so the trunk is called
    directly and the prediction un-normalised by hand, exactly as it would be.
    """
    assert model.trunk == 'transformer', (
        'the DiT trunk needs a noised track and a timestep, so this would '
        'silently score garbage')
    n_ep, m = sup['ep_mask'].shape
    b = {k: v for k, v in sup.items() if k != 'ep_mask'}
    b['q'] = q.repeat_interleave(m, dim=0)

    mem = model.encode_context(b)
    pred = model._decode_transformer(mem, b)
    if 'gt_3d_track' in model.normalizer:
        pred = model.normalizer['gt_3d_track'].unnormalize(pred)

    gt = b['gt_3d_track']                                   # [n_ep*m, N, H, 3]
    keep = subsets(b)['mv:object'] & sup['ep_mask'].reshape(-1, 1)
    w = keep.unsqueeze(-1).to(gt.dtype)                     # [n_ep*m, N, 1]
    err = (pred - gt).norm(dim=-1)                          # [n_ep*m, N, H]

    num = (err * w).reshape(n_ep, -1).sum(-1)
    den = w.expand_as(err).reshape(n_ep, -1).sum(-1)
    return num / den.clamp(min=1.0)


def ttt_inner_update(q, model, sup, create_graph=True, detach_q=False):
    """One inner step on `q` using the object-only track loss over the support.

      per_ep     = object_loss_per_ep(model, sup, q)        # [n_ep], metres
      q_k        = q - softplus(log_alpha) * unit(dL/dq)

    `detach_q` controls how `q0` receives the outer meta-gradient, matching the
    reference's `DETACH_Q`:
      * False (default): the inner loss is differentiated w.r.t. `q` directly,
        so `q0` receives full second-order MAML curvature.
      * True (FOMAML): the inner gradient is taken against a detached copy, so
        `q0` is updated only through the linear `q_k = q - alpha*g` term.

    THE GRADIENT IS ALWAYS NORMALISED, which is the one deliberate deviation
    from the reference. The reference caps only when `|g| > 1`, preserving
    magnitude when the gradient is weak. Measured here `|dL/dq| ~ 1.7e-5`, so
    that cap never fires: `q_k == q0` to seven figures, the outer loss sees no
    difference between adapting and not, and there is therefore no meta-signal
    that could ever grow the gradient -- a deadlock, and the exact bug that made
    the previous program's adaptation phase inert. A scale-free direction plus a
    learned step size (`model.alpha()`) breaks it and makes one knob mean the
    same thing at init, at convergence, and at test time.

    CONSEQUENCE OF NORMALISING, and it constrains the choice of `z_dim`:
    `|q_k - q0| == alpha` EXACTLY, for every episode and every width (measured:
    1.000e-01 at alpha=0.1). All of the information therefore lives in the
    DIRECTION, i.e. on the unit sphere S^(z_dim-1), giving `z_dim - 1` continuous
    degrees of freedom per inner step. So:

      z_dim = 1  ->  S^0 = {-1, +1}. The step is q0 +- alpha and nothing else:
                     a ONE-BIT belief, not a line search. `g/(|g|+eps)` for a
                     scalar is just `sign(g)`, and the magnitude -- the only
                     thing that could encode "how heavy" -- is exactly what the
                     normaliser discards.
      z_dim = 2  ->  S^1, one continuous DOF. The MINIMAL width at which the
                     belief is a continuum under this update rule.

    A 1-D latent is therefore not the clean line-search control it looks like.
    If a genuinely 1-D continuous belief is wanted, the step must keep the
    gradient's magnitude -- divide by a LEARNED scale instead of the per-sample
    norm -- which is a different (and untested here) rule.

    Returns `(q_k, inner_loss)`; `inner_loss` is the mean over episodes, kept as
    a tensor so a caller with `create_graph=False` can still backward it as an
    auxiliary objective.
    """
    q_diff = q.detach().requires_grad_(True) if detach_q else q
    per_ep = object_loss_per_ep(model, sup, q_diff)
    inner = per_ep.sum()
    g = th.autograd.grad(inner, q_diff, create_graph=create_graph)[0]
    g = th.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    g = g / (g.norm(dim=-1, keepdim=True) + 1e-12)
    q_k = q - model.alpha() * g
    return q_k, per_ep.mean()


def adapt(model, sup, q, k_steps, create_graph=False, detach_q=False):
    """`k_steps` applications of `ttt_inner_update`. `k_steps <= 0` returns `q`.

    With `create_graph=False` each step starts from a fresh leaf and the result
    is detached, so no graph is carried across steps -- inference needs no meta
    gradient, and `model.alpha()` would otherwise keep one alive. This mirrors
    the reference's `ttt_adapt` exactly.
    """
    last = None
    for _ in range(max(int(k_steps), 0)):
        if not create_graph:
            q = q.detach().requires_grad_(True)
        q, last = ttt_inner_update(q, model, sup, create_graph=create_graph,
                                   detach_q=detach_q)
        if not create_graph:
            q = th.nan_to_num(q.detach(), nan=0.0)
    return q, last


# ---------------------------------------------------------------------------
# 3. Scoring
# ---------------------------------------------------------------------------
@th.no_grad()
def score_per_frame(model, batch, q, chunk=16):
    """Per-frame `(err_sum_mm, n_patch_steps)` over moving object patches.

    Returned per frame rather than pooled so the evaluator can bucket by elapsed
    -window count and by frame index without re-running the model. Pooling a
    ratio of sums is NOT the same as averaging per-frame ratios, so the caller
    sums both columns and divides once.

    `q` is `[1, z_dim]` (broadcast) or `[B, z_dim]`.
    """
    B = batch['patch_centers'].shape[0]
    es, ns = [], []
    for s in range(0, B, chunk):
        b = {k: v[s:s + chunk] for k, v in batch.items() if k in MODEL_KEYS}
        nb = b['patch_centers'].shape[0]
        b['q'] = q.expand(nb, -1) if q.shape[0] == 1 else q[s:s + chunk]
        pred = model.predict(b)
        keep = subsets(b)['mv:object']
        w = keep.unsqueeze(-1).expand_as(b['gt_3d_track'][..., 0]).to(pred.dtype)
        err = (pred - b['gt_3d_track']).norm(dim=-1)
        es.append((err * w).sum(dim=(1, 2)) * 1000.0)
        ns.append(w.sum(dim=(1, 2)))
    return th.cat(es).cpu().numpy(), th.cat(ns).cpu().numpy()


def score(model, batch, q, chunk=16):
    """(err_sum_mm, n_patch_steps) pooled over the batch."""
    e, n = score_per_frame(model, batch, q, chunk)
    return float(e.sum()), int(n.sum())


def walk_episode(model, data, t_rel, scorable, H, q0, k_steps, m_max, every,
                 detach_q=False, collect=False):
    """Causal TTT across one episode; returns `(err_pre, err_post, n, info)`.

    Frames are grouped by their anchor (`anchor_row`), the latent is fitted ONCE
    per anchor group and every frame in the group is scored with it. That is the
    cadence of §4 of the design, and it is also what makes evaluation affordable:
    a 124-frame episode has ~8 anchor groups at `every=16`, not 124 fits.

    Frames whose support is empty (`t < H`) are scored at `q0` in BOTH columns,
    so `pre` and `post` agree there to the digit -- a built-in causality check
    rather than an approximation.
    """
    rows_all = np.nonzero(scorable.cpu().numpy())[0]
    groups: Dict[int, List[int]] = {}
    for i in rows_all:
        groups.setdefault(anchor_row(t_rel, int(i), every), []).append(int(i))

    pre = post = 0.0
    n = 0
    qs: List[Tuple[int, np.ndarray]] = []
    for a, idx in groups.items():
        rows = support(t_rel, a, H, m_max)
        qb = take(data, np.asarray(idx, np.int64))
        e0, n0 = score(model, qb, q0)
        pre += e0
        n += n0
        if len(rows) == 0:
            post += e0                       # nothing elapsed: identical by rule
            continue
        sup = pack_support([take(data, rows)], len(rows))
        with th.enable_grad():
            qk, _ = adapt(model, sup, q0.clone(), k_steps, create_graph=False,
                          detach_q=detach_q)
        e1, _ = score(model, qb, qk)
        post += e1
        if collect:
            qs.append((a, qk[0].detach().cpu().numpy()))
    return pre, post, n, qs


def evaluate_metrics(model, episodes, H, k_steps, m_max, every,
                     detach_q=False):
    """Reference-shaped `(pre, post)` in mm, over a list of cached episodes.

    `episodes` is a list of `(data, t_rel, scorable)` from `cache_episode`.
    The success criterion is `pre - post > 0`, measured WITHIN one model on
    identical frames -- which is why it is the primary metric here. A
    between-model comparison on this data carries a ~0.13 mm seed spread, the
    same size as the effects being measured.
    """
    was = model.training
    model.eval()
    sp = sq = 0.0
    n = 0
    q0 = model.q0.detach().unsqueeze(0)
    for data, t_rel, scorable in episodes:
        a, b, k, _ = walk_episode(model, data, t_rel, scorable, H, q0,
                                  k_steps, m_max, every, detach_q=detach_q)
        sp += a
        sq += b
        n += k
    model.train(was)
    d = max(n, 1)
    return sp / d, sq / d


# ---------------------------------------------------------------------------
# 4. Self-checks
# ---------------------------------------------------------------------------
def assert_causal(t_rel, H, m_max, every, n_check=64):
    """`support(anchor(i))` never reaches at or past `t_rel[i]`, and is empty
    before the first window can have elapsed."""
    rows = np.linspace(0, len(t_rel) - 1,
                       min(n_check, len(t_rel))).astype(np.int64)
    for i in rows:
        a = anchor_row(t_rel, int(i), every)
        assert t_rel[a] <= t_rel[i], (a, i)
        s = support(t_rel, a, H, m_max)
        if len(s):
            assert t_rel[s].max() + H <= t_rel[i], (t_rel[s].max(), t_rel[i], H)
        if t_rel[i] < H:
            assert len(s) == 0, (int(i), len(s))
