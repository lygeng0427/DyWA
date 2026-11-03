#!/usr/bin/env python3
"""Second-order meta-training of a latent-conditioned track model.

The model is `LatentTrackModel`: `FricTrackModel` with the privileged mass input
replaced by a free latent `q`. The latent is fitted at test time from the
episode's own ELAPSED motion, and this trainer teaches the model to be good AT
being fitted that way -- the outer loss is differentiated THROUGH the inner
gradient step, so `q0`, the step size, and the trunk all learn to make one step
of TTT worth taking.

    for each meta-batch of B episodes:
        pick a query frame, quantise it to its TTT anchor a = every*floor(t/every)
        support = the m_max elapsed windows at a           (causal: s + H <= a)
        q_K     = adapt(q0, support, K steps)              create_graph=True
        loss    = compute_loss(Q query frames | q_K)       standard objective
        loss.backward()                                    -> every parameter

NO PER-EPISODE LATENT BANK. An episode's latent exists only for the duration of
one forward pass, so meta-training and test-time TTT run the same procedure by
construction. The previous program kept a bank refined over dozens of epochs
while test-time TTT got K steps -- a train/test mismatch with exactly the shape
of the failure it then diagnosed.

HISTORY DROPOUT (`META_DROPOUT`, 0.3 as in the reference) zeroes an episode's
support mask. `object_loss_per_ep` then returns a constant 0 for it, its
gradient is 0, and `q_K == q0` exactly -- so `q0` receives the outer gradient
directly and stays a good BLIND prior. That matters because `t < H` (no window
elapsed) is a real and frequent state, and because the previous program measured
a wrong latent costing more than no latent at all (+0.44 vs +0.15 mm).

`val_loss` IN THE CHECKPOINT IS MILLIMETRES (post-TTT `mv:object`), matching
`train_stream_tracks.py`, NOT the normalised weighted MSE that
`train_fric_track.py` puts under the same key. The in-training validation is a
SUBSET probe for curve-watching; every reported number must come from
`eval_meta_track.py`, because 20-episode probes in this project were 10x wrong
twice.

    META_DATA=... META_VAL_IDS_FROM=... bash dywa/exp/scripts/train_meta_track.sh 8 0
"""
import os

# Must precede any torch import: `expandable_segments:True` trips a
# CUDACachingAllocator assert during double backward (meta_train_ttt.py:30-33).
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import sys
import time

import h5py
import numpy as np
import torch as th
import wandb

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.fric.dataset import (FricTrackDataset, read_grid, bucketize,
                                 episode_mu, count_episodes,
                                 balanced_episode_split, assert_phys_grid)
from models.fric.latent_model import LatentTrackModel

import meta_ttt as M
from train_fric_track import fit_track_norm, PINNED_ALL

PHYS_NAMES = ('mass', 'table_fric', 'obj_fric', 'hand_fric', 'restitution')

# -- data / split -----------------------------------------------------------
DATA = os.environ.get('META_DATA', '')
SAVE_DIR = os.environ.get('META_SAVE_DIR', '/home/user/DyWA/output/mass/ckpt')
VAL_IDS_FROM = os.environ.get('META_VAL_IDS_FROM', '')
VAL_RATIO = float(os.environ.get('META_VAL_RATIO', 0.2))
SPLIT_SEED = int(os.environ.get('META_SPLIT_SEED', 0))
DINO_DIM = int(os.environ.get('META_DINO_DIM', 64))
# -- model ------------------------------------------------------------------
D_MODEL = int(os.environ.get('META_D_MODEL', 512))
ACTION_MODE = os.environ.get('META_ACTION_MODE', 'command')
OBJ_WEIGHT = float(os.environ.get('META_OBJ_WEIGHT', 9.0))
DROPOUT = float(os.environ.get('META_DROPOUT_P', 0.1))
Z_DIM = int(os.environ.get('META_Z_DIM', 8))
Q0_STD = float(os.environ.get('META_Q0_STD', 0.1))
ALPHA0 = float(os.environ.get('META_ALPHA0', 0.1))
# -- TTT --------------------------------------------------------------------
H = int(os.environ.get('META_H', 16))
K_INNER = int(os.environ.get('META_K', 1))
M_MAX = int(os.environ.get('META_M_MAX', 8))
EVERY = int(os.environ.get('META_EVERY', 16))
N_QUERY = int(os.environ.get('META_QUERY', 8))
HIST_DROP = float(os.environ.get('META_DROPOUT', 0.3))
DETACH_Q = os.environ.get('META_DETACH_Q', '0') == '1'   # 1 = FOMAML ablation
# -- optimisation -----------------------------------------------------------
BATCH_EP = int(os.environ.get('META_BATCH_EP', 8))
STEPS = int(os.environ.get('META_STEPS', 200000))
LR = float(os.environ.get('META_LR', 1e-4))
WD = float(os.environ.get('META_WD', 1e-6))
SEED = int(os.environ.get('META_SEED', 0))
WORKERS = int(os.environ.get('META_WORKERS', 6))
DEVICE = os.environ.get('META_DEVICE', 'cuda:0')
VAL_EVERY = int(os.environ.get('META_VAL_EVERY', 2000))
VAL_EPS = int(os.environ.get('META_VAL_EPS', 24))
LOG_EVERY = int(os.environ.get('META_LOG_EVERY', 50))
TAG = os.environ.get('META_TAG', '')
# Warm start from an UNCONDITIONED (`phys_cond='none'`) checkpoint. That model
# is a strict subset of this one: verified, the only keys it lacks are the 12
# FiLM weights/biases plus `q0`/`log_alpha`/`z_mlp.*`, with no shape conflicts
# anywhere. So 85% of the parameters arrive already trained on the full 26.9M
# frame-presentations, and meta-training only has to learn the latent path.
#
# NOT the confound the previous program had to remove: that warm-started from
# `film`, which already knows how to use mass. `none` never saw the mass input,
# so the latent starts genuinely uninformed.
INIT_FROM = os.environ.get('META_INIT_FROM', '')
# FiLM computes h <- (1+g)h + b. The bias inits to zero, so scaling the weight
# down makes the conditioning near-identity and the warm start begins AT the
# source model's error. Not exactly zero: at zero, dL/dq is identically 0 and
# the latent receives no gradient until the FiLM weight moves off it.
FILM_INIT = float(os.environ.get('META_FILM_INIT', 0.01))
WANDB_PROJECT = os.environ.get('META_WANDB_PROJECT', 'dywa-latent-ttt')
WANDB_MODE = os.environ.get('META_WANDB_MODE', 'online')


# ---------------------------------------------------------------------------
# Sampler: one item = one episode's (support, query) group
# ---------------------------------------------------------------------------
class MetaEpisodeSampler(th.utils.data.Dataset):
    """Draws `(support, query)` groups, matching the eval procedure exactly.

    A query frame is drawn uniformly, quantised to its TTT anchor, and the
    support is that anchor's causal elapsed windows. Every frame sharing the
    anchor is a query candidate, and `n_query` of them are drawn -- they share
    one fitted latent, so outer supervision per step is `B * n_query` while the
    inner cost stays `B * m_max`. That amortisation is why `every` can be 16
    without starving the outer loop.

    Episode ranges come from `ds.index`, which is sorted by `(episode, frame)`
    and contiguous per episode, so this needs ONE pass over the index and reads
    no HDF5 at construction. Frame-in-episode is `index[i][2] - index[i][4]`;
    a global row index would silently let a previous episode's window count as
    evidence.
    """

    def __init__(self, ds, h, m_max, every, n_query, length, seed):
        # `ds` is one dataset or a LIST of them (the streaming window is one
        # dataset per shard). Episodes are addressed globally and mapped back to
        # (dataset, local range), so a shard boundary can never be crossed by a
        # support window -- every shard numbers its episodes from 0 and a global
        # row index would let a previous shard's frame count as evidence.
        self.dss = list(ds) if isinstance(ds, (list, tuple)) else [ds]
        self.h, self.m_max = h, m_max
        self.every, self.n_query = every, n_query
        self.length, self.seed = length, seed
        self.t_rels, los, his, which = [], [], [], []
        for si, d_ in enumerate(self.dss):
            idx = d_.index
            ep_of = np.asarray([r[0] for r in idx], np.int64)
            self.t_rels.append(np.asarray([r[2] - r[4] for r in idx], np.int64))
            b = np.nonzero(np.diff(ep_of))[0] + 1
            lo = np.concatenate([[0], b])
            hi = np.concatenate([b, [len(idx)]])
            assert (hi > lo).all(), 'episodes are not contiguous in index'
            los.append(lo)
            his.append(hi)
            which.append(np.full(len(lo), si, np.int64))
        self.lo = np.concatenate(los)
        self.hi = np.concatenate(his)
        self.which = np.concatenate(which)
        self.n_ep = len(self.lo)

    def __len__(self):
        return self.length

    def __getitem__(self, k):
        rng = np.random.RandomState((self.seed * 1000003 + k) % (2 ** 31 - 1))
        e = rng.randint(self.n_ep)
        si = int(self.which[e])
        ds = self.dss[si]
        lo, hi = int(self.lo[e]), int(self.hi[e])
        tr = self.t_rels[si][lo:hi]

        i = rng.randint(len(tr))
        a = M.anchor_row(tr, i, self.every)
        rows = M.support(tr, a, self.h, self.m_max)

        # Query candidates: every frame that quantises to the SAME anchor time.
        # A finer partition than "same anchor row", so it can never group frames
        # the evaluator would have given different latents.
        anc = (tr // max(self.every, 1)) * max(self.every, 1)
        cand = np.nonzero(anc == anc[i])[0]
        qsel = rng.choice(cand, self.n_query,
                          replace=len(cand) < self.n_query)

        # An empty support still needs a row so the batch has static shape; the
        # mask keeps it out of the loss, which makes q_K == q0 for this episode.
        pad = rows if len(rows) else np.zeros(1, np.int64)
        mask = th.zeros(self.m_max, dtype=th.bool)
        mask[:len(rows)] = True
        take = np.concatenate([pad, np.full(self.m_max - len(pad), pad[-1])])

        sup = [ds[lo + int(r)] for r in take]
        qry = [ds[lo + int(r)] for r in qsel]
        keys = M.MODEL_KEYS
        return {'sup': {kk: th.stack([s[kk] for s in sup]) for kk in keys},
                'qry': {kk: th.stack([s[kk] for s in qry]) for kk in keys},
                'sup_mask': mask,
                'n_sup': th.tensor(len(rows), dtype=th.long),
                'phys': qry[0]['phys'],
                'ep': th.tensor(e, dtype=th.long)}


def meta_collate(items):
    out = {}
    for grp in ('sup', 'qry'):
        out[grp] = {k: th.stack([it[grp][k] for it in items])
                    for k in items[0][grp]}
    for k in ('sup_mask', 'n_sup', 'phys', 'ep'):
        out[k] = th.stack([it[k] for it in items])
    return out


def flat(d):
    """[B, m, ...] -> [B*m, ...], episode-major (what object_loss_per_ep wants)."""
    return {k: v.reshape(-1, *v.shape[2:]) for k, v in d.items()}


# ---------------------------------------------------------------------------
def main():
    paths = [p for p in DATA.split(',') if p and os.path.exists(p)]
    assert paths, 'no data files found in META_DATA=%s' % DATA
    th.manual_seed(SEED)
    np.random.seed(SEED)
    dev = th.device(DEVICE)

    g = read_grid(paths)
    col, grid = g['col'], g['grid']
    pinned, _, n_hit = assert_phys_grid(
        paths, {k: v for k, v in PINNED_ALL.items() if k != PHYS_NAMES[col]},
        col, grid)
    print('[phys] swept=%s (col %d, %s) over %d/%d grid values in [%.4f, %.4f]'
          % (g['channel'], col, g['spacing'], n_hit, len(grid), g['lo'], g['hi']))

    # -- split: identical discipline to train_fric_track.py -----------------
    if VAL_IDS_FROM:
        ref = th.load(VAL_IDS_FROM, map_location='cpu')
        va_ids = np.sort(np.asarray([int(x) for x in ref['val_ids']], np.int64))
        n_ep = count_episodes(paths)
        assert va_ids.max() < n_ep, (va_ids.max(), n_ep)
        assert [os.path.realpath(x) for x in ref['paths']] == \
               [os.path.realpath(x) for x in paths[:len(ref['paths'])]], (
            'reference ckpt paths are not a prefix of META_DATA -- id k would '
            'mean a different episode in the two runs')
        tr_ids = np.setdiff1d(np.arange(n_ep, dtype=np.int64), va_ids)
        print('[data] val ids REUSED from %s: %d episodes' % (VAL_IDS_FROM,
                                                              len(va_ids)))
    else:
        tr_ids, va_ids, n_ep, counts, _ = balanced_episode_split(
            paths, col, grid, VAL_RATIO, SPLIT_SEED)
        print('[data] per-bucket split: counts %d..%d'
              % (counts[counts > 0].min(), counts.max()))
    print('[data] %d files | %d episodes -> %d train / %d val'
          % (len(paths), n_ep, len(tr_ids), len(va_ids)))

    ds_tr = FricTrackDataset(paths, phys_col=col, episodes=tr_ids)
    print('[data] train frames %d' % len(ds_tr))

    # The mass corpora were narrowed to 64-d ON DISK (shrink_dino.py). Wrapping
    # in ProjectedDino would be a no-op that also hides `.index`, which the
    # sampler needs -- so assert the widths already match instead.
    widths = set()
    for p in paths:
        with h5py.File(p, 'r', locking=False) as f:
            widths.add(int(f['dino_features'].shape[-1]))
    assert widths == {DINO_DIM}, (
        'stored DINO widths %s != META_DINO_DIM=%d; this trainer needs the '
        'files pre-projected so the sampler can see ds.index'
        % (sorted(widths), DINO_DIM))

    src = None
    if INIT_FROM:
        src = th.load(INIT_FROM, map_location='cpu')
        # `track_rms` must come from whichever checkpoint supplies the WEIGHTS:
        # FieldNorm.scale/offset are persistent buffers that ride inside
        # state_dict, so refitting here would leave the buffer and the recorded
        # scalar disagreeing.
        rms = float(src['track_rms'])
        norm = FieldNorm(3).fit_isotropic_rms(rms)
        print('[norm] track_rms = %.8f FROZEN from the weight source' % rms)
    else:
        norm, rms = fit_track_norm(ds_tr)
        print('[norm] track_rms = %.8f' % rms)

    model = LatentTrackModel(trunk='transformer', d_model=D_MODEL,
                             action_mode=ACTION_MODE, obj_weight=OBJ_WEIGHT,
                             dropout=DROPOUT, dino_dim=DINO_DIM, z_dim=Z_DIM,
                             q0_init_std=Q0_STD, alpha_init=ALPHA0,
                             phys_lo=g['lo'], phys_hi=g['hi'],
                             phys_log=(g['spacing'] == 'log')).to(dev)
    model.normalizer.add('gt_3d_track', norm.to(dev))
    if src is not None:
        miss, unexp = model.load_state_dict(src['model'], strict=False)
        latent = lambda k: (k in ('q0', 'log_alpha')
                            or k.startswith('z_mlp.') or '.film.' in k)
        bad = [k for k in miss if not latent(k)]
        assert not bad, 'warm start is missing non-latent weights: %s' % bad[:8]
        assert not unexp, 'warm start has unexpected keys: %s' % list(unexp)[:8]
        with th.no_grad():
            for nm, pr in model.named_parameters():
                if '.film.' in nm and nm.endswith('weight'):
                    pr.mul_(FILM_INIT)
        print('[init] warm-started from %s (phys_cond=%s, val %.4f)'
              % (INIT_FROM, src['cfg'].get('phys_cond'),
                 src.get('val_loss', float('nan'))))
        print('[init] %d latent-path tensors freshly initialised; '
              'FiLM weights scaled by %g so the run starts at the source model'
              % (len(miss), FILM_INIT))
    print('[model] %d params | z_dim=%d K=%d m_max=%d every=%d Q=%d '
          'detach_q=%d' % (sum(p.numel() for p in model.parameters()), Z_DIM,
                           K_INNER, M_MAX, EVERY, N_QUERY, int(DETACH_Q)))

    sampler = MetaEpisodeSampler(ds_tr, H, M_MAX, EVERY, N_QUERY,
                                 STEPS * BATCH_EP, SEED)
    dl = th.utils.data.DataLoader(
        sampler, batch_size=BATCH_EP, shuffle=False, num_workers=WORKERS,
        collate_fn=meta_collate, pin_memory=True, drop_last=True,
        persistent_workers=WORKERS > 0)

    val_ds = [FricTrackDataset(paths, phys_col=col, episodes=[int(e)])
              for e in va_ids[np.linspace(0, len(va_ids) - 1,
                                          min(VAL_EPS, len(va_ids))
                                          ).astype(int)]]
    print('[data] val probe: %d episodes' % len(val_ds))

    opt = th.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = th.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=STEPS,
                                             pct_start=0.05)

    run = 'meta_z%d_k%d_m%d_e%d_%s' % (Z_DIM, K_INNER, M_MAX, EVERY, ACTION_MODE)
    if DETACH_Q:
        run += '_fomaml'
    if SEED:
        run += '_s%d' % SEED
    run += TAG
    out_dir = os.path.join(SAVE_DIR, run)
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=run, mode=WANDB_MODE, dir=out_dir,
               config={'z_dim': Z_DIM, 'k_inner': K_INNER, 'm_max': M_MAX,
                       'every': EVERY, 'n_query': N_QUERY, 'H': H,
                       'hist_drop': HIST_DROP, 'detach_q': DETACH_Q,
                       'alpha0': ALPHA0, 'q0_std': Q0_STD, 'batch_ep': BATCH_EP,
                       'steps': STEPS, 'lr': LR, 'wd': WD, 'seed': SEED,
                       'd_model': D_MODEL, 'action_mode': ACTION_MODE,
                       'obj_weight': OBJ_WEIGHT, 'dino_dim': DINO_DIM,
                       'track_rms': rms, 'channel': g['channel'],
                       'n_train_ep': len(tr_ids), 'n_val_ep': len(va_ids),
                       'val_ids_from': VAL_IDS_FROM, 'paths': paths,
                       'init_from': INIT_FROM,
                       'film_init': FILM_INIT})
    print('[wandb] %s' % getattr(wandb.run, 'url', '(offline)'))

    cfg = {'d_model': D_MODEL, 'action_mode': ACTION_MODE,
           'obj_weight': OBJ_WEIGHT, 'dropout': DROPOUT, 'dino_dim': DINO_DIM,
           'phys_lo': g['lo'], 'phys_hi': g['hi'],
           'phys_log': (g['spacing'] == 'log'), 'phys_channel': g['channel'],
           'phys_col': col, 'z_dim': Z_DIM, 'q0_init_std': Q0_STD,
           'alpha_init': ALPHA0, 'split_seed': SPLIT_SEED,
           'k_inner': K_INNER, 'm_max': M_MAX, 'every': EVERY, 'H': H,
           'init_from': INIT_FROM,
           'detach_q': DETACH_Q, 'hist_drop': HIST_DROP}

    best = float('inf')
    run_loss = run_inner = 0.0
    nacc = 0
    t0 = time.time()
    it = iter(dl)
    for step in range(STEPS):
        b = next(it)
        sup = flat({k: v.to(dev, non_blocking=True) for k, v in b['sup'].items()})
        qry = flat({k: v.to(dev, non_blocking=True) for k, v in b['qry'].items()})
        mask = b['sup_mask'].to(dev)                       # [B, m_max]
        if HIST_DROP > 0.0:
            drop = th.rand(mask.shape[0], device=dev) < HIST_DROP
            mask = mask & ~drop.unsqueeze(1)
        sup['ep_mask'] = mask

        q = model.q0.unsqueeze(0).expand(mask.shape[0], -1)
        model.eval()                       # inner: no dropout, as at test time
        qk, inner = M.adapt(model, sup, q, K_INNER, create_graph=True,
                            detach_q=DETACH_Q)
        model.train()
        qry['q'] = qk.repeat_interleave(N_QUERY, dim=0)
        loss, info = model.compute_loss(qry)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        run_loss += info['loss']
        run_inner += float(inner.detach())
        nacc += 1

        if (step + 1) % LOG_EVERY == 0:
            with th.no_grad():
                trav = float((qk.detach() - model.q0.detach()).norm(dim=-1).mean())
            wandb.log({'train/loss': run_loss / nacc,
                       'train/inner_mm': 1000.0 * run_inner / nacc,
                       'train/lr': sched.get_last_lr()[0],
                       'z/alpha': float(model.alpha()),
                       'z/travel': trav,
                       'z/q0_norm': float(model.q0.detach().norm()),
                       'data/n_sup': float(b['n_sup'].float().mean()),
                       'perf/step_s': (time.time() - t0) / nacc}, step=step)
            run_loss = run_inner = 0.0
            nacc = 0
            t0 = time.time()

        if (step + 1) % VAL_EVERY == 0 or step + 1 == STEPS:
            eps = [M.cache_episode(d, dev) for d in val_ds]
            pre, post = M.evaluate_metrics(model, eps, H, K_INNER, M_MAX,
                                           EVERY, detach_q=DETACH_Q)
            del eps
            th.cuda.empty_cache()
            print('[%6d] train %.5f | val pre %.3f post %.3f  gain %+.3f mm'
                  % (step + 1, info['loss'], pre, post, pre - post))
            wandb.log({'val/pre_mm': pre, 'val/post_mm': post,
                       'val/gain_mm': pre - post}, step=step)
            ck = {'model': model.state_dict(), 'trunk': 'transformer',
                  'epoch': step + 1, 'gstep': step + 1, 'track_rms': rms,
                  'val_loss': post, 'val_pre': pre, 'cfg': cfg,
                  'train_ids': tr_ids, 'val_ids': va_ids, 'paths': paths}
            th.save(ck, os.path.join(out_dir, 'last.ckpt'))
            if post < best:
                best = post
                th.save(ck, os.path.join(out_dir, 'best.ckpt'))
                print('        [*] best (post %.3f mm)' % best)

    wandb.finish()
    print('[done] %s  best post %.3f mm' % (out_dir, best))


if __name__ == '__main__':
    main()
