#!/usr/bin/env python3
"""Second-order meta-training of a latent track model on STREAMED shards.

Same meta-learning as `train_meta_track.py` -- one function, `meta_ttt`, does the
inner update at training and at test time -- but the training episodes come from
a FIFO window of shards a collector is writing continuously, instead of a frozen
corpus. Bounded disk, unlimited unique episodes.

WHY. The offline arms landed at 15.1 mm against a `none` floor of 11.96 and a
`film` ceiling of 11.18. That gap is fully accounted for by supervision, not by
method: 200k steps x 32 query frames is 6.4M frame-presentations against the
baseline's 26.9M, and 11.96 x 4.2^0.20 = 15.9 predicted. Most of each meta step's
memory goes to support frames retained through a double backward, so the only
ways to close it are more steps or more unique data. This does the second, and
warm-starts from the offline arm so the first 200k steps are not repeated.

NO DAgger, and that is not an omission: a track model is a passive predictor, it
never acts, so the teacher alone fixes the data distribution and frame reuse from
a window is legitimate. What is wanted is streaming data, not online RL.

`ShardWindow`, `lr_at` and `infinite` are imported from `train_stream_tracks.py`
rather than reimplemented -- a divergent copy of the shard admission guards
(`.done` markers, per-shard `read_grid`/`assert_phys_grid`, the deletion
whitelist) is exactly the kind of drift that trains fine and poisons the result.
Only `_build` is overridden, and only because `ProjectedDino` hides `.index`,
which the meta sampler needs to find episode boundaries.

    META_INIT_FROM=<offline best.ckpt> bash dywa/exp/scripts/train_meta_stream.sh 0
"""
import os

# Must precede any torch import (meta_train_ttt.py:30-33).
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import gc
import math
import sys
import time

import h5py
import numpy as np
import torch as th
import wandb

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.fric.dataset import (FricTrackDataset, read_grid, count_episodes,
                                 assert_phys_grid)
from models.fric.latent_model import LatentTrackModel

import meta_ttt as M
import train_stream_tracks as S
from train_meta_track import MetaEpisodeSampler, meta_collate, flat, PHYS_NAMES
from train_fric_track import PINNED_ALL

# -- what to stream ---------------------------------------------------------
EVAL_DATA = os.environ.get('META_EVAL_DATA', '')
VAL_IDS_FROM = os.environ.get('META_VAL_IDS_FROM', '')
INIT_FROM = os.environ.get('META_INIT_FROM', '')
SAVE_DIR = os.environ.get('META_SAVE_DIR', '/home/user/DyWA/output/mass/ckpt')
DINO_DIM = int(os.environ.get('META_DINO_DIM', 64))
# -- model / TTT (must match the warm-start checkpoint) ---------------------
D_MODEL = int(os.environ.get('META_D_MODEL', 512))
ACTION_MODE = os.environ.get('META_ACTION_MODE', 'command')
OBJ_WEIGHT = float(os.environ.get('META_OBJ_WEIGHT', 9.0))
DROPOUT = float(os.environ.get('META_DROPOUT_P', 0.1))
Z_DIM = int(os.environ.get('META_Z_DIM', 8))
Q0_STD = float(os.environ.get('META_Q0_STD', 0.1))
ALPHA0 = float(os.environ.get('META_ALPHA0', 0.1))
# META_ALPHA_SET overwrites the inner step size AFTER the warm-start load (the
# checkpoint's log_alpha would otherwise win, since it rides in state_dict).
# META_ALPHA_FREEZE keeps it there by excluding it from the optimizer.
#
# Both exist because meta-training drives alpha DOWN as the trunk improves --
# 0.1035 -> 0.0946 -> 0.0750 -> 0.0544 across the arms -- while the measured
# optimum is ~5x the trained value. The cause is dL/dalpha ~ 0 on the flat
# surface near q0: the better the trunk, the flatter that region, so the outer
# loss has no signal with which to grow the step.
ALPHA_SET = float(os.environ.get('META_ALPHA_SET', 0.0))
ALPHA_FREEZE = os.environ.get('META_ALPHA_FREEZE', '0') == '1'
H = int(os.environ.get('META_H', 16))
K_INNER = int(os.environ.get('META_K', 1))
M_MAX = int(os.environ.get('META_M_MAX', 8))
EVERY = int(os.environ.get('META_EVERY', 16))
N_QUERY = int(os.environ.get('META_QUERY', 8))
HIST_DROP = float(os.environ.get('META_DROPOUT', 0.3))
DETACH_Q = os.environ.get('META_DETACH_Q', '0') == '1'
# -- optimisation -----------------------------------------------------------
BATCH_EP = int(os.environ.get('META_BATCH_EP', 4))
TOTAL_STEPS = int(os.environ.get('META_TOTAL_STEPS', 400000))
BLOCK_STEPS = int(os.environ.get('META_BLOCK_STEPS', 5000))
LR = float(os.environ.get('META_LR', 3e-5))
WARMUP = int(os.environ.get('META_WARMUP', 2000))
ANNEAL = int(os.environ.get('META_ANNEAL', 60000))
# Global step of the checkpoint this warm-starts from. Purely cosmetic for the
# optimiser (`lr_at` uses the LOCAL step, so the schedule belongs to THIS run)
# but it makes the WandB x-axis continue where the previous round stopped
# instead of restarting at 0 and overlaying two unrelated curves.
STEP0 = int(os.environ.get('META_STEP0', 0))
WD = float(os.environ.get('META_WD', 1e-6))
SEED = int(os.environ.get('META_SEED', 0))
WORKERS = int(os.environ.get('META_WORKERS', 6))
DEVICE = os.environ.get('META_DEVICE', 'cuda:0')
VAL_EPS = int(os.environ.get('META_VAL_EPS', 40))
LOG_EVERY = int(os.environ.get('META_LOG_EVERY', 50))
TAG = os.environ.get('META_TAG', '')
WANDB_PROJECT = os.environ.get('META_WANDB_PROJECT', 'dywa-latent-ttt')
WANDB_MODE = os.environ.get('META_WANDB_MODE', 'online')


class MetaShardWindow(S.ShardWindow):
    """`ShardWindow` whose datasets keep `.index` visible.

    The base class wraps every shard in `ProjectedDino` whenever
    `STREAM_DINO_DIM != 1024`. That is a no-op on files already written at 64-d
    (all of ours), but the wrapper defines no `__getattr__`, so it hides
    `.index` -- and the meta sampler needs it to find episode boundaries without
    reading any HDF5. Assert the stored width instead of projecting it; every
    other admission guard is inherited untouched.
    """

    def _build(self, path):
        g = read_grid([path] + self.eval_paths)
        assert g['col'] == self.col, (g['col'], self.col)
        assert_phys_grid([path], self.expect_pinned, self.col, self.grid)
        with h5py.File(path, 'r', locking=False) as f:
            w = int(f['dino_features'].shape[-1])
        assert w == DINO_DIM, (
            '%s stores %d-d DINO features but META_DINO_DIM=%d; this trainer '
            'needs them pre-projected so the sampler can see ds.index'
            % (path, w, DINO_DIM))
        return FricTrackDataset([path], phys_col=self.col)


def lr_at(step):
    """Warmup, constant, then a terminal cosine anneal.

    The anneal is NOT optional dressing. A constant LR has an equilibrium error
    floor -- SGD noise at 3e-5 keeps the model a fixed distance above the
    minimum -- and the checkpoint this warm-starts from reached 15.1 mm only
    because OneCycle annealed it to ~0. Without a terminal anneal this run can
    accumulate unlimited unique data and still never beat its own starting
    point, which would read as "streaming does not help" when it actually means
    "the LR schedule never cashed the data in".

    Still not OneCycleLR: that needs `total_steps` up front and raises past it,
    which makes a run un-extendable and a resume depend on scheduler state.
    """
    if step < WARMUP:
        return LR * float(step + 1) / max(WARMUP, 1)
    if ANNEAL and step >= TOTAL_STEPS - ANNEAL:
        p = min(1.0, (step - (TOTAL_STEPS - ANNEAL)) / float(ANNEAL))
        return LR * 0.5 * (1.0 + math.cos(math.pi * p))
    return LR


def main():
    assert EVAL_DATA, 'META_EVAL_DATA is required (the frozen eval corpus)'
    assert VAL_IDS_FROM, 'META_VAL_IDS_FROM is required'
    th.manual_seed(SEED)
    np.random.seed(SEED)
    dev = th.device(DEVICE)
    eval_paths = [p for p in EVAL_DATA.split(',') if p and os.path.exists(p)]
    assert eval_paths, EVAL_DATA

    g = read_grid(eval_paths)
    col, grid = g['col'], g['grid']
    expect_pinned = {k: v for k, v in PINNED_ALL.items() if k != PHYS_NAMES[col]}
    print('[phys] swept=%s (col %d, %s) grid %.3f..%.3f'
          % (g['channel'], col, g['spacing'], g['lo'], g['hi']))

    # -- val: the SAME 419 episodes as every other number in this project.
    # They live in the frozen corpus and are never streamed, so train/val stay
    # disjoint by construction rather than by bookkeeping.
    ref = th.load(VAL_IDS_FROM, map_location='cpu')
    va_ids = np.sort(np.asarray([int(x) for x in ref['val_ids']], np.int64))
    n_ep_eval = count_episodes(eval_paths)
    assert va_ids.max() < n_ep_eval, (va_ids.max(), n_ep_eval)
    assert [os.path.realpath(x) for x in ref['paths']] == \
           [os.path.realpath(x) for x in eval_paths[:len(ref['paths'])]], (
        'reference ckpt paths are not a prefix of META_EVAL_DATA')
    print('[data] val ids REUSED from %s: %d episodes'
          % (VAL_IDS_FROM, len(va_ids)))

    # -- model, warm-started -------------------------------------------------
    assert INIT_FROM and os.path.exists(INIT_FROM), (
        'META_INIT_FROM must point at the offline meta checkpoint')
    src = th.load(INIT_FROM, map_location='cpu')
    c = src['cfg']
    # META_ALLOW_SHAPE_CHANGE names keys that MAY differ from the warm-start
    # checkpoint. Only the TTT-procedure knobs are waiveable: k_inner, m_max and
    # every change what the inner loop does, not what any parameter means, so a
    # deliberate change is legitimate (e.g. K=1 -> 3 after measuring that one
    # inner step is ~2.4x short). z_dim / H / action_mode / dino_dim can never
    # be waived -- those would load weights that mean something else.
    WAIVEABLE = {'k_inner', 'm_max', 'every'}
    allow = {x for x in os.environ.get('META_ALLOW_SHAPE_CHANGE', '').split(',')
             if x}
    assert allow <= WAIVEABLE, ('not waiveable: %s' % (allow - WAIVEABLE))
    for k, want in (('z_dim', Z_DIM), ('k_inner', K_INNER), ('m_max', M_MAX),
                    ('every', EVERY), ('H', H), ('action_mode', ACTION_MODE),
                    ('dino_dim', DINO_DIM)):
        if c[k] == want:
            continue
        assert k in allow, (
            'warm-start checkpoint has %s=%r but this run is configured with %r'
            ' (set META_ALLOW_SHAPE_CHANGE=%s if deliberate)' % (k, c[k], want, k))
        print('[init] DELIBERATE procedure change: %s %r -> %r' % (k, c[k], want))
    # `track_rms` comes from whichever checkpoint supplies the WEIGHTS.
    # FieldNorm.scale/offset are PERSISTENT buffers, so they ride inside
    # state_dict and load_state_dict silently overwrites anything set here --
    # taking the scalar from the val-ids reference instead would leave the
    # buffer and the recorded number disagreeing.
    rms = float(src['track_rms'])
    model = LatentTrackModel(trunk='transformer', d_model=D_MODEL,
                             action_mode=ACTION_MODE, obj_weight=OBJ_WEIGHT,
                             dropout=DROPOUT, dino_dim=DINO_DIM, z_dim=Z_DIM,
                             q0_init_std=Q0_STD, alpha_init=ALPHA0,
                             phys_lo=g['lo'], phys_hi=g['hi'],
                             phys_log=(g['spacing'] == 'log')).to(dev)
    model.normalizer.add('gt_3d_track',
                         FieldNorm(3).fit_isotropic_rms(rms).to(dev))
    model.load_state_dict(src['model'], strict=True)
    if ALPHA_SET > 0.0:
        with th.no_grad():
            model.log_alpha.fill_(math.log(math.expm1(ALPHA_SET)))
        print('[init] alpha OVERRIDDEN to %.4f%s'
              % (float(model.alpha()), ' and FROZEN' if ALPHA_FREEZE else ''))
    live = float(model.normalizer['gt_3d_track'].scale.reshape(-1)[0])
    assert abs(live - 1.0 / rms) < 1e-6, (live, 1.0 / rms)
    print('[init] warm-started from %s (step %s, val %.3f mm) | alpha=%.4f'
          % (INIT_FROM, src.get('gstep', src.get('epoch')),
             src.get('val_loss', float('nan')), float(model.alpha())))
    print('[norm] track_rms = %.8f FROZEN from the weight source' % rms)

    # -- the streaming window ------------------------------------------------
    win = MetaShardWindow(S.STREAM_DIR, S.WINDOW_SHARDS, col, grid,
                          expect_pinned, eval_paths,
                          static=[p for p in S.STATIC.split(',') if p] or None)
    win.wait_for_min()
    win.refresh()
    print('[stream] %d shards resident, %d frames'
          % (len(win.resident), win.n_frames))

    val_ds = [FricTrackDataset(eval_paths, phys_col=col, episodes=[int(e)])
              for e in va_ids[np.linspace(0, len(va_ids) - 1,
                                          min(VAL_EPS, len(va_ids))
                                          ).astype(int)]]

    params = [p_ for n_, p_ in model.named_parameters()
              if not (ALPHA_FREEZE and n_ == 'log_alpha')]
    assert not ALPHA_FREEZE or len(params) == len(list(model.parameters())) - 1
    opt = th.optim.AdamW(params, lr=LR, weight_decay=WD)
    run = 'meta_z%d_stream%s' % (Z_DIM, TAG)
    out_dir = os.path.join(SAVE_DIR, run)
    os.makedirs(out_dir, exist_ok=True)
    wandb.init(project=WANDB_PROJECT, name=run, mode=WANDB_MODE, dir=out_dir,
               config={'z_dim': Z_DIM, 'k_inner': K_INNER, 'm_max': M_MAX,
                       'every': EVERY, 'n_query': N_QUERY, 'H': H,
                       'hist_drop': HIST_DROP, 'detach_q': DETACH_Q,
                       'batch_ep': BATCH_EP, 'total_steps': TOTAL_STEPS,
                       'block_steps': BLOCK_STEPS, 'lr': LR, 'warmup': WARMUP, 'anneal': ANNEAL,
                       'wd': WD, 'seed': SEED, 'track_rms': rms,
                       'alpha_set': ALPHA_SET, 'alpha_freeze': ALPHA_FREEZE,
                       'stream_dir': S.STREAM_DIR, 'init_from': INIT_FROM,
                       'window_shards': S.WINDOW_SHARDS,
                       'val_ids_from': VAL_IDS_FROM, 'streaming': True,
                       'step0': STEP0})
    print('[wandb] %s' % getattr(wandb.run, 'url', '(offline)'))

    cfg = dict(c)
    # The TTT-procedure knobs must be THIS run's, not the warm-start source's.
    # `dict(c)` copies the source cfg wholesale, so a run that deliberately
    # changed k_inner/m_max/every (via META_ALLOW_SHAPE_CHANGE) used to save a
    # checkpoint that LIED about its own procedure -- and `eval_meta_track.py`
    # reads K and m_max straight out of cfg, so the model would be evaluated
    # with the wrong inner loop and nothing would raise.
    cfg.update({'streaming': True, 'stream_dir': S.STREAM_DIR,
                'init_from': INIT_FROM, 'lr': LR,
                'k_inner': K_INNER, 'm_max': M_MAX, 'every': EVERY})

    best = float('inf')
    step = 0
    consumed = 0
    t_block = time.time()
    while step < TOTAL_STEPS:
        # The DataLoader's workers hold open h5py handles, so it MUST be gone
        # before refresh(), which may delete evicted shards. `gc.collect()`
        # makes the teardown deterministic -- a lazily collected loader shows up
        # later as a bare `Traceback (most recent call last):` with no body.
        added, evicted = win.refresh()
        gc.collect()
        dss = [d for _, _, d in win.resident]
        n_res = sum(len(d) for d in dss)
        sampler = MetaEpisodeSampler(dss, H, M_MAX, EVERY, N_QUERY,
                                     BLOCK_STEPS * BATCH_EP, SEED + step)
        dl = th.utils.data.DataLoader(
            sampler, batch_size=BATCH_EP, shuffle=False, num_workers=WORKERS,
            collate_fn=meta_collate, pin_memory=True, drop_last=True,
            persistent_workers=False)
        print('[stream] block @%d: %d shards (+%d/-%d), %d frames, %d episodes'
              % (STEP0 + step, len(win.resident), added, evicted, n_res,
                 sampler.n_ep))

        run_loss = run_inner = 0.0
        nacc = 0
        model.train()
        for b in dl:
            if step >= TOTAL_STEPS:
                break
            for gparam in opt.param_groups:
                gparam['lr'] = lr_at(step)
            sup = flat({k: v.to(dev, non_blocking=True)
                        for k, v in b['sup'].items()})
            qry = flat({k: v.to(dev, non_blocking=True)
                        for k, v in b['qry'].items()})
            mask = b['sup_mask'].to(dev)
            if HIST_DROP > 0.0:
                drop = th.rand(mask.shape[0], device=dev) < HIST_DROP
                mask = mask & ~drop.unsqueeze(1)
            sup['ep_mask'] = mask

            q = model.q0.unsqueeze(0).expand(mask.shape[0], -1)
            model.eval()
            qk, inner = M.adapt(model, sup, q, K_INNER, create_graph=True,
                                detach_q=DETACH_Q)
            model.train()
            qry['q'] = qk.repeat_interleave(N_QUERY, dim=0)
            loss, info = model.compute_loss(qry)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            th.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            run_loss += info['loss']
            run_inner += float(inner.detach())
            nacc += 1
            step += 1
            consumed += mask.shape[0] * (M_MAX + N_QUERY)
            if nacc % LOG_EVERY == 0:
                wandb.log({'train/loss': run_loss / nacc,
                           'train/inner_mm': 1000.0 * run_inner / nacc,
                           'train/lr': lr_at(step),
                           'z/alpha': float(model.alpha()),
                           'z/q0_norm': float(model.q0.detach().norm()),
                           'stream/shards': len(win.resident),
                           'stream/frames_resident': n_res,
                           'stream/unique_admitted': win.n_frames_admitted,
                           'stream/reuse': consumed /
                           max(win.n_frames_admitted, 1)}, step=STEP0 + step)
                run_loss = run_inner = 0.0
                nacc = 0

        del dl, sampler
        gc.collect()
        th.cuda.empty_cache()

        eps = [M.cache_episode(d, dev) for d in val_ds]
        pre, post = M.evaluate_metrics(model, eps, H, K_INNER, M_MAX, EVERY,
                                       detach_q=DETACH_Q)
        del eps
        th.cuda.empty_cache()
        reuse = consumed / max(win.n_frames_admitted, 1)
        print('[%7d] val pre %.3f post %.3f  gain %+.3f mm | unique %d frames, '
              'reuse %.2fx | %.0f s/block'
              % (STEP0 + step, pre, post, pre - post, win.n_frames_admitted,
                 reuse, time.time() - t_block))
        t_block = time.time()
        wandb.log({'val/pre_mm': pre, 'val/post_mm': post,
                   'val/gain_mm': pre - post}, step=STEP0 + step)
        ck = {'model': model.state_dict(), 'trunk': 'transformer',
              'epoch': STEP0 + step, 'gstep': STEP0 + step,
              'track_rms': rms,
              'val_loss': post, 'val_pre': pre, 'cfg': cfg,
              'val_ids': va_ids, 'paths': eval_paths}
        th.save(ck, os.path.join(out_dir, 'last.ckpt'))
        if post < best:
            best = post
            th.save(ck, os.path.join(out_dir, 'best.ckpt'))
            print('        [*] best (post %.3f mm)' % best)

    wandb.finish()
    print('[done] %s  best post %.3f mm' % (out_dir, best))


if __name__ == '__main__':
    main()
