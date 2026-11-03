#!/usr/bin/env python3
"""Held-out evaluation of a meta-trained latent track model, in mm.

    META_CKPT=.../best.ckpt python3 eval_meta_track.py [--n-ep 0] [--every 1,8,16,32,64]

Columns, all on IDENTICAL frames of the same model, so no between-model seed
term enters (that term is ~0.13 mm here, the same size as the effects):

  blind   the learned prior q0, no adaptation                       the FLOOR
  e<N>    causal TTT with the latent refitted every N frames        the METHOD
  once    causal TTT fitted ONCE at the first anchor with full support, then
          frozen; frames BEFORE that anchor are scored at q0, or the column
          leaks future evidence backwards
  ceil    NON-CAUSAL fit on windows spanning the whole episode, big budget
          -- an upper bound on what any latent of this width could deliver
  shuf    the PREVIOUS episode's fitted latent -- if this is not clearly worse
          than blind the trunk is IGNORING the latent and nothing else here means
          anything

`e<N>` is a sweep AND a diagnostic. The latent is refitted every N frames, so
larger N means a staler belief. Flat in N => the latent is a persistent physics
belief. Degrading sharply => it is a time-local correction, which is the failure
the previous program diagnosed; `once` is the extreme of that axis.

THE INNER UPDATE IS `meta_ttt.ttt_inner_update`, the same function meta-training
calls, with `create_graph=False`. It is not reimplemented here and its step size
comes from the checkpoint's own learned `log_alpha` -- so train and test cannot
drift apart by argument.
"""
import os
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import argparse

import numpy as np
import torch as th

sys.path.insert(0, '/home/user/DyWA/dywa/src')

from models.gap.common import FieldNorm
from models.fric.dataset import FricTrackDataset
from models.fric.latent_model import LatentTrackModel

import meta_ttt as M

CKPT = os.environ.get('META_CKPT', '')
DEVICE = os.environ.get('META_DEVICE', 'cuda:0')
EVAL = os.environ.get('META_EVAL', ','.join([
    '/home/user/DyWA/output/mass/tracks_m_a.hdf5',
    '/home/user/DyWA/output/mass/tracks_m_b.hdf5',
    '/home/user/DyWA/output/mass/tracks_m64_0.hdf5',
    '/home/user/DyWA/output/mass/tracks_m64_1.hdf5',
    '/home/user/DyWA/output/mass/tracks_m64_b0.hdf5',
    '/home/user/DyWA/output/mass/tracks_m64_b1.hdf5']))

NSUP_BINS = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 10 ** 9)]
T_BINS = [(0, 16), (16, 32), (32, 64), (64, 128), (128, 10 ** 9)]


def build(dev):
    assert CKPT, 'META_CKPT is required'
    ck = th.load(CKPT, map_location='cpu')
    c = ck['cfg']
    m = LatentTrackModel(
        trunk=ck['trunk'], d_model=c['d_model'], action_mode=c['action_mode'],
        obj_weight=c['obj_weight'], dropout=c.get('dropout', 0.1),
        dino_dim=c['dino_dim'], z_dim=c['z_dim'],
        q0_init_std=c['q0_init_std'], alpha_init=c.get('alpha_init', 0.1),
        phys_lo=c['phys_lo'], phys_hi=c['phys_hi'],
        phys_log=c['phys_log']).to(dev)
    # Register the normalizer BEFORE load_state_dict: FieldNorm.scale/offset are
    # PERSISTENT buffers, so they ride inside state_dict and would otherwise be
    # reported as unexpected keys and silently dropped.
    m.normalizer.add('gt_3d_track',
                     FieldNorm(3).fit_isotropic_rms(ck['track_rms']).to(dev))
    m.load_state_dict(ck['model'], strict=True)
    # META_ALPHA_EVAL overrides the inner step size at TEST time. alpha is a
    # single scalar that scales a unit-norm direction, so changing it is a pure
    # test-time knob -- no retraining, no shape change. r3b showed a frozen
    # alpha=0.27 triples the gain; this lets that be applied to a DIFFERENT
    # trunk, separating "better step size" from "worse trunk".
    _ae = float(os.environ.get('META_ALPHA_EVAL', 0.0))
    if _ae > 0.0:
        import math as _m
        with th.no_grad():
            m.log_alpha.fill_(_m.log(_m.expm1(_ae)))
        print('[cfg ] alpha OVERRIDDEN for eval: %.4f' % float(m.alpha()))
    m.eval()
    return m, ck, c


def fit_noncausal(model, data, t_rel, q0, steps, m_cap):
    """The ceiling: fit on windows spanning the WHOLE episode.

    Deliberately not causal -- it answers "how good could this latent be if the
    episode were already over", which bounds every causal scheme from above.
    Uses the same `ttt_inner_update`, just with more steps and better evidence.
    """
    rows = np.unique(np.linspace(0, max(len(t_rel) - 1 - 0, 0),
                                 m_cap).astype(np.int64))
    rows = rows[t_rel[rows] + 0 <= t_rel[-1]]
    sup = M.pack_support([M.take(data, rows)], len(rows))
    with th.enable_grad():
        q, _ = M.adapt(model, sup, q0.clone(), steps, create_graph=False)
    return q


def walk(model, data, t_rel, scorable, q0, H, k, m_max, every):
    """Causal TTT at cadence `every`; per-frame errors for the scorable frames.

    Frames are grouped by anchor and the latent is fitted ONCE per group, which
    is both the cadence being tested and what makes this affordable: a
    124-frame episode has ~8 groups at every=16, not 124 fits.
    """
    rows = np.nonzero(scorable.cpu().numpy())[0]
    groups = {}
    for i in rows:
        groups.setdefault(M.anchor_row(t_rel, int(i), every), []).append(int(i))
    err = np.zeros(len(rows))
    cnt = np.zeros(len(rows))
    nsup = np.zeros(len(rows), np.int64)
    pos = {int(v): j for j, v in enumerate(rows)}
    for a, idx in groups.items():
        sr = M.support(t_rel, a, H, m_max)
        qb = M.take(data, np.asarray(idx, np.int64))
        if len(sr) == 0:
            q = q0
        else:
            sup = M.pack_support([M.take(data, sr)], len(sr))
            with th.enable_grad():
                q, _ = M.adapt(model, sup, q0.clone(), k, create_graph=False)
        e, n = M.score_per_frame(model, qb, q)
        for j, i in enumerate(idx):
            p = pos[i]
            err[p], cnt[p], nsup[p] = e[j], n[j], len(sr)
    return err, cnt, nsup, rows


def first_anchor_with(t_rel, H, m_max, want):
    """Earliest row whose causal support has at least `want` windows."""
    for i in range(len(t_rel)):
        if len(M.support(t_rel, i, H, m_max)) >= want:
            return i
    return None


def mm(e, n):
    return float(np.sum(e) / max(np.sum(n), 1.0))


def table(title, cols, err, cnt, bins, key, base='blind'):
    print('\n%s' % title)
    hdr = '%-14s %8s' % ('', 'frames')
    for c in cols:
        hdr += ' %9s' % c
    print(hdr + ' %9s' % ('gain'))
    for lo, hi in bins:
        s = (key >= lo) & (key < hi)
        if not s.any():
            continue
        row = '%-14s %8d' % ('[%d,%s)' % (lo, 'inf' if hi > 10 ** 8 else hi),
                             int(s.sum()))
        vals = {}
        for c in cols:
            vals[c] = mm(err[c][s], cnt[c][s])
            row += ' %9.2f' % vals[c]
        main = [c for c in cols if c not in (base, 'ceil', 'shuf')]
        g = vals[base] - vals[main[0]] if main else 0.0
        print(row + ' %+9.3f' % g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-ep', type=int, default=0, help='0 = all val episodes')
    ap.add_argument('--every', type=str, default='1,8,16,32,64')
    ap.add_argument('--ceil-steps', type=int, default=150)
    ap.add_argument('--ceil-m', type=int, default=16)
    ap.add_argument('--out', type=str, default=os.environ.get('META_OUT', ''))
    a = ap.parse_args()

    dev = th.device(DEVICE)
    paths = [p for p in EVAL.split(',') if p and os.path.exists(p)]
    model, ck, cfg = build(dev)
    H, K, M_MAX = cfg['H'], cfg['k_inner'], cfg['m_max']
    # Checkpoints written before the cfg fix in train_meta_stream.py record the
    # WARM-START source's k_inner/m_max rather than their own, so a run that
    # deliberately changed them needs these overrides to be evaluated correctly.
    M_MAX = int(os.environ.get('META_M_MAX_EVAL', M_MAX))
    # META_K_EVAL overrides the number of inner steps at TEST time only. The
    # model was meta-trained for cfg['k_inner'] steps, so anything else is
    # off-distribution -- but `_probe_e1_steps.py` measured the gain PEAKING at
    # K=5 (+0.394 vs +0.165 at K=1), so the trained value is not the best one.
    # Accepts a COMMA LIST: META_K_EVAL=1,3,5 emits one cadence column per K.
    # K=3 and K=5 measured +0.366 and +0.394 on 30 episodes -- a 0.028 mm
    # difference that this data cannot separate, so neither is ruled out and
    # both should be reported rather than one picked.
    KL = [int(x) for x in str(os.environ.get('META_K_EVAL', K)).split(',') if x]
    K = KL[0]
    q0 = model.q0.detach().unsqueeze(0)
    everies = [int(x) for x in a.every.split(',') if x]
    print('[ckpt] %s  step %s  val_loss %.3f mm' %
          (CKPT, ck.get('gstep', ck.get('epoch')), ck.get('val_loss', float('nan'))))
    print('[cfg ] z_dim=%d K=%d (trained %d) m_max=%d train_every=%d alpha=%.4f'
          % (cfg['z_dim'], K, cfg['k_inner'], M_MAX, cfg['every'],
             float(model.alpha())))

    va = np.asarray([int(x) for x in ck['val_ids']], np.int64)
    if a.n_ep:
        va = va[np.linspace(0, len(va) - 1, min(a.n_ep, len(va))).astype(int)]
    print('[data] %d val episodes' % len(va))

    def ecol(ev, k):
        return 'e%d' % ev if len(KL) == 1 else 'e%dk%d' % (ev, k)

    cols = (['blind'] + [ecol(e, k) for k in KL for e in everies]
            + ['once', 'ceil', 'shuf'])
    err = {c: [] for c in cols}
    cnt = {c: [] for c in cols}
    nsup_all, t_all, ep_all, mass_all = [], [], [], []
    qfit = []
    q_prev = None

    for k_ep, e in enumerate(va):
        ds = FricTrackDataset(paths, phys_col=cfg['phys_col'], episodes=[int(e)])
        data, t_rel, sc = M.cache_episode(ds, dev)
        if not bool(sc.any()):
            continue
        rows = np.nonzero(sc.cpu().numpy())[0]

        e0, n0 = M.score_per_frame(model, M.take(data, rows), q0)
        err['blind'].append(e0)
        cnt['blind'].append(n0)

        ref_nsup = None
        for k in KL:
            for ev in everies:
                ee, nn, ns, _ = walk(model, data, t_rel, sc, q0, H, k, M_MAX, ev)
                err[ecol(ev, k)].append(ee)
                cnt[ecol(ev, k)].append(nn)
                if ev == cfg['every'] and k == K:
                    ref_nsup = ns
        if ref_nsup is None:
            _, _, ref_nsup, _ = walk(model, data, t_rel, sc, q0, H, K, M_MAX,
                                     cfg['every'])

        # once: fit at the first anchor with a full support set, then freeze
        i0 = first_anchor_with(t_rel, H, M_MAX, M_MAX)
        if i0 is None:
            q_once = q0
        else:
            sup = M.pack_support([M.take(data, M.support(t_rel, i0, H, M_MAX))],
                                 M_MAX)
            with th.enable_grad():
                q_once, _ = M.adapt(model, sup, q0.clone(), K,
                                    create_graph=False)
        # ONLY frames at or after the fit anchor. Scoring earlier frames with
        # this latent is ACAUSAL -- it applies evidence that had not elapsed
        # yet -- and it inflated `once` from ~+0.03 to +0.152 mm before this
        # guard, visibly: the [0,1)-support bucket showed a +0.59 mm "gain" on
        # frames where no window exists. Frames before the anchor are scored at
        # q0, exactly as a deployed system would have to.
        eo, no = M.score_per_frame(model, M.take(data, rows), q0)
        if i0 is not None:
            post = t_rel[rows] >= t_rel[i0]
            if post.any():
                e1, n1 = M.score_per_frame(
                    model, M.take(data, rows[post]), q_once)
                eo = eo.copy(); no = no.copy()
                eo[post], no[post] = e1, n1
        err['once'].append(eo)
        cnt['once'].append(no)

        q_ceil = fit_noncausal(model, data, t_rel, q0, a.ceil_steps, a.ceil_m)
        ec, nc = M.score_per_frame(model, M.take(data, rows), q_ceil)
        err['ceil'].append(ec)
        cnt['ceil'].append(nc)

        qs = q_prev if q_prev is not None else q0
        es, ns_ = M.score_per_frame(model, M.take(data, rows), qs)
        err['shuf'].append(es)
        cnt['shuf'].append(ns_)
        q_prev = q_ceil.detach()

        nsup_all.append(ref_nsup)
        t_all.append(t_rel[rows])
        ep_all.append(np.full(len(rows), int(e), np.int64))
        mass_all.append(np.full(len(rows), float(data['phys'][0, 0]), np.float64))
        qfit.append(q_ceil[0].detach().cpu().numpy())
        del data
        if (k_ep + 1) % 25 == 0:
            th.cuda.empty_cache()
            print('  .. %d/%d episodes' % (k_ep + 1, len(va)))

    err = {c: np.concatenate(v) for c, v in err.items()}
    cnt = {c: np.concatenate(v) for c, v in cnt.items()}
    nsup = np.concatenate(nsup_all)
    tpos = np.concatenate(t_all)
    epid = np.concatenate(ep_all)
    mass = np.concatenate(mass_all)

    print('\n=== mv:object mean per-step L2 (mm), %d frames, %d episodes ==='
          % (len(nsup), len(np.unique(epid))))
    w = '%-10s' % 'column'
    print('%-10s %10s %10s' % ('column', 'mm', 'vs blind'))
    base = mm(err['blind'], cnt['blind'])
    for c in cols:
        v = mm(err[c], cnt[c])
        print('%-10s %10.3f %+10.3f' % (c, v, base - v))

    # The bucket tables show ONE cadence column. With a K list the column names
    # carry the K (`e16k3`), so name it through `ecol` -- hardcoding 'e16' threw
    # KeyError only AFTER the main table had printed, which looked like a crash
    # in the breakdown rather than a naming bug.
    main_col = ecol(cfg['every'], KL[-1] if len(KL) > 1 else K)
    if main_col not in err:
        main_col = ecol(everies[0], KL[-1] if len(KL) > 1 else K)
    table('by elapsed-window count (support size at the anchor):',
          ['blind', main_col, 'once', 'ceil', 'shuf'], err, cnt,
          NSUP_BINS, nsup)
    table('by frame index within the episode:',
          ['blind', main_col, 'once', 'ceil', 'shuf'], err, cnt, T_BINS, tpos)

    # per-episode win rate -- a pooled mean can hide a method that helps a few
    # episodes a lot and hurts most of them
    ue = np.unique(epid)
    wins = 0
    for e in ue:
        s = epid == e
        if mm(err[main_col][s], cnt[main_col][s]) < mm(err['blind'][s],
                                                       cnt['blind'][s]):
            wins += 1
    print('\nper-episode: %d/%d better than blind (%.1f%%)'
          % (wins, len(ue), 100.0 * wins / max(len(ue), 1)))

    Q = np.stack(qfit)
    lm = np.log(np.array([mass[epid == e][0] for e in ue]))
    cc = [abs(float(np.corrcoef(Q[:, d], lm)[0, 1])) for d in range(Q.shape[1])]
    print('max |corr(latent dim, log mass)| = %.3f  (ceiling-fit latents)'
          % max(cc))

    if a.out:
        np.savez_compressed(a.out, q=Q, ep=ue, log_mass=lm, nsup=nsup,
                            tpos=tpos, epid=epid, ep_mass=mass,
                            **{'err_%s' % c: err[c] for c in cols},
                            **{'cnt_%s' % c: cnt[c] for c in cols})
        print('[out] %s' % a.out)


if __name__ == '__main__':
    main()
