#!/usr/bin/env python3
"""
Offline TTT meta-training for the DyWA student.

DyWA analog of `ttt_with_dyn_bars.py`. Loads the teacher-rollout dataset
produced by `collect_teacher_dataset.py`; for each successful episode:

  * sample a query timestep `t`; support = transitions `[0..t)`,
  * inner loop: adapt the belief `q0 -> q_k` by TTT on the support (second-order,
    `create_graph=True`, `detach_q=False`),
  * outer loss: `GaussianKLDivLoss(student.belief_forward(query_obs, q_k),
    teacher_action_t)`.

`q0`, `dyn`, `outcome_enc` and the student trunk are trained jointly, purely
through the outer imitation loss via the inner-loop graph (no separate dynamics
loss). Support clouds/actions are normalized in the query's frame (invariant #3).

Runs inside the Docker container (needs the DyWA point encoder / CUDA), but does
NOT instantiate Isaac Gym.

Example:
    TTT_DATA=./dataset_ttt_teacher.pkl TTT_EPOCHS=200 \
    PYTORCH_JIT=0 python3 meta_train_ttt.py \
        +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
        ++student.norm=ln ++student.ttt.latent_dim=64 ++student.ttt.detach_q=False
"""

import os
# Must precede any torch import: `expandable_segments:True` trips a
# CUDACachingAllocator assert during gradient-checkpointed double backward
# (offline second-order). Harmless even with checkpointing off.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')

import isaacgym  # noqa: F401  (keep early: some DyWA modules assume it is imported)

import pickle
from dataclasses import replace

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import wandb
from tqdm.auto import tqdm

from util.hydra_cli import hydra_cli
from util.config import recursive_replace_map
from train_rma import Config as RMAConfig, get_config_path
from distill import StudentAgentTTT
from train.losses import GaussianKLDivLoss
from models.ttt import frame_of, to_frame, to_frame_vec, rigid_flow, ee_pose_delta


def _envs(name, d):
    return os.environ.get(name, d)


DATA_PATH = _envs('TTT_DATA', './dataset_ttt_teacher.pkl')
EPOCHS = int(_envs('TTT_EPOCHS', 200))
ACCUM = int(_envs('TTT_ACCUM', 32))          # effective batch (grad accumulation)
END_DROPOUT = float(_envs('TTT_END_DROPOUT', 0.1))
TRAIN_RATIO = float(_envs('TTT_TRAIN_RATIO', 0.9))
SAVE_DIR = _envs('TTT_SAVE_DIR', './checkpoints/ttt_dywa')
SEED = int(_envs('TTT_SEED', 0))
QUERY_STRIDE = int(_envs('TTT_QUERY_STRIDE', 1))   # subsample query steps (1 = every step)
# Cap iterations per epoch: with many samples/trajectory a full pass is huge, so
# each epoch draws this many random samples (0 = full pass). Optimizer steps per
# epoch ≈ STEPS_PER_EPOCH / ACCUM.
STEPS_PER_EPOCH = int(_envs('TTT_STEPS_PER_EPOCH', 10000))
# Anti-collapse FlowDecoder reconstruction weight. Overrides the student config's
# `ttt.recon_weight` so it can be swept without editing the config. Larger values
# push harder against OutcomeEncoder collapse (probes showed eff_rank ~6/64).
RECON_WEIGHT = float(_envs('TTT_RECON_WEIGHT', 5.0))

# WandB logging (replaces the old TensorBoard SummaryWriter), mirroring
# ttt_with_dyn_bars.py. Set TTT_WANDB_MODE=disabled for local/debug runs that
# should not log (parallels the old `+platform=debug` no-wandb behavior);
# 'offline' logs to disk for later `wandb sync`.
WANDB_PROJECT = _envs('TTT_WANDB_PROJECT', 'ttt_dywa')
WANDB_ENTITY = _envs('TTT_WANDB_ENTITY', 'r-pad')
WANDB_MODE = _envs('TTT_WANDB_MODE', 'online')


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — flat over (episode, query-step): a length-T successful trajectory
# emits one sample per valid query `t`, each with its K-strided macro-history.
# ─────────────────────────────────────────────────────────────────────────────
class TeacherEpisodes(Dataset):
    def __init__(self, episodes, state_keys, window_k, history_len, query_stride=1):
        self.eps = episodes
        self.state_keys = state_keys
        self.K = int(window_k)
        self.H = int(history_len)          # max macro-transitions kept
        # Flat index: every valid query step becomes its own sample (multi-step
        # TTT). `t >= K` guarantees ≥1 K-window support transition [t-K, t].
        self.index = []

        for i, ep in enumerate(episodes):
            T = int(ep['length'])
            for t in range(self.K, T, max(1, query_stride)):
                self.index.append((i, t))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        i, t = self.index[idx]
        ep = self.eps[i]
        K = self.K

        query_obs = {k: th.as_tensor(np.asarray(ep[k][t], dtype=np.float32))
                     for k in (list(self.state_keys) + ['partial_cloud', 'goal_cloud'])
                     if k in ep}
        teacher_action = th.as_tensor(np.asarray(ep['teacher_action'][t], dtype=np.float32))  # [2, A]

        # K-strided macro-transition support windows ending at t, most-recent
        # first: [t-K, t], [t-2K, t-K], … back to step 0 — the FULL history, no
        # sliding cap. Friction is constant within an episode, so every past
        # window informs the same belief; truncating would discard evidence. Each
        # window `s` spans steps [s, s+K] (s+K ≤ t ≤ T-1, so all indices valid).
        starts = np.asarray(list(range(t - K, -1, -K)), dtype=np.int64)       # [M]
        M = len(starts)
        pc = ep['partial_cloud']; hs = ep['hand_state']
        ta = ep['teacher_action']; op = ep['object_pose']

        sup_c = th.as_tensor(np.asarray(pc[starts], dtype=np.float32))        # [M,N,3] cloud_s
        # rigid flow over each window (world frame; a constant target/input).
        pose_s = th.as_tensor(np.asarray(op[starts], dtype=np.float32))       # [M,7]
        pose_sK = th.as_tensor(np.asarray(op[starts + K], dtype=np.float32))  # [M,7]
        sup_flow = rigid_flow(sup_c, pose_s, pose_sK)                         # [M,N,3]
        # macro-action pieces (assembled by student.build_macro per mode):
        hand_s = th.as_tensor(np.asarray(hs[starts], dtype=np.float32))        # [M,9] pose6d
        hand_sK = th.as_tensor(np.asarray(hs[starts + K], dtype=np.float32))   # [M,9]
        sup_ee = ee_pose_delta(hand_s, hand_sK)                                # [M,6] Δpos ‖ rotvec
        sup_mg = th.as_tensor(np.stack([ta[s:s + K, 0, 6:].mean(0) for s in starts]).astype(np.float32))  # [M,n_gains]
        sup_seq = th.as_tensor(np.stack([ta[s:s + K, 0, :] for s in starts]).astype(np.float32))          # [M,K,A]
        return query_obs, teacher_action, sup_c, sup_flow, sup_ee, sup_mg, sup_seq


def _split(episodes, ratio, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(episodes))
    k = int(len(episodes) * ratio)
    tr = [episodes[j] for j in idx[:k]]
    va = [episodes[j] for j in idx[k:]]
    return tr, va


def _prep_support(query_pcd, sup_c, sup_flow, sup_ee, device):
    """Normalize support into the query's shared frame (invariant #3).

    Clouds get center+scale; the flow and the EE Δpos are *displacements* → scale
    only (no centering). The EE Δrot6d (`sup_ee[..., 3:]`) is frame-invariant.
    """
    center, scale = frame_of(query_pcd.to(device))            # [1,1,3], [1,1,1]
    sup_c = to_frame(sup_c.to(device), center, scale)         # [M,N,3]
    sup_flow = to_frame_vec(sup_flow.to(device), scale)       # [M,N,3]
    sup_ee = sup_ee.to(device).clone()
    sup_ee[..., :3] = to_frame_vec(sup_ee[..., :3], scale)    # Δpos scale-normalized
    return sup_c, sup_flow, sup_ee


def _outer_loss(student, loss_fn, query_obs, teacher_action, q, device):
    q_b = q.unsqueeze(0) if q.dim() == 1 else q            # [1, D]
    obs = {k: v.to(device) for k, v in query_obs.items()}
    out = student.belief_forward(obs, q_b)                 # [1, 2, A]
    ta = teacher_action.to(device)
    if ta.dim() == 2:
        ta = ta.unsqueeze(0)                              # [1, 2, A]
    return loss_fn(out[..., 0, :], out[..., 1, :], ta[..., 0, :], ta[..., 1, :])


@th.no_grad()
def _collate_query(query_obs):
    # DataLoader(batch_size=1) already adds the leading [1, ...] dim; nothing to do.
    return query_obs


def _adapt(student, q0, query_obs, sup_c, sup_flow, sup_ee, sup_mg, sup_seq,
           device, create_graph, detach_q):
    """Prep support → build macro-action → run TTT adaptation. Returns
    `(q_k, sc, sfl, macro)`: `sc`/`sfl` are the shared-frame cloud/flow (reused
    by the reconstruction loss), `macro` the per-window macro-action (reused by
    the inner-loss diagnostic)."""
    sc, sfl, see = _prep_support(query_obs['partial_cloud'], sup_c, sup_flow, sup_ee, device)
    macro = student.build_macro(see, sup_mg.to(device), sup_seq.to(device))   # [M, macro_dim]
    q_k = student.adapt(q0, sc, macro, sfl, create_graph=create_graph, detach_q=detach_q)
    q_k = th.nan_to_num(q_k, nan=0.0)
    return q_k, sc, sfl, macro


@th.no_grad()
def _inner_loss(student, q, sc, macro, sfl):
    """The TTT inner objective at belief `q`:
    `MSE(dyn(q, macro, cloud_s), OutcomeEncoder(cloud_s, flow))` over the support
    windows — the exact quantity `ttt_inner_update` descends. Evaluated before
    and after the inner step, it says whether the inner loop is functioning as an
    optimizer at all (doors' `pre_in_loss`/`post_in_loss`)."""
    H = sc.shape[0]
    q_exp = q.detach().unsqueeze(0).expand(H, -1)          # [H, D]
    z_pred = student.dyn(q_exp, macro, sc)                 # [H, D]
    z_act = student.outcome_enc(sc, sfl)                   # [H, D]
    return float(F.mse_loss(z_pred, z_act))


def evaluate(student, loader, loss_fn, device):
    """Returns (val_pre, val_post, q_move, val_pre_rand, in_pre, in_post).

    Two independent questions, four diagnostics:

    *Does the INNER loop work?* — `in_pre`/`in_post` are the TTT inner objective
    (dyn-vs-outcome latent MSE) before/after the inner step, and `q_move` is
    ‖q_post − q0‖. `in_post < in_pre` means the step genuinely descends its own
    objective; `in_post ≈ in_pre` (or `q_move ≈ 0`) means it does not — the inner
    signal, step size, or dyn's q-sensitivity is broken, and no amount of outer
    training will help.

    *Does the POLICY use q?* — `pre_rand` is the val loss at a RANDOM belief
    q0+ε·N(0,1), ε=ttt_alpha. `pre_rand ≈ pre` means the policy is insensitive to
    belief perturbations of TTT's own step size, i.e. it ignores `q` outright.

    Reading the combination when `post ≈ pre`:
      * in_post ≈ in_pre                 → inner loop is not optimizing.
      * in_post < in_pre, pre_rand ≈ pre → inner loop works, policy ignores `q`.
      * in_post < in_pre, pre_rand ≫ pre → policy uses `q`, but TTT moves it in
                                           an unhelpful direction.
    """
    student.eval()
    pre, post, qmove, rand, in_pre, in_post = [], [], [], [], [], []
    for query_obs, teacher_action, sup_c, sup_flow, sup_ee, sup_mg, sup_seq in loader:
        sup_c, sup_flow, sup_ee, sup_mg, sup_seq = (
            sup_c.squeeze(0), sup_flow.squeeze(0), sup_ee.squeeze(0),
            sup_mg.squeeze(0), sup_seq.squeeze(0))
        q0 = student.q0
        with th.no_grad():
            pre.append(_outer_loss(student, loss_fn, query_obs, teacher_action, q0, device).item())
            qr = q0.detach() + th.randn_like(q0) * float(student.cfg.ttt.ttt_alpha)
            rand.append(_outer_loss(student, loss_fn, query_obs, teacher_action, qr, device).item())
        if sup_c.shape[0] > 0:
            with th.enable_grad():
                q_post, sc, sfl, macro = _adapt(student, q0.detach(), query_obs, sup_c,
                                                sup_flow, sup_ee, sup_mg, sup_seq, device,
                                                create_graph=False, detach_q=True)
            q_post = q_post.detach()
            qmove.append((q_post - q0.detach()).norm().item())
            # Inner objective before/after the step, on the same support windows.
            in_pre.append(_inner_loss(student, q0.detach(), sc, macro, sfl))
            in_post.append(_inner_loss(student, q_post, sc, macro, sfl))
        else:
            q_post = student.q0.detach()
        with th.no_grad():
            post.append(_outer_loss(student, loss_fn, query_obs, teacher_action, q_post, device).item())
    student.train()
    mean = lambda xs: float(np.mean(xs)) if xs else 0.0
    return (float(np.mean(pre)), float(np.mean(post)),
            mean(qmove), float(np.mean(rand)), mean(in_pre), mean(in_post))


@hydra_cli(config_path=get_config_path(), config_name='train_rl')
def main(cfg: RMAConfig):
    cfg = recursive_replace_map(cfg, {'finalize': True})
    device = cfg.global_device or ('cuda' if th.cuda.is_available() else 'cpu')
    th.manual_seed(SEED)
    np.random.seed(SEED)

    # Each ttt_alpha gets its own checkpoint/TB dir, so an alpha sweep never
    # overwrites itself and TensorBoard can compare the runs side by side
    # (point tensorboard at the parent SAVE_DIR).
    alpha = float(cfg.student.ttt.ttt_alpha)
    trunk = str(cfg.student.ttt.policy_trunk)
    # ttt_alpha, policy_trunk AND recon_weight are all part of the run dir, so any
    # sweep over them writes to a distinct location instead of overwriting. (The
    # trunk matters for loading too: the two trunks have incompatible state_dicts
    # and `load(..., strict=False)` would silently leave one random.)
    run_name = f'alpha_{alpha:g}' if trunk == 'dywa' else f'{trunk}_alpha_{alpha:g}'
    run_name += f'_recon{RECON_WEIGHT:g}'
    save_dir = os.path.join(SAVE_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"ttt_alpha={alpha:g}  policy_trunk={trunk}  ->  ckpt dir: {save_dir}")

    tp_cfg = cfg.student.ttt
    wandb.init(
        project=WANDB_PROJECT, name=run_name, entity=(WANDB_ENTITY or None),
        mode=WANDB_MODE, dir=save_dir,
        config={
            'epochs': EPOCHS, 'accum': ACCUM, 'end_dropout': END_DROPOUT,
            'train_ratio': TRAIN_RATIO, 'steps_per_epoch': STEPS_PER_EPOCH,
            'query_stride': QUERY_STRIDE, 'seed': SEED,
            'ttt_alpha': alpha, 'policy_trunk': trunk,
            'recon_weight': RECON_WEIGHT,
            'window_k': int(tp_cfg.window_k),
            'macro_action': str(tp_cfg.macro_action),
            'latent_dim': int(tp_cfg.latent_dim),
            'k_inner_steps': int(tp_cfg.k_inner_steps),
            'detach_q': bool(tp_cfg.detach_q),
        })
    # Two logging cadences: per-optimizer-step train metrics (x=gstep) vs
    # per-epoch val metrics (x=epoch). define_metric ties each to its own axis.
    wandb.define_metric('gstep')
    wandb.define_metric('epoch')
    wandb.define_metric('train/*', step_metric='gstep')
    wandb.define_metric('val/*', step_metric='epoch')
    wandb.define_metric('epoch/*', step_metric='epoch')
    print(f"wandb: project={WANDB_PROJECT} entity={WANDB_ENTITY} run={run_name} mode={WANDB_MODE}")

    with open(DATA_PATH, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} successful episodes from {DATA_PATH}")
    train_eps, val_eps = _split(episodes, TRAIN_RATIO, SEED)

    state_keys = list(cfg.student.state_keys or [])
    student_cfg = replace(cfg.student, batch_size=1)
    student = StudentAgentTTT(student_cfg, writer=None, device=device).to(device)
    student.train()

    loss_fn = GaussianKLDivLoss()
    opt = student.optimizer  # covers q0 / dyn / outcome_enc / trunk

    tp = student.cfg.ttt
    train_set = TeacherEpisodes(train_eps, state_keys, window_k=tp.window_k,
                                history_len=tp.history_len, query_stride=QUERY_STRIDE)
    # Coarser val (stride K) keeps per-epoch evaluation cheap.
    val_set = TeacherEpisodes(val_eps, state_keys, window_k=tp.window_k,
                              history_len=tp.history_len, query_stride=tp.window_k)
    if STEPS_PER_EPOCH > 0 and STEPS_PER_EPOCH < len(train_set):
        # draw a fresh random subset each epoch (covers all query steps over epochs)
        sampler = RandomSampler(train_set, replacement=True, num_samples=STEPS_PER_EPOCH)
        train_loader = DataLoader(train_set, batch_size=1, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4) # set num_workers as 0 for debugging, 4 otherwise
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2) # set num_workers as 0 for debugging, 2 otherwise
    print(f"  train samples={len(train_set)} (many per trajectory); {len(train_loader)} steps/epoch"
          f"  |  val samples={len(val_set)}")
    print(f"  window_k={tp.window_k}  macro_action={tp.macro_action}  macro_dim={student.macro_dim}"
          f"  recon_weight={RECON_WEIGHT}")

    print(f"  q0/dyn/outcome params: "
          f"{sum(p.numel() for p in student.dyn.parameters())/1e6:.2f}M (dyn) + "
          f"{sum(p.numel() for p in student.outcome_enc.parameters())/1e6:.2f}M (outcome)")

    best_post = float('inf')
    best_gap = float('-inf')          # biggest val_pre - val_post (TTT-helps-most)
    gstep = 0
    for epoch in range(EPOCHS):
        opt.zero_grad()
        running, n_acc, win, win_recon = 0.0, 0, 0.0, 0.0
        for i, (query_obs, teacher_action, sup_c, sup_flow, sup_ee, sup_mg, sup_seq) in enumerate(
                tqdm(train_loader, desc=f'epoch {epoch+1}/{EPOCHS}')):
            sup_c, sup_flow, sup_ee, sup_mg, sup_seq = (
                sup_c.squeeze(0), sup_flow.squeeze(0), sup_ee.squeeze(0),
                sup_mg.squeeze(0), sup_seq.squeeze(0))

            use_hist = (np.random.rand() >= END_DROPOUT) and (sup_c.shape[0] > 0)
            if use_hist:
                q_k, sc, sfl, _ = _adapt(student, student.q0, query_obs, sup_c, sup_flow,
                                         sup_ee, sup_mg, sup_seq, device,
                                         create_graph=True, detach_q=tp.detach_q)
            else:
                q_k = student.q0

            kl = _outer_loss(student, loss_fn, query_obs, teacher_action, q_k, device)

            # Anti-collapse reconstruction: FlowDecoder(pcd+flow, OutcomeEncoder(pcd,flow))
            # must reproduce the flow (outer-loop only). Recomputes the outcome latent
            # (the inner-loop one is on a detached-q graph).
            recon = th.zeros((), device=device)
            if use_hist and RECON_WEIGHT > 0.0 and tp.recon_on_history:
                z_hist = student.outcome_enc(sc, sfl)                 # [M, D]
                flow_pred = student.flow_decoder(sc + sfl, z_hist)    # [M, N, 3]
                recon = F.mse_loss(flow_pred, sfl)

            loss = kl + RECON_WEIGHT * recon
            (loss / ACCUM).backward()
            running += kl.item(); win += kl.item(); win_recon += float(recon)

            if (i + 1) % ACCUM == 0:
                nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad()
                wandb.log({'train/kl': win / ACCUM,
                           'train/recon': win_recon / ACCUM,
                           'gstep': gstep})
                gstep += 1; n_acc += 1; win = 0.0; win_recon = 0.0

        # flush the remainder
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()

        (val_pre, val_post, val_qmove, val_rand,
         val_in_pre, val_in_post) = evaluate(student, val_loader, loss_fn, device)
        train_kl = running / max(len(train_loader), 1)
        print(f"epoch {epoch+1} | train_kl={train_kl:.4f} "
              f"| val_pre={val_pre:.4f} val_post={val_post:.4f} gap={val_pre-val_post:+.4f} "
              f"| q_move={val_qmove:.4f} pre_rand={val_rand:.4f} "
              f"| inner {val_in_pre:.4f}->{val_in_post:.4f} ({val_in_pre-val_in_post:+.4f})")
        wandb.log({
            'epoch': epoch,
            'epoch/train_kl': train_kl,
            'val/pre': val_pre,
            'val/post': val_post,
            'val/gap_pre_minus_post': val_pre - val_post,        # >0 = TTT helps
            'val/q_move': val_qmove,                             # ‖q_post-q0‖
            'val/pre_rand': val_rand,                            # loss at random belief
            'val/inner_pre': val_in_pre,                         # inner objective @ q0
            'val/inner_post': val_in_post,                       # inner objective @ q_post
            # >0 = the inner step actually descends its own objective
            'val/inner_gap_pre_minus_post': val_in_pre - val_in_post,
        })

        student.save(os.path.join(save_dir, 'latest.ckpt'))
        if val_post < best_post:
            best_post = val_post
            student.save(os.path.join(save_dir, 'best.ckpt'))
            print(f"  [*] best (val_post={best_post:.4f})")
        if (val_pre - val_post) > best_gap:
            best_gap = val_pre - val_post
            student.save(os.path.join(save_dir, 'best_gap.ckpt'))
            print(f"  [*] best_gap (val_pre-val_post={best_gap:+.4f})")

    wandb.finish()
    print("Meta-training complete.")


if __name__ == '__main__':
    main()
