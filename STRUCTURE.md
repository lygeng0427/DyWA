# Repository structure

DyWA — non-prehensile (pushing) manipulation on Isaac Gym, extended from CORN.
Everything runs **inside the Docker container** (`docker/run.sh`); the pinned
CUDA 11.3 / Python 3.8 / PyTorch3D stack is not expected to work on a bare host.
Build and run instructions, plus the measured gotchas behind every pipeline
below, are in `CLAUDE.md` — this file is only the map.

## Top level

```
docker/          container build + run.sh (edit the *_PATH vars first)
setup.sh         installs isaacgym, then `pip install -e dywa/`
dywa/
  c_src/, src/cxx/   CUDA/C++ kinematics kernels (franka_kin_cuda, ur5_kin_cuda)
  src/               the installed python package  (see "Package layout")
  exp/train/         runnable entry points (python)
  exp/scripts/       shell wrappers that supply the Hydra args
fig/             figures from the original paper
```

**Package layout is a flat namespace.** `dywa/src/` subdirectories are exposed
as *top-level* import roots, not under a `dywa.` prefix — code says
`from models.common import ...`, `from env.env.wrap.base import ...`. Roots:
`models/`, `env/`, `data/`, `train/`, `util/`. Grep for `from models`, never
`from dywa.models`.

Runs are configured through **Hydra**, config root `dywa/src/data/cfg/`,
selected on the command line with `+platform=`, `+env=`, `+run=`, `+student=`
and overridden leaf-by-leaf with `++key=value`. Outputs default to
`output/` (gitignored).

---

## A. Original DyWA / CORN pipeline (teacher RL → vision student)

A privileged-state PPO teacher is trained in simulation, then distilled into a
point-cloud student that sees no privileged physics.

| stage | entry point | wrapper |
|---|---|---|
| 1. teacher PPO (privileged state) | `train_ppo_arm.py` | `train_teacher_stage1.sh` |
| 2. teacher fine-tune (action-space reduction) | `train_ppo_arm.py` | `train_teacher_stage2.sh` |
| 3. teacher → student distillation (RMA) | `train_rma.py` | `train_distill_{abs_goal_1view,abs_goal_3view,rel_goal_3view}.sh` |
| 4. evaluation on unseen objects | `test_rma.py` | `eval_student_unseen_obj.sh` |

Supporting modules:

- `distill.py` — the distillation losses and student/teacher plumbing shared by
  `train_rma.py`, `test_rma.py` and the TTT pipeline below.
- `rma_env.py` — builds the student's observation stack (`setup_rma_env_v2`);
  imported by almost everything that needs a live env, including our collectors.
- `rma.py` — agent construction/loading used by `test_rma.py`.
- `distill_ppo_arm.py`, `show_ppo_arm.py`, `valid_ppo_arm.py` — run, visualise
  and validate a teacher policy.
- `render_train_objects.py` — one scene image per object in a filtered set.
- Bottle-only ablations: `train_bottle_ablation.sh`, `eval_bottle_ablation.sh`,
  `render_train_can_bottle.sh`, `eval_record_can_bottle.sh`.

The student network itself lives in `src/models/` — `rl/` (PPO, encoders,
normalizers), `cloud/` (Point-MAE point-cloud encoder), `sdf/`, with shared
blocks (e.g. `HistoryEncoder`) in `modules.py`.

---

## B. Our pipeline

Five layers, each building on the one above. They share one simulator-replay
data format and one set of model packages.

### B1. Offline TTT on the teacher dataset (`src/models/ttt.py`)

Replaces the RMA feed-forward `cond` with a latent belief adapted by gradient
descent at test time.

`collect_teacher_dataset.py` → `meta_train_ttt.py` → `eval_ttt.py`
(wrappers `collect_ttt.sh` / `meta_train_ttt.sh` / `eval_ttt.sh`, plus the
`*_ttt_mass.sh` variants that sweep object mass).

### B2. GAP 3D-track collection (`src/models/gap/`)

Ground-truth point-flow tracks by **simulator replay**, not tracking: unproject
each DINOv2 patch centre, express it in the owning rigid body's local frame,
replay it forward through that body's recorded poses. Drift-free by
construction.

- `collect_gap_tracks.py` / `collect_gap_tracks.sh` — the collector.
- `_verify_gap_tracks.py` — **run this after every collection**; "it ran" is not
  evidence.
- `visualize_gap_tracks.py` — overlay tracks (and model predictions) on frames.
- `_render_gap_ghosts.py` — render the translucent goal-pose ghost for videos.
- `train_gap_track.py` / `eval_gap_track.py` (+ `.sh`) — the track models.
- `src/models/gap/`: `common.py` (torch-1.11 ports, `FieldNorm`, FiLM encoder
  layer), `dataset.py` (episode-level splits), `track_model.py` (transformer and
  DiT trunks behind one interface).

### B3. Physics-conditioned track models (`src/models/fric/`)

Sweeps one hidden dynamics variable — table/object friction or object **mass** —
across envs and asks whether handing the model the true value recovers track
accuracy. `FricTrackModel` subclasses `gap.TrackModel`; `phys_cond='none'` is
bit-identical to the GAP model, so the baseline arm *is* the GAP model.

- `collect_fric_tracks.py` / `.sh` — per-env physics bake (`FRIC_CHANNEL`).
- `train_fric_track.py` / `eval_fric_track.py` (+ `.sh`).
- `visualize_fric_conditioning.py`, `visualize_mass_pred.py` — make the effect
  of conditioning visible.
- `shrink_dino.py` — narrow a dataset's `dino_features` 1024 → k in place
  (storage is the binding constraint: features are ~93% of each frame).

### B4. Streaming collection + training

Unbounded unique data on bounded disk: generate shards, train on a FIFO window
of recent ones, delete the rest. Answers the overfitting problem, not a new
scientific question — a track model is a *passive* predictor, so there is no
on-policy requirement and no DAgger.

`collect_stream_tracks.py` / `train_stream_tracks.py` (+ `.sh`).

### B5. Meta-learned belief + joint action/track policy (`src/models/policy/`)

The track model promoted to a policy. Two models, **no shared weights**, bridged
only by a latent belief `q`:

- **inner** — `models/fric/latent_model.py` (`LatentTrackModel`). Its object-only
  track error on already-elapsed windows defines `dL/dq`; `meta_ttt.adapt` takes
  K unit-normalised steps. Trained by `train_meta_track.py` /
  `train_meta_stream.py`, evaluated by `eval_meta_track.py`.
- **outer** — `models/policy/joint_dit.py` (`JointPolicyDiT`). GAPPolicy's joint
  denoiser, de-bimanualised: a context encoder over `[CLS, state, 391 patches]`
  and one self-attention stack over `[16 action, 391 track, state, timestep]` at
  a single shared timestep. Hand-rolled DDIM — no `diffusers` in this container.

| role | entry point | wrapper |
|---|---|---|
| headroom gate (no inner loop) | `train_policy.py` | `train_policy.sh` |
| second-order meta-training | `train_meta_policy.py` | `train_meta_policy.sh` |
| offline eval (blind/ttt/ceil/shuf) | `eval_meta_policy.py` | `eval_meta_policy.sh` |
| **closed-loop rollout, success rate** | `eval_rollout_policy.py` | `eval_rollout.sh` |

`meta_policy.py` and `meta_ttt.py` are the shared cores (inner-loop adaptation,
per-episode walks, diagnostics). Design figures are produced by
`_fig_joint_pipeline_clean.py` (method) and `_fig_joint_pipeline.py` (annotated).

`ttt_with_dyn_bar.py` is **reference only** — the pick-bar implementation this
design follows (FiLM on the raw latent, second-order MAML); it is not part of
any DyWA pipeline and is not runnable here.

---

## Model packages at a glance

```
src/models/
  common.py, modules.py   shared blocks (HistoryEncoder, ...)
  rl/ cloud/ sdf/         original DyWA student + teacher networks
  ttt.py                  offline TTT belief for the RMA student        (B1)
  gap/                    track prediction: trunks, dataset, FieldNorm  (B2)
  fric/                   physics-conditioned + latent track models     (B3, B5 inner)
  policy/                 joint action+track DiT                        (B5 outer)
```

---

## Conventions

- `PYTORCH_JIT=0` in every script — JIT stability is hardware-dependent.
- Visualisation: `++env.use_viewer=1 ++draw_debug_lines=1`, and drop
  `++env.num_env` to a handful.
- Outputs, checkpoints, datasets, wandb and notes are gitignored. Long-form
  experiment write-ups (`*_REPORT.md`, `_plan_*.md`) stay on disk but out of the
  repo; their durable conclusions live in `CLAUDE.md`.
- One-off probes, diagnostics and per-experiment run wrappers are **not** in the
  repo. `CLAUDE.md` cites many `_probe_*.py` as the evidence for its claims;
  those files were moved out of the tree during cleanup and are kept beside it
  in an archive directory rather than deleted.
