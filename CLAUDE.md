# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DyWA (Dynamics-adaptive World Action Model) is a research codebase for generalizable
non-prehensile (pushing) robot manipulation, built on NVIDIA Isaac Gym and extended from
the prior CORN project. Everything runs **inside a Docker container** — Isaac Gym, the
custom CUDA kinematics kernels, and the pinned dependency stack (CUDA 11.3, PyTorch3D
0.7.2, Python 3.8) are not expected to work on a bare host.

## Environment & build

The container is the unit of work, not a virtualenv. Typical lifecycle:

```bash
# On the host: launch/enter the container (edit the *_PATH vars at the top first —
# they hardcode host paths for Isaac Gym, the dataset, and caches).
bash docker/run.sh                      # first run, creates container "dywa_1"
docker start -ai dywa_1                 # subsequent runs

# Inside the container, from the repo root (mounted at /home/user/DyWA):
bash setup.sh                           # installs isaacgym + builds/installs the dywa pkg
python3 -c 'import isaacgym; print("OK")'
python3 -c 'import dywa; print("OK")'
```

`setup.sh` runs `pip install --no-build-isolation -e ~/DyWA/dywa`. The editable install
compiles the C++/CUDA extensions (`franka_kin_cuda`, `ur5_kin_cuda`) from `dywa/c_src` +
`dywa/src/cxx` via `cmake_build_extension`. If you change a `.cu`/`.cpp` kernel, re-run the
pip install. There is no separate lint/test suite — validation is done by running the
training/eval scripts.

Host paths the container expects (set in `docker/run.sh`): repo → `/home/user/DyWA`,
dataset → `/input` (e.g. `/input/DGN`), Isaac Gym → `/opt/isaacgym`, HF/torch cache →
`/home/user/.cache/pkm`, scratch → `/tmp/docker`.

## Package layout (important: flat namespace)

The installed package lives in `dywa/src/` but its subdirectories are exposed as
**top-level** import roots, not under a `dywa.` prefix. Code imports
`from models.common import ...`, `from env.env.wrap.base import ...`, `from util.hydra_cli import ...`.
The roots are: `models/`, `env/`, `data/`, `train/`, `util/`. Keep this in mind — `grep`
for `from models` / `from env`, not `from dywa.models`.

- `dywa/src/models/` — neural nets. `rl/` (PPO, encoders, normalizers, LR schedulers),
  `cloud/` (Point-MAE point-cloud encoder), `sdf/`. `modules.py` holds shared blocks
  (e.g. `HistoryEncoder`).
- `dywa/src/env/` — Isaac Gym simulation. `arm_env.py`/`push_env.py` are the env wrappers;
  `task/`, `robot/` (franka, ur5), `scene/` define the manipulation setup. `env/wrap/` is a
  stack of composable observation/recording wrappers.
- `dywa/src/data/cfg/` — **Hydra config root** (see below).
- `dywa/exp/train/` — runnable entry-point scripts (Python).
- `dywa/exp/scripts/` — shell wrappers that invoke the entry points with the right Hydra args.

## Training/eval entry points

All entry scripts are in `dywa/exp/train/` and the shell wrappers in `dywa/exp/scripts/`
`cd` into `dywa/exp/train` before running. The canonical pipeline:

1. **Teacher RL (stage 1)** — `train_ppo_arm.py` — privileged-state PPO policy.
   `bash dywa/exp/scripts/train_teacher_stage1.sh`
2. **Teacher RL (stage 2)** — action-space reduction fine-tune.
   `bash dywa/exp/scripts/train_teacher_stage2.sh /path/to/stage1_ckpt.pt`
3. **Teacher→student distillation (RMA)** — `train_rma.py` / `distill.py` — distills the
   privileged teacher into a vision (point-cloud) student. Three observation regimes:
   `train_distill_abs_goal_1view.sh`, `..._abs_goal_3view.sh`, `..._rel_goal_3view.sh`.
4. **Evaluation** — `test_rma.py` — `bash dywa/exp/scripts/eval_student_unseen_obj.sh [GPU]`
   evaluates a student on the unseen-object `test_set.json`. Results → `output/test_rma/`.

`show_ppo_arm.py` runs/visualizes a teacher policy; `valid_ppo_arm.py` validates.

See `dywa/exp/train/README.md` for the full walkthrough and CLI option reference.

## Hydra configuration model

Runs are configured entirely through Hydra. Config groups live under `dywa/src/data/cfg/`
and are selected on the command line:

- `+platform=` (`debug`|`dr`|`desktop`|`srv`|...) — `debug` disables WandB; `dr` enables
  WandB/HF logging (requires `wandb login` / `huggingface-cli login`).
- `+env=` — environment/object-set/domain-randomization (e.g. `icra_base`, `abs_goal_1view`).
- `+run=` — policy/run config (e.g. `icra_ours`, `teacher_base`).
- `+student=` — student-network config (e.g. `dywa/base`).
- `++key=value` — override any leaf in the composed config (e.g. `++env.num_env=1024`,
  `++global_device=cuda:0`, `++path.root=...`, `++load_ckpt=...`).

Checkpoint args (`++icp_obs.icp.ckpt`, `++load_ckpt`, `+load_student`) accept either a local
path or a HuggingFace ref formatted as `entity/repository:filename`.

The Hydra search path resolves to `dywa/src/data/cfg` via `util/hydra_cli.py` (overridable
with the `PKM_CFG_PATH` env var).

## Conventions & gotchas

- **JIT is disabled by default** via `PYTORCH_JIT=0` in every script — JIT stability is
  hardware-dependent. Enable with `PYTORCH_JIT=1` only if your GPU+Docker setup supports it.
  If a JIT compile is wedged: `rm -rf ~/.cache/torch_extensions`.
- **Visualization**: add `++env.use_viewer=1 ++draw_debug_lines=1` and export `$DISPLAY` to
  match the host. It is slow — drop `++env.num_env` to a handful when visualizing.
- `HF_ENDPOINT='https://hf-mirror.com'` works around HuggingFace network issues.
- The "goal" observation key was unified to `rel_goal` (see git history) — watch for
  key-name mismatches between teacher and student obs when touching distillation code.
- Outputs default to `/home/user/DyWA/output/...`; the eval categorical-results figure and
  per-object success rates land under `output/test_rma/`.
- **Bottle-only training budget**: for the bottle-only ablations
  (`dywa/exp/scripts/train_bottle_ablation.sh`), `++train_step=20000` is a decent
  convergence budget. Checkpoint frequency is `++save_step` (top-level `cfg.save_step`,
  default 10000) — set it to `2000` for these shorter runs so intermediate ckpts are saved.

