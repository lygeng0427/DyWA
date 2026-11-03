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

## TTT (test-time training) pipeline

A separate three-stage pipeline that replaces the RMA feed-forward `cond` with a latent
belief `q` adapted by gradient descent at test time. Entry points:
`collect_teacher_dataset.py` → `meta_train_ttt.py` → `eval_ttt.py`, wrapped by
`collect_ttt.sh` / `meta_train_ttt.sh` / `eval_ttt.sh`. All commands run **inside the
container**; the trailing integer is the GPU index (`eval_ttt.sh` takes `<k_inner_steps>
[GPU]`).

**One bottle** (`bottle_one.json`, 1 object — friction is the only hidden variable, so
`TTT_FIX_PHYS=1` pins scale/mass/restitution):

```bash
# 1. collect
TTT_FIX_PHYS=1 TTT_FILTER=/input/DGN/bottle_one.json \
TTT_NUM_EPISODES=50 TTT_MAX_STEPS=40000 \
TTT_OUT=/home/user/DyWA/output/ttt/dataset_one_bottle.pkl \
bash dywa/exp/scripts/collect_ttt.sh 0

# 2. meta-train  (-> ckpt_one_bottle/film_transformer_alpha_0.5_recon5/)
TTT_DATA=/home/user/DyWA/output/ttt/dataset_one_bottle.pkl \
TTT_SAVE_DIR=/home/user/DyWA/output/ttt/ckpt_one_bottle \
TTT_POLICY_TRUNK=film_transformer TTT_ALPHA=0.5 TTT_RECON_WEIGHT=5 \
bash dywa/exp/scripts/meta_train_ttt.sh 0

# 3. eval sweep (k=0 is the no-TTT baseline)
for K in 0 1 2 5; do
  TTT_SAVE_DIR=/home/user/DyWA/output/ttt/ckpt_one_bottle \
  TTT_POLICY_TRUNK=film_transformer TTT_ALPHA=0.5 TTT_RECON_WEIGHT=5 \
  TTT_FIX_PHYS=1 TTT_FILTER=/input/DGN/bottle_one.json TTT_NUM_ENV=60 \
  bash dywa/exp/scripts/eval_ttt.sh $K 0
done
```

**All bottles** (train on `bottle_train.json` = 19 bottles, eval on the held-out
`bottle_test.json` = 5; full DR, no `TTT_FIX_PHYS`). These are the script defaults for
data/ckpt/filter paths, so only the trunk/alpha need setting:

```bash
# 1. collect  (-> output/ttt/dataset_ttt_teacher.pkl)
bash dywa/exp/scripts/collect_ttt.sh 0

# 2. meta-train  (-> output/ttt/ckpt/film_transformer_alpha_1_recon5/)
TTT_POLICY_TRUNK=film_transformer TTT_ALPHA=1 TTT_RECON_WEIGHT=5 \
bash dywa/exp/scripts/meta_train_ttt.sh 0

# 3. eval sweep on the held-out bottles
for K in 0 1 2 5; do
  TTT_POLICY_TRUNK=film_transformer TTT_ALPHA=1 TTT_RECON_WEIGHT=5 \
  bash dywa/exp/scripts/eval_ttt.sh $K 0
done
```

Gotchas, all of which fail *silently*:

- `{TTT_POLICY_TRUNK, TTT_ALPHA, TTT_RECON_WEIGHT}` are what compose the run-dir name, so
  they select the checkpoint at eval and **must match meta-training**. A trunk mismatch
  loads nothing under `strict=False` and leaves the whole policy random (`eval_ttt.py`'s
  `_assert_ckpt_covers_policy` catches it — unless you bypass it with `TTT_STUDENT_CKPT=`).
- `TTT_FIX_PHYS` must match between collect and eval, or the student is evaluated on a
  different dynamics distribution than it was trained on.
- Success metric is `val_pre − val_post > 0` during meta-training (printed per epoch, and
  logged to WandB; `TTT_WANDB_MODE=disabled` to turn logging off). Eval success rates are
  printed to stdout — every `k` overwrites the same file under `output/test_rma/`.
- A done env is reset in the **next** `env.step`'s `pre_physics`, so obs/`tensors['root']`
  read after a done step still hold the old episode's terminal state. Both the collector
  and `eval_ttt.py` compensate (`skip_next` / `blk_ok = ~stale`); any new code that
  snapshots state across an episode boundary must do the same, or windows spanning the
  reset teleport poison the belief. `_probe_reset_staleness.py` verifies this.

## Reference: GAP (`third_party/GAP`)

Read-only reference checkout (CVPR 2026, joint action + 3D-track diffusion for bimanual
RoboTwin). **Not executable here** — it needs RoboTwin/SAPIEN and a torch-2.4.1/CUDA-12.1
stack. We refer to it while porting ideas into DyWA. Architecture details are already in its
`README.md` and the `gap_policy/policy/gap.py` docstring — read those, don't re-derive.

The four files we care about:

- `scripts/generate_gt_tracks.py` — GT 3D point-flow tracks by *simulator replay*, not
  tracking: unproject each patch center to a world point, express it in the owning link's
  local frame, push it forward through that link's recorded pose. Drift-free by
  construction. The **method** ports to DyWA; the **script** does not (RoboTwin envs,
  `cam.get_picture("Segmentation")`, cached `_traj_data/episode{N}.pkl` joint paths).
  DyWA's equivalent pose source is `env.tensors['root'][scene.cur_ids, :7]` — raw and
  **env-local**, not `obs['object_state']`.
- `gap_policy/policy/gap.py` — the DiT. Interface: obs `agent_pos`[B,16] /
  `dino_features`[B,V,391,1024] / `patch_centers`[B,N,3]; normalizer keys `action` and
  `gt_3d_track`; N=391 hardcoded as 17×23; track horizon 16.
- `scripts/visualize_tracks.py` (tracks drawn on RGB → mp4) and `scripts/analyze_track_loss.py`
  (per-step L2 in mm). Both need the raw `episode{N}.hdf5` for K/extrinsics and load
  checkpoints as `{cfg, model, ema, normalizer}` via `hydra.utils.instantiate(cfg.policy)`.

Gotchas:

- **Track key names disagree across the repo.** `generate_gt_tracks.py` writes `3d_track` /
  `track_valid_mask`; the policy, `TrackDataset` and `GAP.yaml` default to `gt_3d_track` /
  `track_valid_mask_gt`. Selectable via `track_key`/`mask_key`, so a mismatch reads the
  *wrong dataset* rather than crashing.
- `generate_gt_tracks.py` can't run from where it sits: `PROJECT_ROOT = __file__/../..`
  resolves to the GAP root, so it looks for `GAP/RoboTwin` and `GAP/GAP`. It was written for
  a parent workspace with sibling `RoboTwin/` and `GAP/`. It also `os.chdir(ROBOTWIN_ROOT)`
  at import and resolves CLI paths against the launch CWD.
- `analyze_track_loss.py`'s `"move"` group is a **validity** split, not a movement split —
  the mask is `depth_valid & is_object`, which includes stationary object points. The actual
  movement weighting is a separate displacement sigmoid inside `compute_loss`.
- **Dependency conflict, unresolved.** GAP pins torch 2.4.1+cu121, diffusers 0.38.0,
  numpy<2, zarr 2.18.3; our container is Python 3.8 / CUDA 11.3. Verify before porting
  anything that imports `diffusers`.

## GAP 3D-track generation (DyWA side)

`dywa/exp/train/collect_gap_tracks.py` (wrapper: `dywa/exp/scripts/collect_gap_tracks.sh`)
ports GAP's ground-truth point-flow *method* to Isaac Gym. Full design notes + the
measurements behind it: `_plan_gap_tracks.md`.

**Mechanism, per frame:** render RGB + depth + per-body labels with a dedicated
`NvdrCameraWrapper` (238x322, so the 17x23=391 DINOv2 patch centres land on exact integer
pixels `14k+7`) -> unproject each patch centre to a 3D point -> look up the rigid body that
owns it from the label -> express the point in that body's local frame -> replay it forward
through that body's recorded poses for H=16 steps. The stored track is the per-step
*displacement* from the anchor. **No tracker, no drift** — it is simulator replay, exact by
construction. DINOv2 ViT-L/14-reg features for the same 391 patches are extracted inline.

**Caveats** (each measured, details below/in the plan):

- Colour is **greyscale** — DyWA meshes carry no materials. DINOv2 still separates
  object/arm/table from background (0.50–0.82 cosine dissimilarity), but this is not GAP's
  textured RGB.
- **Occlusion is not modelled**: a patch anchored on a visible surface keeps a valid track
  even after the point is hidden. GAP behaves the same way.
- The final H frames of an episode carry `track_valid_mask_gt=False` (no future window
  exists). They are kept so the goal-reaching tail is visible — **filter on the mask, not on
  frame index**.
- Mass DR is **8 discrete levels**, not a continuum, across any number of episodes.
- Camera framing dominates supervision density; the fov default is load-bearing (see below).
- `episode_success` is recorded, but success is the task's own criterion (`pos_err <=
  goal_radius` on a pose/keypoint error) — the object *centre* at the last frame is a
  median ~3.6 cm from the goal centre, not 0.

```bash
# prereq (HOST): DINOv2 ViT-L/14-reg weights, 1.2 GB, into the mounted cache
#   host ~/.cache/dywa  ==  container /home/user/.cache/pkm
mkdir -p ~/.cache/dywa/dinov2 && cd ~/.cache/dywa/dinov2 && curl -L -C - -o model.safetensors \
  https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m/resolve/main/model.safetensors

# collect (container)
GAP_OUT=/home/user/DyWA/output/gap/tracks.hdf5 GAP_NUM_EPISODES=50 \
bash dywa/exp/scripts/collect_gap_tracks.sh 0

# ALWAYS verify -- "it ran" is not evidence
python3 dywa/exp/train/_verify_gap_tracks.py /home/user/DyWA/output/gap/tracks.hdf5

# overlay the tracks on the frames -> mp4 (GAP's visualize_tracks.py equivalent)
python3 dywa/exp/train/visualize_gap_tracks.py /home/user/DyWA/output/gap/tracks.hdf5 \
  --episode 0 --out /home/user/DyWA/output/gap/ep0.mp4 --upscale 3 --skip-bodies 0

# translucent goal-pose GHOST (as the TTT videos have): render it once, then blend.
# A ghost is a real nvdr render of the object at its goal pose -- it cannot be
# reconstructed from stored frames, so this render-only pass rebuilds the env
# (never steps physics) and asserts its camera matches the dataset's T_cam attr.
python3 dywa/exp/train/_render_gap_ghosts.py \
  --spec mass:/home/user/DyWA/output/gap/tracks_100succ_massvar.hdf5:97,10 \
  --out /home/user/DyWA/output/gap/ghosts.npz <the usual hydra args>
python3 dywa/exp/train/visualize_gap_tracks.py <tracks.hdf5> --episode 97 \
  --ghost /home/user/DyWA/output/gap/ghosts.npz --ghost-key mass:97 --out ...
```

`visualize_gap_tracks.py` inverts nvdr's `unproject` using the camera written into the
file's attrs, and **asserts every anchor re-projects onto its own patch centre (`14i+7`)
before drawing** — a wrong projection would otherwise produce a plausible-looking but
meaningless overlay. Colour ramps green (static) → red (moving). `/tmp/docker/` in the
container is host `$TMP_PATH` (`/home/liyuan/tmp`).

**Video codec**: it writes H.264/yuv420p via `imageio_ffmpeg`'s bundled binary (there is no
system ffmpeg in the container). Do **not** go back to `cv2.VideoWriter(...'mp4v')` — that
produces MPEG-4 Part 2, which most players reject with "unsupported codec". H.264 also needs
even frame dimensions, so the output size is rounded down to even.

Knobs: `GAP_{OUT,NUM_EPISODES,MAX_STEPS,TRACK_HORIZON,DINO,STORE_IMAGE,NUM_ENV,FILTER,
MIN_EP_LEN,MAX_EP_LEN,SUCCESS_ONLY,EMIT_TAIL,CAM_FOV_DEG,CAM_EYE,CAM_AT,FIX_PHYS}`.
Length/outcome policy matches `collect_ttt.sh` exactly: `MIN_EP_LEN=4`, **no** upper cap,
success-only. `GAP_FIX_PHYS=friction|mass` pins everything except one dynamics variable
(mirrors `TTT_FIX_PHYS`); unset = full `icra_base` DR.
Output is GAP-flat HDF5: `gt_3d_track`, `track_valid_mask_gt`, `image`, `dino_features`,
`image_patch_centre_3d_position`, `robot_state`, `action`, `patch_owner_body`, plus
`goal_pose` / `object_pose` / `phys_params` (all env-local, physical units) and per-episode
`episode_ends` / `episode_success` / `episode_length`. ~780 KB/frame with features+images.
`phys_params` is `[mass, table_fric, obj_fric, hand_fric, restitution]` read straight from
Isaac Gym — **not** `obs['phys_params']`, which the env normalizer standardises (same trap as
`obs['object_state']`).

Physics sensitivity, measured on the two fixed-physics datasets (object track magnitude vs
the free variable, 100 episodes each): **mass r = −0.268** (heavier moves less),
table/object friction **r ≈ +0.13–0.14**, hand friction **r ≈ 0**. Friction is weak but not
inert — the earlier "friction is inert" finding was measured on *success rate*; tracks are a
finer readout.

Facts this depends on, each measured by a `_probe*.py` — **do not re-derive from the Isaac
Gym docs, which disagree**:

- **The Isaac Gym camera is unusable here**: it never renders the object mesh or the franka
  (only the ground + table). Use `NvdrCameraWrapper`, which renders everything and supplies
  colour + depth + per-body labels in one pass. `_diag_nvdr_rgb.py` dumps them as PNGs.
- **nvdr's `unproject()` returns ENV-LOCAL coordinates** — the same frame as
  `tensors['body']`, so no origin correction. The Isaac Gym camera's view matrix is
  **global**, the opposite; that code path must not be copied (`_probe_nvdr_frame.py`,
  `_probe_ig_camera_frame.py`). Env origins are a 2 m grid, and **env 0 has origin (0,0,0),
  so a bug here looks perfect in env 0 and is metres wrong everywhere else** — always check
  an env with a nonzero origin.
- **nvdr labels are `1 + rigid_body_index`, `0` = background.** They are float32 and
  interpolate across silhouette edges, so round and reject non-integers.
- Rendering natively at 238×322 makes the 17×23=391 patch centres land on exact integer
  pixels (`14k+7`), so there is no resize and no sub-pixel sampling — both sources of error
  in GAP's own script are avoided.
- **Colour is greyscale** (DyWA meshes carry no materials; CoACD `.obj`s have no `vt`/`mtl`).
  Accepted: pooled over 235 frames, DINOv2 still separates object (0.75), arm links
  (0.78–0.82) and table (0.50) from background by cosine dissimilarity.
- `patch_owner_body` is `-1` on invalid patches, so a consumer that forgets
  `track_valid_mask_gt` gets an obviously-wrong index instead of silently reading background
  as body 0 (the table).
- Frames whose H-step window crosses an episode reset are dropped (`ep_start` bookkeeping) —
  see the reset-staleness gotcha in the TTT section.
- **Camera framing dominates supervision density and is easy to get wrong.** nvdr's default
  90° fov spans ~1.8 m at this eye distance while the region of interest is ~0.8 m, so most
  of the 391 patches land on black background. Measured: 90° → 18.8% valid patches and
  **1.6 patches/frame on the object**; 45° → 58.4% valid and **8.3 on the object**. Defaults
  are now 45° (`GAP_CAM_FOV_DEG`, `GAP_CAM_EYE`, `GAP_CAM_AT`). Re-measure after any change
  — the per-body patch census is the quantity to watch, not the picture.

`_verify_gap_tracks.py` is the guard: it checks the table's tracks are exactly 0, that arm
links move monotonically more down the kinematic chain, and that a track predicts frame
`t+h`'s anchors ~3× better than assuming no motion.

**Collector throughput** is ~45 frames/s and the work is per-FRAME, not per-step: 8→32 envs
moves throughput only 37→47 f/s while step rate falls 8.34→2.53. So `GAP_NUM_ENV` is a weak
lever (32 is the measured knee, now the default) and the real one was the writer — `gzip`
level 4 cost 26.4 ms/frame and capped everything at 38 f/s; `lzf` costs 3.3 ms for a file
11% larger. 300 episodes ≈ 5 min/GPU. DINOv2 is not a bottleneck (4.4 ms/frame at E=8).

## GAP track models (`dywa/src/models/gap/`)

Track prediction only — no action head, no bimanual split (DyWA has one 7-D EEF pose).
`common.py` holds the torch-1.11 ports (explicit softmax attention, buffer-based normalizer,
`RoPE4D.rotate` renamed off `nn.Module.apply`) plus a FiLM-on-FFN `EncoderLayer`.
`track_model.py` has two trunks behind one interface; `dataset.py` splits by EPISODE.

Inputs are `dino_features` [391,1024], `patch_centers` [391,3] and `agent_pos` [7] (EEF
pose). **There is no latent belief**: every episode shares one pinned friction config, so
there is no hidden dynamics variable to infer and a latent could only memorise per-episode
nuisance (initial pose, goal). `EncoderLayer` keeps its `film_dim` hook, unused — the
varying-friction experiment needs exactly that path back, and that is the point at which a
belief becomes meaningful. See git history for the removed TTT-style
per-episode-latent version and its measurements.

```bash
bash dywa/exp/scripts/collect_gap_tracks.sh 0     # GAP_FIX_PHYS=fixed pins ALL physics
bash dywa/exp/scripts/train_gap_track.sh transformer 0 both  # realized|command|both|realized_gain
bash dywa/exp/scripts/eval_gap_track.sh output/gap/ckpt/transformer_d512_both 0

# see the PREDICTIONS: cyan = model, green->red ramp = ground truth
python3 dywa/exp/train/visualize_gap_tracks.py output/gap/tracks_act_a.hdf5 \
  --episode 118 --only-bodies 1 --upscale 3 \
  --pred output/gap/ckpt/transformer_d512_both/best.ckpt --out output/gap/pred.mp4
```

**Curves go to WandB**, project `dywa-gap-track` (`GAP_WANDB_PROJECT`;
`GAP_WANDB_MODE=disabled` to turn it off). The run URL is printed as `[wandb] ...` at
startup — `train/loss`, `val/loss`, `val/gap` (val−train, the overfitting readout) and
`train/lr`. Run name is the ckpt dir name (`transformer_d512`, `dit_d512`).

Use `--only-bodies 1` (the pushed object) when visualising: the franka links carry ~40× more
patches with far larger displacements and bury the object. Body ids: 0 table, **1 object**,
7–12 franka links.

### Action conditioning ablation (`GAP_ACTION_MODE`)

The 16 action tokens can carry either of two things DyWA offers, and they are NOT
interchangeable. `action` in the HDF5 is the **realized** next EEF pose read back out of the
sim; `action_cmd` is the **policy's own 20-D command** (3 dpos, 3 dori axis-angle, 7 joint KP,
7 joint KD — `ctrl_mode: cpos_n`, `gain: variable`, bounds in the `act_lo`/`act_hi` attrs).
Commanded vs realized dpos correlate only **0.79/0.92/0.42** per axis: IK, impedance, control
noise and a randomized delay sit between them, and the 14 gain dims have no counterpart in a
pose at all.

Transformer, 20 epochs, 301 single-friction episodes (241/60 split), mean per-step L2 in mm.
`realized_gain` = realized pose + the 14 gain dims only, dropping the commanded dpos/dori:

| arm | act in_dim | **mv:object** | mv:arm | moving | all |
|---|---|---|---|---|---|
| realized | 7 | 28.6 / 28.6 | **5.1 / 5.4** | 7.3 | 32.9 |
| command | 20 | 30.6 | 22.2 | 22.3 | 32.4 |
| realized_gain | 21 | 26.5 | 7.6 | 9.3 | 29.4 |
| both | 27 | **26.0** | 8.9 | 10.3 | 29.9 |
| zero-pred | – | 55.4 | 104.1 | 95.6 | 16.7 |

(two numbers = two init seeds at a FIXED split; `GAP_SPLIT_SEED` is separate from `GAP_SEED`
precisely so a seed repeat does not also change the val episodes)

- **Seed spread on the object is 0.05 mm** (28.62 vs 28.57), so the ordering above is real and
  nowhere near init noise. Arm spread is 0.29 mm.
- **The aggregate row inverts the conclusion — always read `mv:object`.** On `moving`,
  `realized` looks 3x better than `command`. That gap is almost entirely the ARM: the realized
  path *is* the arm's future pose, so predicting arm tracks from it is nearly a lookup. On the
  OBJECT the same comparison is only 7%.
- **The 14 gain dims carry real information, and MORE so with more data.** Re-tested at the
  2491-episode scale (obj_weight=9, 10 epochs, same 60 val episodes):

  | action input | transformer object | DiT object | tf arm | dit arm |
  |---|---|---|---|---|
  | **realized_gain** (7+14) | **14.7** | **15.9** | **4.5** | **6.0** |
  | realized (7 only) | 18.6 | 19.8 | 4.8 | 6.7 |

  Dropping the gains costs **+27% object error** (14.7 -> 18.6) on the transformer and +24% on
  the DiT, versus only 7% back at 241 episodes. So stiffness/damping say something about
  contact force that the model can only exploit once it has enough data to learn it -- a small
  ablation in an overfitting regime UNDERSTATED the effect nearly 4x. Undershoot worsens too
  (magnitude ratio 0.99 -> 0.96 tf, 0.99 -> 0.91 dit).
- **The arm penalty from input width does NOT survive scale.** At 241 episodes arm error
  tracked action width (7 dims -> 5.1 mm, 21 -> 7.6, 27 -> 8.9) and looked like an
  architectural dilution problem. At 2491 the narrower input is slightly WORSE on the arm too
  (4.8 vs 4.5), so that effect was also a small-data artifact. Do not carry it forward.
- **Planning usability**: only `command` needs nothing executed. It costs 30.6 vs 26.0 on the
  object (18%) — far less than the aggregate implies.

**Is the object prediction good? No — and measure it against the right scale.** Divide the
error by the *horizon-averaged* GT magnitude (the `zero-pred` column, 55 mm), NOT by the
endpoint displacement (95 mm): the error is averaged over all H steps, so the endpoint is the
wrong denominator and flatters the result nearly 2x. Correct figures for the best object arm:

| quantity | object | arm |
|---|---|---|
| mean per-step error | 27.4 mm | 8.0 mm |
| mean per-step \|gt\| | 55.1 mm | 104.0 mm |
| **relative error** | **50%** | **8%** |
| endpoint direction cosine(pred, gt) | 0.79 (~37 deg off) | – |

- **It is not just magnitude shrinkage.** The magnitude ratio is a healthy 0.95, but the
  endpoint direction cosine is 0.79 — the predicted direction is ~37 deg off on average. The
  overlay videos read as "right direction, slightly short"; the aggregate says otherwise.
- **Almost no skill at the first step**: h=1 error 7.3 mm against \|gt\| 8.3 mm (89% relative).
  Relative error then IMPROVES with horizon (h=16: 47%). The model captures the coarse "object
  gets shoved along" trend and misses the contact response that generates it.
- It is nonetheless non-trivial: 2.0x better than predict-zero and **3.4x better than "the
  object translates rigidly with the EEF"** (94.5 mm) — that physical baseline is worse than
  zero, because the EEF moves 143 mm while the object moves 95 mm and contact is intermittent.
- **The arm reaching 8% relative proves the architecture is not the ceiling.** The object is
  data/objective limited, not capacity limited.
- **The loss is 90% arm — and REWEIGHTING IT BARELY HELPS.** Movement-weighted share is
  object 12.9% / arm 87.1%. `GAP_OBJ_WEIGHT` multiplies object patches (9 -> 57% share,
  30 -> 82%). Measured on `realized_gain`:

  | obj_weight | object share | mv:object | mv:arm | cos(dir) | h=1 err/\|gt\| |
  |---|---|---|---|---|---|
  | 1 | 12.9% | 26.5 | **7.6** | 0.792 | 7.3 / 8.3 mm |
  | **9** | 57.1% | **25.0** | 10.5 | 0.810 | 7.0 / 8.3 mm |
  | 30 | 81.6% | 25.3 | 14.4 | 0.801 | 7.1 / 8.3 mm |

  A 4.4x increase in object gradient share buys **5.7%** (26.5 -> 25.0 mm, vs a 0.05 mm seed
  spread, so real but small), and going further to 82% buys nothing. Arm degrades
  monotonically. Direction cosine moves 0.79 -> 0.81 and first-step skill does not move at
  all — the model did not learn more about contact, it just re-allocated capacity.
  **Conclusion: the objective was not the bottleneck; information is.** Use `obj_weight=9`
  (the arm cost is irrelevant downstream — a planner already knows the arm's future) but do
  not expect more from this knob.
- **It is OVERFITTING, not an observability floor.** Object error on TRAIN vs VAL episodes,
  same model: **8.0 mm (14% rel) vs 26.7 mm (51% rel) -- a 3.35x gap.** Seven patches are
  enough to determine object motion to 14%; what fails is generalisation. An earlier reading
  of this codebase concluded "information-limited" because the loss reweighting saturated --
  that inference was wrong, since saturating reweighting is equally consistent with
  memorising the training objects. **Always separate the two with a train-vs-val split before
  blaming observability.**
- **More data is the lever that works, and it keeps working.** Val set held bit-identical
  across all three runs via `GAP_VAL_POOL` (draws val only from the first N episodes so new
  files can only enlarge train -- otherwise appending a file repermutes the split and the
  comparison is against different episodes):

  | train eps | object | rel | arm | obj train mm | obj val mm | gap |
  |---|---|---|---|---|---|---|
  | 241 | 25.0 | 45% | 10.5 | 8.0 | 26.7 | 3.35x |
  | 691 | 19.1 | 35% | 5.9 | 6.8 | 20.3 | 3.01x |
  | **2491** | **14.7** | **26%** | **4.5** | 7.7 | 15.5 | **2.00x** |

  Object error **25.0 -> 14.7 mm (-41%)**, now **3.78x** better than predict-zero, and
  magnitude shrinkage is gone (ratio 0.94 -> 0.99). Train error is FLAT at ~7-8 mm across all
  three while val falls -- textbook: the model was always able to do the task, it just could
  not generalise. The gap finally moved (3.35 -> 2.00).
- **Scaling exponent, three points**: error ~ N^-0.26 over 241->691, then N^-0.21 over
  691->2491. Slightly bending but still paying. A 2-point extrapolation predicted 13.8 mm at
  2500 and the actual was 14.7 -- close enough to trust the shape, not the digits. Halving
  again (-> ~7 mm) needs roughly 25x more data, which is NOT storable here: `dino_features` is
  **93.4% of the 837 KB/frame** (782 KB), so 2551 episodes already costs 205 GB and 5000 would
  cost 381 GB. Dropping `image` saves only 4.7%. Any larger run needs the features stored
  smaller (quantised/projected) or recomputed on the fly (~150 ms/batch, ~75 min/epoch at 5k).
- DiT tracks the transformer closely (15.9 object) but stays worse on the arm (6.0 vs 4.5).
- Two structural fixes were tried and BOTH FAILED at 241 episodes -- and note both were
  attempts to add inductive bias to what turned out to be a sample-count problem, so neither
  addressed the actual failure. Re-test them at 691+ before dismissing outright.

  | change | object | arm | train loss | verdict |
  |---|---|---|---|---|
  | baseline | 25.0 | 10.5 | 0.024 | – |
  | **A** temporal context (`GAP_HIST_K=2`) | 24.9 | 10.6 | 0.023 | **exact null** |
  | **B** rigid per-body SE(3) head (`GAP_HEAD=rigid`) | 29.9 | 14.3 | 0.098 | **worse** |

  **A (velocity) changes nothing** — 25.0 -> 24.9, inside the 0.05 mm seed spread, and the
  whole loss curve is superimposed on the baseline's. Reading: the model is already given the
  arm's FUTURE EEF path, which pins down where the object will be pushed far better than the
  object's current velocity does. Past velocity is redundant with future action. (This would
  likely NOT hold for an action-free or command-only model.)

  **B (rigidity) actively hurts**, and it is UNDERFITTING, not undertrained: train loss
  plateaus at 0.098 against the free head's 0.024, i.e. 4x worse on data it has seen. The
  constraint is *true* of the labels — `compute_tracks` makes each body's field exactly 6 DoF
  — so this is an optimization/conditioning failure, not a modelling error. Suspects: mean
  pooling over a body's patches is a hard bottleneck, and rotation error multiplied by a
  ~0.5 m lever arm is badly conditioned. A residual formulation (rigid motion as a correction
  to a free prediction) or attention pooling might rescue it; plain substitution does not.

## Mass-conditioned track models (`FRIC_CHANNEL=mass`, `output/mass/`)

Same question as the friction section, different hidden variable: **does varying
object mass degrade track prediction, and does the ground-truth value recover
it?** Same pipeline -- `mass` is a third channel of the `fric` code, not a fork;
`models/gap/*` and `collect_gap_tracks.py` remain untouched.

```bash
# range-find FIRST (the grid is an empirical question, see below)
bash dywa/exp/scripts/probe_mass_range.sh 0        # 64 masses x 2 seeds

FRIC_CHANNEL=mass FRIC_LO=0.1 FRIC_HI=1.2 FRIC_SPACING=log GAP_SEED=0 \
  GAP_OUT=output/mass/tracks_m_a.hdf5 bash dywa/exp/scripts/collect_fric_tracks.sh 0
python3 dywa/exp/train/_probe_fric_census.py output/mass/tracks_m_a.hdf5
FRIC_CHANNEL=mass FR_CKPT= bash dywa/exp/scripts/probe_fric_openloop.sh 1   # gate
bash _run_mass_ab.sh                               # 8 runs, ~14 h on 2 GPUs
```

### The grid is 0.1-1.2 kg, 32 LOG steps -- both ends measured, not chosen

`_probe_mass_range.py` sweeps a deliberately too-wide range and reports, per
mass: closed-loop success, episode length, **object path per step**, and
open-loop displacement. Pooled over 64 masses x 2 seeds, 2558 episodes:

| mass | mm/step | vs light end | success | ep len |
|---|---|---|---|---|
| 0.03-0.15 | 7.0-8.3 | **~1.00 (saturated)** | 0.63-0.91 | 105-150 |
| 0.65 | 5.3 | 0.70 | 0.86 | 139 |
| 1.16 | 3.2 | 0.42 | 0.50 | 213 |
| 1.55 | 1.9 | 0.25 | 0.23 | 274 |
| 2.08+ | <=1.0 | <=0.14 | **0.00** | **299 = timeout** |

Two facts that fix the endpoints, and they push in OPPOSITE directions from the
obvious guess: (a) **below ~0.15 kg the response saturates** -- the pusher fully
entrains the bottle and mass stops mattering, so points there are wasted; (b)
**"barely moves" starts at ~1.5 kg**, not 1.0, and 2.0 kg is dead (success 0.00,
every episode at the timeout). Log spacing because mass enters as `a ~ F/m`, so
equal RELATIVE steps are what should be evenly sampled.
`corr(log mass, mm/step) = -0.90`, versus table friction's +0.13 on track
magnitude.

### Gotchas specific to the mass channel

- **`recomputeInertia=True` is load-bearing** (4th arg of
  `set_actor_rigid_body_properties`). Without it inertia stays at the config mass
  while mass changes: the object translates as if heavy and rotates as if light.
  A pushed bottle's track is mostly rotation, so this corrupts exactly the signal
  under study **while the readback still looks perfect**.
- A mistimed friction write is silently inert; a mistimed (mid-sim) **mass write
  CORRUPTS the sim** -- the object explodes. Pre-`prepare_sim` is mandatory.
- Reset-time DR never touches mass, so `read_raw_phys`'s mass readback is
  truthful on its own (unlike friction). The by-construction stamp is kept anyway.
- `assert len(pr) == 1`: `for p in pr` writes EVERY body of the actor, so a
  multi-body asset would sit at N x the labelled mass.
- **The log map lives ONLY in `FricTrackModel.phys_vec`.** The file, `fric_grid`,
  `bucketize` and `assert_phys_grid` all speak raw kg. Splitting it would leave
  two components disagreeing about what `phys` means, and the failure mode is a
  silently compressed input, not an exception. `phys_lo`/`phys_hi` are now
  REQUIRED -- the old friction-valued 0.1/1.5 defaults would squash a mass grid
  into ~[-1.0, -0.29].
- **`GAP_SUCCESS_ONLY=0` for mass** (default flips per channel). Mass strongly
  affects success, so keeping only successes conditions on an effect of the swept
  variable -- a collider that deletes precisely the episodes where mass mattered.
  Filter at train time with `FRIC_SUCCESS_ONLY` instead.
- **Do NOT trim buckets to equal episode counts here** -- see
  `balanced_episode_split`'s docstring and the census. Raw frames are balanced to
  1.06x; trimming makes them 2.62x and discards 43% of them.
- `action_mode` and `success_only` are now in the run-directory name. They were
  not, so a `command` run and a `realized_gain` run both landed on
  `transformer_none` and the second silently overwrote the first.

### Result: mass conditioning buys ~0 on the OBJECT, and the cause is a low ceiling

Transformer, 20 epochs, 2100 episodes, `obj_weight=9`, `mv:object` mm, 2 seeds
per arm. `FRIC_TRIM=0` (raw frames are already balanced, see above).

| action input | mass reaches `none` via | `none` | `both` | gain |
|---|---|---|---|---|
| `realized_gain` | teacher's gains (R2 +0.30) | 13.41 | 13.42 | +0.01 |
| `realized` (leak-free) | nothing (R2 -0.04) | 17.65 | 17.61 | -0.04 |

Seed spread is 0.02-0.10 on `realized_gain` but **0.35-0.53 on `realized`**, and
the leak-free gain REVERSES SIGN between seeds (-0.13, +0.05). Never quote a
single-seed number from the `realized` arm.

**The teacher DOES leak mass, and it is not the explanation.** All of this
replicates (`_probe_mass_leak.py`, `_probe_none_knows_mass.py`): the teacher sees
privileged `phys_params` and encodes mass in its commanded joint stiffness
(gains R2 +0.30; realized path -0.04; object motion -0.05); the unconditioned
model rebuilds mass internally at R2 0.45, collapsing to 0.08-0.15 once the gains
are withheld. It simply does not convert into a conditioning gain.

**The cause is a LOW CEILING** (`_probe_mass_ceiling.py`). Grant an ORACLE
per-mass-bucket rescale -- strictly more mass information than any conditioning
scheme could exploit through the magnitude channel -- and it buys **0.05%**
(leaky) / **0.09-0.14%** (leak-free). Mass moves GT track magnitude 55.7 -> 41.3 mm
(-26%) across the grid, but every model already predicts that trend: the
calibration ratio |pred|/|gt| is flat across buckets even for the leak-free
`none` (0.93-0.96) whose internal mass R2 is only 0.15. **Knowing the mass is not
required to predict the magnitude -- the arm's 16-step path plus the object pose
already constrain it.** What is left is DIRECTION error (~17 mm on ~50 mm tracks),
which mass does not govern; the open-loop decomposition says the same thing
independently (direction is chaos in mass and friction alike).

**Controls, both clean.** Shuffled mass labels -> substitution delta 0.00 per
bucket: the model learns to IGNORE a meaningless input, so the real effect is not
an artefact of the extra token/parameters. 8 held-out masses -> the substitution
ramp reappears on values never trained on, and stronger (+0.89 vs +0.74), so it
learned mass -> dynamics rather than 32 memorised buckets.

### It is OUTCOME vs INTENT, not the leak (`command_pose`, 6-D)

`FricTrackModel` adds a fifth action mode, `command_pose`: the policy's 3 dpos +
3 dori and nothing else -- causally upstream of contact AND leak-free (mass ridge
R2 -0.019). Added in `models/fric/`; `models/gap/` stays unedited. **The
`action_xyz` override is load-bearing**: the base class only integrates commanded
deltas when `action_mode == 'command'` exactly and otherwise falls through to the
REALIZED path, which would feed the realized trajectory back in through RoPE,
silently. Guard test: perturbing `action_seq` must move the prediction by exactly
0.0 in this mode.

| action input | realized path? | gains? | `none` | `both` | gain | subst. (heavy) |
|---|---|---|---|---|---|---|
| `realized_gain` | yes | yes | 13.41 | 13.42 | +0.01 (0%) | +0.21 (+0.74) |
| `realized` | yes | no | 17.65 | 17.61 | -0.04 (-0.2%) | +0.25 (+0.82) |
| `command` (20-D) | no | yes | 17.06 | 16.60 | **-0.46 (-2.7%)** | **+1.43 (+5.50)** |
| `command_pose` (6-D) | no | no | 22.26 | 21.69 | **-0.57 (-2.6%)** | **+1.54 (+6.68)** |

**The gains are irrelevant to whether conditioning pays; the realized path is
everything.** 20-D and 6-D gain identically (~2.6%) despite differing by exactly
the 14 leaking gain dims. So the earlier hypothesis -- that the teacher's gain
leak suppresses the effect -- is WRONG, and so was the guess that the realized
EEF path leaks mass (R2 -0.043, it does not).

The dissociation is the point:
- **gains** carry mass AS A NUMBER (R2 +0.30) yet explicit mass still helps 2.7%
  on top of them;
- **the realized path** carries NO recoverable mass (R2 -0.043) yet makes explicit
  mass worthless.

Knowing the mass and knowing the OUTCOME are not interchangeable. The realized
path does not say what the object weighs, it says what the weight already DID --
strictly more useful, and it makes the parameter redundant.

Reliance follows the same split, seed-free (same model, same frames): removing
mass costs +1.4-1.5 mm in the commanded arms vs +0.21-0.25 in the realized ones,
7-9x, with a monotone ramp (+0.23 / -0.16 / +0.56 / **+6.68**) that is flat where
mass saturates and huge where it governs.

**Caveat: the 2.6% is ONE SEED per arm.** Substitution is large and seed-free but
measures RELIANCE, not VALUE -- a model never given mass learns other routes and
ends up only 2.6% behind. Two independent arms agreeing at 2.6% is supporting
evidence, not replication. `mv:arm` also explodes to ~21 mm (6-D) / ~15 mm (20-D)
from ~3.7: without the realized path the arm's own future is no longer given.

### At 12K episodes the gain more than doubles, and FiLM is the WHOLE conditioning path

The 2.6% above was measured on 2100 episodes and **understated the effect**. Same
`command` 20-D arm, same val episodes, 12,101 episodes across 6 files
(`tracks_m_{a,b}` + `tracks_m64_{0,1,b0,b1}`), `mv:object` mm:

| episodes (train/val) | `none` | `both` | gain |
|---|---|---|---|
| 2,100 (1,681 / 419) | 17.06 | 16.60 | -0.46 (**2.7%**) |
| **12,101 (11,682 / 419)** | **11.96** | **11.20** | **-0.76 (6.4%)** |

Absolute error falls 32.5% for 7x the TRAINING episodes (matching the ~N^-0.20
exponent from the gap section), and the CONDITIONING gain more than doubles. Same
lesson as the two failed structural fixes in the gap section: a conditioning
ablation run in an overfitting regime understates itself.

**The prefix token contributes nothing; FiLM does all of it.** Third arm,
`phys_cond=film` (FiLM on every encoder/decoder FFN, no extra sequence token):

| cond | mv:object | mv:arm | subst. no-info | subst. shuffled | params |
|---|---|---|---|---|---|
| `none` | 11.96 | 10.80 | - | - | 19,196,982 |
| `both` (token+FiLM) | 11.20 | 10.31 | +2.22 | +3.64 | 22,613,046 |
| **`film`** (FiLM only) | **11.18** | 10.31 | +2.20 | +3.64 | 22,612,534 |

`both` is `film` plus **512 parameters** -- exactly one `d_model` token embedding --
and buys 0.02 mm. Prefer `phys_cond=film`: it drops a sequence slot and the
`n_prefix` bookkeeping (2+H vs 3+H), which is the most silent-failure-prone detail
in this pipeline.

- **Do not quote the 0.02 mm as a measured difference.** Both arms ran
  `FRIC_SEED=0` on identical data and val ids, so they share an init/batch-order
  stream and are correlated by construction. The tie is solid; its precision is
  not. The seed-free part is substitution, and it agrees across the two arms to
  0.02 mm (no-info) and exactly (shuffled).
- A `token`-only arm was cancelled at epoch 9 of 20 (val 0.06903 vs film's 0.06681).
  **Mid-`OneCycleLR` and NOT comparable** to annealed runs; its ckpt is parked at
  `output/mass/ckpt/_killed_token_ep9` so it cannot be swept up in a comparison.
- **`fan / err = 4.13`** on the overlay videos (`output/mass/vizfilm_succ/`, 12
  successes, 6 train + 6 val): sweeping the mass input moves the prediction ~4x
  further than the model's residual error. The OPPOSITE of friction, where
  `visualize_fric_conditioning.py`'s own docstring warns a fan much smaller than
  the error is normal. Per-episode train 10.8 vs val 20.4 mm = **1.89x**, the
  overfitting gap visible one clip at a time.

Two gotchas this run depends on, both silent:

- **`FRIC_VAL_POOL` does NOT pin the val set; use `FRIC_VAL_IDS_FROM`.** It caps
  val ids at N, but the per-bucket permutation still runs over ALL ids, so
  appending a file moves the subset (measured: 94/419 = 22% overlap) and the
  "scaling" comparison scores two models on different episodes.
  `FRIC_VAL_IDS_FROM=<ref ckpt>` loads that ckpt's `val_ids` and asserts the
  reference `paths` are a prefix of the current list.
- **`FRIC_DINO_DIM=64` is now mandatory on the mass files.** `tracks_m_{a,b}` were
  narrowed 1024 -> 64 on disk (`dywa/exp/train/shrink_dino.py`, 215 GB -> 27 GB);
  the 1024 default raises a shape error in `ProjectedDino` rather than failing
  quietly. Consequence: checkpoints WITHOUT a `_dino64` suffix can no longer be
  re-evaluated or re-visualised -- no 1024-d features exist. `image` was kept, so
  the overlay videos still render.

### Two traps that produced wrong intermediate conclusions here

- **Substitution OVERSTATES the value of the conditioning input.** Removing mass
  from a conditioned model costs +0.21..+0.82 mm, yet a model never given mass
  MATCHES it. Substitution measures RELIANCE on an in-distribution input --
  feeding a constant is OOD for a model trained to trust it. Only a seeded
  between-model A/B measures value. Use substitution for the SHAPE question (is
  the effect concentrated where the physics is?) and never for the size.
- **The aggregate is ~90% arm and inverted the conclusion AGAIN.** The leak-free
  arm's val loss improved 3.5%. That is the ARM (`mv:arm` 5.20 -> 4.29, -17%,
  consistent in both seeds): mass substituting for the withheld gain dims, i.e. a
  statement about the CONTROLLER, not object physics. On the object it was
  -0.04 +- 0.4. Read `mv:object`.

### Open-loop gate: decompose the fan, do not read its size

`_track_fan` (in `_probe_fric_openloop_replay.py`) reports the across-env spread
of the GT 16-step tracks **in mm/step, the unit the model is scored in**, and
splits it into the part the swept value explains and contact chaos
(`fan^2 = systematic^2 + residual^2`). Three pushes per channel:

| channel | total fan | systematic | % var (chance 3.2%) | magnitude ratio |
|---|---|---|---|---|
| mass | 70 / 21 / 43 mm | 18.8 / 5.3 / 20.5 | 7.3 / 6.3 / 22.8 | **0.56, 0.47, 0.42** |
| table friction | 38 / 39 / 47 mm | 6.7 / 8.2 / 16.8 | 3.1 / 4.5 / 12.5 | 1.14, 0.91, 1.08 |

**Raw fan size rates the two channels as equals; they are not.** Mass halves the
track magnitude in every push; friction's trend straddles 1.0 -- a large fan made
of chaos, which no conditioning input can recover. Direction is chaotic in both.
Net endpoint displacement (the probe's old headline, "627 mm apart") is not
comparable to a ~15 mm model error and must not be used as the gate.


## Joint action+track policy with a meta-learned belief (`models/policy/`)

The track model promoted to a POLICY. Two models, **no shared weights**, bridged
only by an 8/32-d belief:

- **inner** `models.fric.LatentTrackModel` -- its object-only track error on
  already-ELAPSED windows defines `dL/dq`; `meta_ttt.adapt` takes K
  unit-normalised steps.
- **outer** `models.policy.JointPolicyDiT` -- GAPPolicy's joint denoiser,
  de-bimanualised: context encoder over `[CLS, state, 391 patches]` and one
  self-attention stack over `[16 action, 391 fused track, 1 state, 1 timestep]`
  = 409 tokens at ONE shared timestep. Hand-rolled DDIM (no `diffusers` here),
  cosine `alpha_bar` imported from `gap.TrackModel` so the schedules cannot drift.

Design figures: `output/mass/fig_joint_pipeline_clean.png` (method) and
`fig_joint_pipeline.png` (same, annotated with the traps).

```bash
# 0. ALWAYS run the smoke gate first -- 14 checks, all for SILENT failures
python3 dywa/exp/train/_probe_policy_smoke.py

# 1. HEADROOM GATE, no inner loop. A latent can only recover what the TRUE
#    value carries, so if `mass` does not beat `none` on val/action there is
#    nothing for a belief to infer.
POLICY_EVAL_DATA=output/mass/tracks_m_a.hdf5,output/mass/tracks_m_b.hdf5 \
POLICY_VAL_IDS_FROM=output/mass/ckpt/meta_z8_stream_r3/best.ckpt \
  bash dywa/exp/scripts/train_policy.sh none 0     # and `mass` on GPU 1
#    negative control: POLICY_PHYS_SHUFFLE=1 with cond=mass

# 2. meta-training (second order; METAP_DETACH_Q=1 for FOMAML)
METAP_EVAL_DATA=... METAP_VAL_IDS_FROM=... \
METAP_INIT_POLICY=output/mass/ckpt_policy/none/best.ckpt \
  bash dywa/exp/scripts/train_meta_policy.sh 0

# 3. eval: blind / ttt / ceil / shuf on the pinned 419
METAP_CKPT=.../best.ckpt METAP_K_EVAL=1,3,5 \
  bash dywa/exp/scripts/eval_meta_policy.sh 0
```

**The context encoder carries NO action tokens** (`n_prefix = 2`). In
`gap.TrackModel` the next H actions ARE encoder tokens and are worth ~50x (val
2.008 -> 0.038 without them); here the action is what the model must produce, so
its only path in is the noised action token inside the DiT. `_probe_policy_smoke.py`
asserts the context memory is bit-identical under an `action_cmd_seq`
perturbation -- a token-layout edit would otherwise break it silently.

**FiLM reads the RAW vector, `cond_dim = z_dim`, no embedding MLP**, as the
pick-bar reference does (`ttt_with_dyn_bar.py:268,344,119`). Measured at the real
init, the embedded route buys nothing that matters: `d(gamma)/dq` within 13% of
direct, for 62x the parameters (3.42M vs 55k at z=8; 204k at z=32 here). The
`direct_film=True` flag on `LatentTrackModel` is OFF by default -- `film_dim` is
baked into every layer's shape, so flipping it makes r3-era checkpoints
unloadable (verified: they still load `strict=True` with the default).

Gotchas, each of which fails silently:

- **`STREAM_KEEP_SHARDS=1` is the default in both wrappers, deliberately.**
  `ShardWindow` DELETES the shard it evicts. Two arms of the gate read ONE static
  shard directory, so either would delete the other's data. Set 0 explicitly only
  when a collector is live and bounded disk is the point. (Cost of learning this:
  5 shards, 9 GB.)
- **Pin `noise_seed` for every pre/post or A/B.** The outer loss is stochastic in
  `(t, eps)` and the effects are ~1%, so an unpinned draw is larger than the
  signal. `meta_policy.policy_err` keys the seed to the ANCHOR so pre and post of
  a group see identical noise.
- **Read `val/action`, never `val/loss`.** The total is the action MSE plus a
  track term that is ~87% arm, and the arm's track is the action's own
  consequence. This project has inverted a conclusion twice on a ~90%-arm
  aggregate.
- **`METAP_AUX` (on by default) is load-bearing.** Without the inner model's own
  supervised loss, its ONLY gradient is the meta path, which does not require the
  inner loss to mean anything -- the degenerate optimum is a constant `dL/dq`,
  i.e. `q_k ~ q0 + const`, which is the inert-adaptation failure this program
  already hit once.
- **Multi-step DDIM DESTROYS the action prediction; 1 step is the best readout.**
  Measured on 246 pinned val frames, raw action units, ratio = sampled MSE /
  predicting the dataset mean (`_probe_action_snr.py`):

  | DDIM steps | pre-fix ckpt (285k) | standardised ckpt (45k) |
  |---|---|---|
  | **1** | **0.826** | **0.865** |
  | 2 | 0.848 | 0.880 |
  | 5 | 1.070 | 1.183 |
  | 10 | 1.163 | 1.399 |
  | 25 | 1.298 | 1.594 |

  At 1 step -- a single context-only x0 prediction from pure noise -- the model
  BEATS the mean. Every refinement step makes it worse, monotonically, past 1.0
  by 5 steps. So the action denoiser is **not** a mean-collapsed predictor: it
  has real skill and the sampler injects error faster than it removes it. The
  same signature is recorded on the track branch at `track_model.py:489`
  (74.6 mm at 10 steps -> 129.2 mm at 50 without the clip), so it is a property
  of this `prediction_type='sample'` + eps-recovery loop, not of one branch.
  **Do not read a sampled number without stating the step count**, and do not
  assume more steps is a better estimate of model quality -- here it is strictly
  a worse one.
- **Even at 1 step the policy is WEAK, and that is the binding constraint.**
  Same 246 frames, `none_m0.1-0.5_fixnorm` at 45k, ratio vs predicting the mean:

  | predictor | ratio | note |
  |---|---|---|
  | dataset mean | 1.000 | - |
  | ridge on pooled context (`_probe_action_floor.py`) | 0.966 | the LINEAR floor |
  | **DiT, 1 step** | **0.860** | 4x the ridge's improvement |
  | DiT, 10-step DDIM | 1.399 | worse than the mean |

  So the DiT is NOT failing to use its capacity -- it beats a linear map on the
  same inputs by 4x. It is simply not a competent policy: 14% better than the
  dataset mean cannot drive the task (teacher success 0.817, policy/blind 0.034).
  **A sampler fix is capped by the 1-step number, so fix the predictor first.**
  Per dimension group, since the 20 dims are not one quantity:

  | dims | ratio | share of action variance |
  |---|---|---|
  | dpos 0:3 | 0.823 | 26.9% |
  | dori 3:6 | 0.770 | 21.1% |
  | gains 6:20 | **0.914** | **52.1%** |

  The gains are half the variance and nearly unpredictable, so the aggregate
  understates the path dims -- but only 0.86 -> 0.80, not a conclusion-inverting
  gap like the 90%-arm cases above. Read the split; do not expect it to rescue
  the headline.
- **The action target must have UNIT STD** -- a real defect, but NOT the cause
  of the above, and fixing it BOUGHT NOTHING MEASURABLE. Both arms run to
  completion (pre-fix 285k, standardised 300k), 1-step, same 246 frames:

  | dims | pre-fix | standardised | var share |
  |---|---|---|---|
  | aggregate | **0.826** | **0.840** | - |
  | dpos 0:3 | 0.796 | 0.794 | 26.9% |
  | dori 3:6 | 0.784 | 0.774 | 21.1% |
  | gains 6:20 | 0.859 | 0.890 | 52.1% |

  Every difference is 1-3% with NO seed replicate, so the sign is not
  established in either direction. Note the aggregate inverts the control dims
  for the THIRD time in this project: it says the fix lost 1.7%, while both
  dims that actually move the robot came out marginally ahead and the whole
  aggregate gap is the gains (52% of variance). The fix is kept because the
  SNR argument is correct on first principles -- the action branch now has the
  track branch's schedule -- not because it was shown to help.
  `x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps` mixes the target with unit-variance
  noise, so the target's std IS its signal-to-noise. GAP's `fit_limits`
  ([min,max] -> [-1,1]) is right for GAP's bounded EEF positions and WRONG for
  DyWA's `action_cmd`, an unsquashed policy mean with 36% of raw values outside
  [-1,1]: the limits are set by outliers and the bulk compressed to std **0.32**
  against the track branch's 1.02. Measured consequence -- SNR > 1 on only the
  first **19** of 100 cosine timesteps (track: 49), so 81% of action gradient
  steps saw a target buried in noise. `fit_action_norm` now uses
  `FieldNorm.fit_standard`. This was diagnosed from the monotone-worse symptom
  and the symptom SURVIVED the fix -- scale and the sampler are two separate
  defects, and the loud one was the sampler.
  - The *denoising loss* is low partly because the target's variance was low;
    `val/action` fell 0.03603 -> 0.02953 over 15k-45k pre-fix while the sampled
    ratio sat above 1.0. A falling denoising loss is not evidence of a working
    generator.
  - **`val/action` cannot detect this** and is not comparable across the fix --
    a normalised MSE moves by the square of the scale (~9x). Score with
    `_probe_action_snr.py`: full DDIM sampling vs predicting the dataset mean,
    in **raw action units**. `ratio >= 1` means the sampler knows nothing the
    mean does not. Pre-fix: **1.175**.
  - `FieldNorm.scale/offset` are PERSISTENT buffers, so every `load_state_dict`
    silently restores the checkpoint's normaliser over a fresh fit (same trap as
    `track_rms`). `train_meta_policy.py` now asserts the live action scale
    matches the fit and names a ~1/3 ratio as a pre-standardisation ckpt. Such a
    ckpt is **not a warm start** -- its output head was trained to emit std-0.32
    values. `_prefix_actnorm_metapolicy_z32_k1_live` is parked for this reason.
  - `clip_sample_range` only became live at this point: under `fit_limits` the
    normalised action never left [-1,1] so the clip never bound, which is why a
    `clip in {0,1,1.5,2,10}` sweep read as a null. Standardised actions reach
    **4.1**, so the default 10.0 still clears them -- verify, do not assume.
- **Teacher-forced track eval is OFF-DISTRIBUTION.** GAP's shared timestep means
  "action clean, track noisy" never occurs in training. `sample(pin_action=True)`
  still gives the closest comparable to an action-conditioned track model, but
  report it as an approximation. Independent per-channel timesteps would make it
  exact and are a deliberate deviation from GAP, not done.
- **Support and query come from ONE trajectory 16 frames apart**, so the belief
  can encode pose rather than physics. `METAP_SHUFFLE_Q=1` / the eval's `shuf`
  column is the control, and it must be clearly WORSE than blind.
- `z_dim = 32` against `d_model = 512` is 1/16; the bar runs 64 against 128 (1/2).
  The unit-normalised step makes the belief `z_dim - 1` DOF of DIRECTION, so wider
  is also more room to memorise episode nuisance.

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

All outputs should be extremely plain and extremely concise!

## Friction-conditioned track models (`dywa/src/models/fric/`, `output/fric/`)

Answers: **does varying a hidden dynamics variable degrade track prediction, and
does handing the model the ground-truth value recover it?** Sweeps ONE friction
channel over 32 discrete values (one per env) with every other physics variable
pinned at the `GAP_FIX_PHYS=fixed` defaults, so the dataset differs from
`output/gap/tracks_act_*.hdf5` in exactly one variable.

New files only; the `gap` pipeline is imported, never edited.
`FricTrackModel` subclasses `gap.TrackModel` (`phys_cond` = none|token|film|both);
`phys_cond='none'` is **bit-identical** to the gap model, so the baseline arm is
the model whose 14.7 mm number it is compared against.

```bash
# collect (2 GPUs, 1250 eps each, ~45 min, ~110 GB per file)
GAP_NUM_EPISODES=1250 GAP_SEED=0 GAP_OUT=output/fric/tracks_tf_a.hdf5 \
  bash dywa/exp/scripts/collect_fric_tracks.sh 0
bash dywa/exp/scripts/train_fric_track.sh transformer both 0   # none|token|film|both
bash dywa/exp/scripts/eval_fric_track.sh output/fric/ckpt/transformer_both 0
```

`FRIC_CHANNEL=table|object`, `FRIC_LO/HI` (default 0.1/1.5), `FRIC_PHYS_SHUFFLE=1`
(control), `FRIC_HELDOUT_MU=8`, `FRIC_EPOCHS` (20; see below). WandB project
`dywa-fric-track`.

### Result: conditioning works, and buys ~1%

Transformer, `realized_gain`, `obj_weight=9`, 20 epochs, 2500 episodes
(balanced split -> 1216 train / 288 val), `mv:object` mm, 2 seeds:

| run | mv:object | vs none |
|---|---|---|
| `none` | 15.09 +- 0.07 | - |
| `both` (true mu) | **14.94 +- 0.03** | **-0.15 (1.0%)** |
| `both_shuf` (control) | 15.05 (1 seed) | -0.04 |

**The model demonstrably uses friction as PHYSICS.** Seed-free evidence, three
ways, all pointing the same direction:

- **mu-substitution is U-shaped in mu.** Cost of replacing true mu with the grid
  mean, same model, same frames: **+0.50 / -0.02 / +0.06 / +0.23 mm** across
  mu = 0.1-0.4 / 0.5-0.8 / 0.8-1.1 / 1.2-1.5. Largest where friction is most
  extreme, ~0 in the middle -- exactly the physical prediction. **Two seeds agree
  to 0.03 mm on every bucket.**
- **It generalises in mu.** With 8 interior values held out of training entirely,
  the same U-shape appears on those unseen values (+0.28/+0.01/+0.04/+0.31), so
  it learned mu -> dynamics, not 32 memorised buckets.
- **The embedding is clean**: 32 distinct vectors, PC1 = 92% of variance, ordered
  by mu at **r = +0.999**; FiLM gamma varies with mu in all 6 layers.
- **The shuffled control self-validates**: it learned to IGNORE its meaningless
  input (substitution delta -0.03 / -0.00 mm).

**Why the net gain is nonetheless only 1%.** The unconditioned model predicts the
mu-MARGINAL, which is nearly as good as the conditional, because friction's
contribution to a 16-step track is small next to push geometry. Note the
asymmetry: giving the conditioned model WRONG mu costs +0.44 mm, more than the
unconditioned model's total ignorance costs (+0.15 mm) -- it has committed to
trusting the input.

### Gotchas, each of which cost a run here

- **`docker exec` does NOT inherit host env.** `export FRIC_EPOCHS=20` on the host
  silently ran 10 epochs (the script default). Pass vars INSIDE the exec string.
  The tell: epoch-0 val differs (0.240 at 20 epochs vs 0.197 at 10) because
  OneCycleLR spans `EPOCHS*steps`.
- **20 epochs, not 10.** Absolute error improves a lot (none 16.08 -> 15.15) AND
  the per-bucket structure only emerges at 20; at 10 the profile looked flat.
- **Between-model seed spread is 0.13 mm on `none`** -- the SAME SIZE as the
  effect. The 0.05 mm figure in the GAP section is from a different, smaller
  dataset and does not transfer. Never compare two separately-trained models here
  without a seed replicate; prefer the within-model substitution test, whose
  seed term is ~0.03 mm.
- **The balanced split discards ~40% of episodes** (buckets run 47-103; trimmed to
  47). That imbalance is itself the friction effect -- high-mu episodes run ~17%
  longer so those envs finish fewer. Both arms are affected identically.
- Friction must be **baked pre-`prepare_sim`**; reset-time DR is inert on GPU and
  its readback is fiction. See [[fric-grid-bake-verified]].

### Screening probes (run these before any similar experiment)

`_probe_fric_track_headroom.py` (ridge + positive/negative controls),
`_probe_fric_marginal.py` (episode-level), `_probe_fric_embedding.py`
(is the learned vector ordered?), `_probe_fric_census.py` (grid/pin gate).

**A ridge-regression screen said NULL and was WRONG about the sign of the
conclusion.** It reported delta R2 = -0.001 with a +0.15 positive control across
12 cells, and I nearly stopped there. It excluded object ROTATION (much of what a
pushed bottle does, and all of it in the tracks) and fed the regression the
object's velocity HISTORY, from which mu is partly inferable. Both flatter the
null. **A proxy that nulls does not license skipping the real A/B** -- run the
model.

## Streaming (online) collection + training (`*_stream_tracks.py`)

Answers the overfitting problem, not a new science question. The 12K mass model
sits at **val 11.18 mm vs 6.39 mm on data it trained on -- 1.751x aggregate /
1.738x per-episode**, and a second trained-on set agrees to 0.01 mm (1.752x) (measured by `_probe_stream_baseline.py`; do NOT use the
1.89x quoted in the mass section, that came from 12 hand-picked successful
episodes via `visualize_fric_conditioning.py` and is a different measurement).
Train error has been FLAT at ~7-8 mm across every data scale while only val
falls, as `N^-0.20`. More unique data is the lever, and it is blocked by
**storage, not compute**: 4.3x more data is ~480 GB against ~440 GB free, and
the 32x that would halve the error is ~3.5 TB.

Streaming makes that a compute problem: generate shards, train on a FIFO window
of the recent ones, delete the rest. Bounded disk, unlimited unique frames.

```bash
# GPU 1: collector, runs until killed. START THIS FIRST.
STREAM_CHANNEL=mass STREAM_DIR=/home/user/DyWA/output/mass/stream \
STREAM_EVAL_DATA=/home/user/DyWA/output/mass/tracks_m_a.hdf5,/home/user/DyWA/output/mass/tracks_m_b.hdf5 \
GAP_SEED=100 bash dywa/exp/scripts/collect_stream_tracks.sh 1

# GPU 0: trainer, warm-started. STREAM_EVAL_DATA + STREAM_VAL_IDS_FROM required.
STREAM_DIR=/home/user/DyWA/output/mass/stream \
STREAM_EVAL_DATA=/home/user/DyWA/output/mass/tracks_m_a.hdf5,/home/user/DyWA/output/mass/tracks_m_b.hdf5 \
STREAM_VAL_IDS_FROM=/home/user/DyWA/output/mass/ckpt/transformer_both_command/best.ckpt \
STREAM_INIT_FROM=/home/user/DyWA/output/mass/ckpt/transformer_film_command_dino64/best.ckpt \
bash dywa/exp/scripts/train_stream_tracks.sh transformer film command 0

# resume a run that has not converged (this is why the LR is warmup+constant,
# not OneCycle -- see below)
STREAM_RESUME=1 ... bash dywa/exp/scripts/train_stream_tracks.sh transformer film command 0
```

**NO DAgger, and that is not an omission.** `train_rma.py` needs it because the
student ACTS, so its own mistakes change the states it later sees. A track model
is a passive predictor -- it never acts -- so the teacher alone fixes the data
distribution and there is **no on-policy requirement**, which is what makes frame
reuse from a window legitimate. What is wanted is streaming/infinite data, not
online RL. Only the *shape* of `train_rma.py:574-580` is borrowed: one unbounded
loop, step counts instead of epochs.

**Channel-neutral.** `STREAM_CHANNEL=mass|table|object`; range/spacing/col are
ADOPTED from `STREAM_EVAL_DATA`'s own attrs (`_adopt_reference_range`), never
hardcoded. This matters because the per-channel Python defaults are NOT what the
corpora used -- mass defaults to (0.05, 1.0) while `tracks_m_*` used [0.1, 1.2],
so trusting the defaults silently builds shards on a different grid.

New files only. `collect_gap_tracks.py`, `collect_fric_tracks.py`,
`train_fric_track.py` and `models/{gap,fric}/*` are untouched.

### Verified, with the checks that did the work

- `_probe_stream_equivalence.py` is **THE GATE**: all 9 model keys bitwise
  identical between the pooled offline dataset and the per-shard
  `ConcatDataset`, over a 463,516-frame pair, plus FIFO order / eviction /
  deletion / unmarked-shard rejection. **15/15 on synthetic AND real shards.**
- `_probe_stream_rotation.py`: rotation, attrs, `.done` markers, no frame loss,
  orphan sweep. **14/14, no simulator needed.**
- `_verify_gap_tracks.py` + `_probe_fric_census.py` pass on real shards.
- A stream checkpoint evaluates through **`eval_fric_track.sh` unmodified**
  (11.17 mm), because the ckpt is a strict superset of the offline schema with a
  synthesised `epoch` and `paths` pointing at the frozen corpus, not the shards.

### Gotchas, each of which cost a cycle here

- **The two mass checkpoints carry DIFFERENT `track_rms`** -- 0.031895
  (`transformer_both_command`) vs 0.031571 (`transformer_film_command_dino64`),
  because each refit `fit_track_norm` on its own 800-frame draw. And
  `FieldNorm.scale/offset` are PERSISTENT buffers, so they ride inside
  `state_dict` and `load_state_dict` silently overwrites whatever you set. So
  `track_rms` must come from whichever ckpt supplies the WEIGHTS; the trainer
  asserts the live buffer equals the scalar it will write.
- **`eval_fric_track.py` could not evaluate ANY `_dino64` ckpt** -- its
  `_ProjectedDino` projected unconditionally, so a pre-projected 64-d file raised
  `mat1 and mat2 shapes cannot be multiplied (391x64 and 1024x64)`. Latent until
  `tracks_m_{a,b}` were narrowed on disk. Fixed with the same width guard
  `train_fric_track.py` always had; it now reproduces 11.18 exactly.
- **`STREAM_NUM_ENV` must equal the corpus grid size**, and raising it would not
  help anyway. `num_env` IS the grid (`collect_fric_tracks.py:170`) and
  `_bake_channel` computes it internally, so there is NO seam for a tiled grid;
  the driver asserts and fails in seconds instead of at train time. Measured at
  a matched step count: **32 envs -> 53.8 frame/s, 48 envs -> 48.5**, because
  the step rate collapses (3.07 -> 2.06, x0.67) faster than per-step frames rise
  (x1.5). So the "32 is the knee, work is per-FRAME" finding **survives DINO-64
  and image-less shards** -- the guess that the knee had moved up was wrong, and
  grid tiling is unnecessary.
- **`RotatingEpisodeWriter` must not define `close`**: `collect_fric_tracks.main()`
  assigns `G.EpisodeWriter.close` to add the `fric_grid`/`dino_*` attrs and would
  clobber it, leaving shards `read_grid` refuses.
- **`gap/window_vs_val` is meaningless until `window/turnovers >= 1`.** At block
  0 the resident shards have been sampled ~0 times, so `window` is a SECOND
  held-out set and the ratio is set difficulty. Measured live on a deliberately
  tiny 3-shard window it CLIMBS 1.40 -> 2.14x as reuse goes 0.27 -> 0.91, i.e.
  the model memorises the resident set. **Window size therefore controls
  memorisability, not just decorrelation: if the gap does not close, ENLARGE THE
  WINDOW, do not add steps.** Primary metric is `val_full/mv_object` below 11.18.
- `val/mv_object` (quick subset) and `val_full/mv_object` (the 419-episode pass,
  comparable to 11.18) are **separate keys on purpose** -- sharing one made a
  sawtooth that read as instability.
- **Reuse is emergent** (consumption/production) and must be READ, not assumed.
  Smoke test: 341-345 f/s consumed, ~75 f/s produced -- but that window was
  page-cache-resident and the collector had a competing reader, so neither is
  the real figure. Measure at `WINDOW_SHARDS=30`, `BLOCK_STEPS=20000` before
  committing to a long run. `collect/blocked_s` is logged because a collector
  blocked on `STREAM_MAX_SHARDS` stops producing unique data.
- **The opening ~2.3 h of a run is a DIFFERENT regime.** `STREAM_MIN_SHARDS=4`
  starts the trainer on 800 resident episodes; the window only reaches its
  designed 6000 (30 shards) after ~29 more land. That early stretch is exactly
  the small-window regime that inflates the gap, so read nothing before
  `window/turnovers >= 1`. Raising `STREAM_MIN_SHARDS` toward
  `STREAM_WINDOW_SHARDS` makes early and late the same experiment, at the cost
  of idling the trainer while the window fills.
- **Collector and trainer on DIFFERENT GPUs.** Sharing one turns the full-val
  probe from well under a minute into several.
- `GAP_STORE_IMAGE=0` on shards: no overlay videos and no
  `visualize_fric_conditioning.py` on stream data, so `fan/err` can only ever be
  recomputed on the frozen corpus.
- A bare `Traceback (most recent call last):` with no body is a DataLoader worker
  torn down by lazy GC, not a crash; `gc.collect()` after dropping the loader
  makes it deterministic.
