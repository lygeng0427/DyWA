#!/bin/bash
# Usage: [STREAM_* env] bash collect_stream_tracks.sh [GPU]
#
# Collect track shards CONTINUOUSLY for streaming training: one HDF5 per
# STREAM_SHARD_EPISODES episodes, each with a `.done` marker, into STREAM_DIR.
# Runs until killed. Drives collect_stream_tracks.py, which drives the unmodified
# collect_fric_tracks.py / collect_gap_tracks.py.
#
# Channel-neutral: STREAM_CHANNEL=mass|table|object. Range and spacing are ADOPTED
# FROM STREAM_EVAL_DATA's attrs by the python driver, so the shards land on the
# same physics grid as the corpus they will be compared against -- do not set
# them by hand unless you mean to diverge.
#
# The pinned-physics block below is duplicated from collect_fric_tracks.sh
# (that script hardcodes its own python entry point, so it cannot be reused
# without editing it, and nothing existing is edited here). Divergence is not
# silent: train_fric_track.py's `assert_phys_grid` requires every non-swept
# column of phys_params to be constant AND equal to its own PINNED_ALL table, so
# a pin that drifts from that table fails loudly at train time.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1          # vendored DINOv2 deps are absent; fallback is fine
mkdir -p "$TORCH_EXTENSIONS_DIR"

GPU=${1:-0}

export STREAM_CHANNEL=${STREAM_CHANNEL:-mass}
export STREAM_DIR=${STREAM_DIR:-/home/user/DyWA/output/${STREAM_CHANNEL}/stream}
export STREAM_SHARD_EPISODES=${STREAM_SHARD_EPISODES:-200}
export STREAM_MAX_SHARDS=${STREAM_MAX_SHARDS:-45}
export STREAM_DINO_PROJ=${STREAM_DINO_PROJ:-64}
export STREAM_EVAL_DATA=${STREAM_EVAL_DATA:-}
mkdir -p "$STREAM_DIR"

# num_env IS the physics grid (one swept value per env), so this must equal the
# reference corpus's grid size; the driver asserts it. Exported for the driver
# AND passed to hydra -- they must agree.
export STREAM_NUM_ENV=${STREAM_NUM_ENV:-32}
# Varies object poses and goals, NOT the grid (that is a linspace/geomspace,
# identical on every GPU by design). Keep it clear of the seeds the existing
# corpora used (mass: 0/1/10/11/20/21) so we do not regenerate episodes they
# already contain.
GAP_SEED=${GAP_SEED:-100}

export GAP_TRACK_HORIZON=${GAP_TRACK_HORIZON:-16}
export GAP_DINO=${GAP_DINO:-1}
export GAP_MIN_EP_LEN=${GAP_MIN_EP_LEN:-4}
export GAP_MAX_EP_LEN=${GAP_MAX_EP_LEN:-0}
# Success-only defaults ON for friction and OFF for mass: mass strongly affects
# whether the teacher reaches the goal, so keeping only successes conditions on
# an effect of the swept variable -- a collider that preferentially deletes the
# episodes where mass mattered most. Collect everything and filter at train time.
if [ "$STREAM_CHANNEL" = "mass" ]; then
  export GAP_SUCCESS_ONLY=${GAP_SUCCESS_ONLY:-0}
else
  export GAP_SUCCESS_ONLY=${GAP_SUCCESS_ONLY:-1}
fi

GAP_FILTER=${GAP_FILTER:-/input/DGN/bottle_one.json}

# Pinned physics -- same numbers as collect_fric_tracks.sh / GAP_FIX_PHYS=fixed.
FIX_S=${GAP_FIX_SCALE:-0.08}            # object scale (m)
FIX_M=${GAP_FIX_MASS:-0.3}              # object mass (kg)
FIX_R=${GAP_FIX_RESTITUTION:-0.0}
FIX_OF=${GAP_FIX_OBJ_FRICTION:-0.6}
FIX_TF=${GAP_FIX_TABLE_FRICTION:-0.55}
FIX_HF=${GAP_FIX_HAND_FRICTION:-1.25}

PIN_ARGS=(
  ++env.single_object_scene.min_scale=${FIX_S} ++env.single_object_scene.max_scale=${FIX_S}
  ++env.single_object_scene.min_mass=${FIX_M} ++env.single_object_scene.max_mass=${FIX_M}
  ++env.single_object_scene.min_object_restitution=${FIX_R}
  ++env.single_object_scene.max_object_restitution=${FIX_R}
  ++env.franka.min_hand_friction=${FIX_HF} ++env.franka.max_hand_friction=${FIX_HF}
)
# The swept channel's CONFIG range is pinned too. Not redundant with the bake:
# the config draw runs first (scene setup) and the bake overwrites the swept body
# afterwards, so pinning here makes the non-swept body definite and keeps the
# phys_params assertion meaningful. For mass, min/max_mass stay at FIX_M on
# purpose -- a bake that silently failed then shows up as FIX_M in every env and
# the readback assert fires, whereas live config DR would hide it behind 32
# plausible random masses.
case "$STREAM_CHANNEL" in
  table)
    PIN_ARGS+=(
      ++env.single_object_scene.min_object_friction=${FIX_OF}
      ++env.single_object_scene.max_object_friction=${FIX_OF})
    ;;
  object)
    PIN_ARGS+=(
      ++env.single_object_scene.min_table_friction=${FIX_TF}
      ++env.single_object_scene.max_table_friction=${FIX_TF})
    ;;
  mass)
    PIN_ARGS+=(
      ++env.single_object_scene.min_object_friction=${FIX_OF}
      ++env.single_object_scene.max_object_friction=${FIX_OF}
      ++env.single_object_scene.min_table_friction=${FIX_TF}
      ++env.single_object_scene.max_table_friction=${FIX_TF})
    ;;
  *) echo "unknown STREAM_CHANNEL='${STREAM_CHANNEL}' (want: mass|table|object)"; exit 1 ;;
esac

echo "[collect_stream] channel=$STREAM_CHANNEL dir=$STREAM_DIR"
echo "[collect_stream] shard=$STREAM_SHARD_EPISODES eps  max_shards=$STREAM_MAX_SHARDS"\
"  envs=$STREAM_NUM_ENV  seed=$GAP_SEED  dino=$STREAM_DINO_PROJ"
echo "[collect_stream] reference corpus: ${STREAM_EVAL_DATA:-<unset: grid precheck SKIPPED>}"

PYTORCH_JIT=0 python3 -u collect_stream_tracks.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
  ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
  ++env.single_object_scene.filter_file=${GAP_FILTER} \
  ++env.num_env=${STREAM_NUM_ENV} \
  ++env.seed=${GAP_SEED} \
  "${PIN_ARGS[@]}"
