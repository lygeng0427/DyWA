#!/bin/bash
# Usage: bash collect_ttt.sh [GPU]
#
# Collect a teacher-rollout dataset (successful episodes only) for offline TTT
# meta-training. Reuses the pretrained bottle teacher.
#
# Prereq: /input/DGN/bottle_train.json must exist (bottle-only object list).
#   jq '[.[]|select(test("-bottle-"))]' /input/DGN/yes.json > /input/DGN/bottle_train.json
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

GPU=${1:-0}

# collector knobs (read from env by collect_teacher_dataset.py)
export TTT_NUM_EPISODES=${TTT_NUM_EPISODES:-2000}
export TTT_MAX_STEPS=${TTT_MAX_STEPS:-20000}
export TTT_OUT=${TTT_OUT:-/home/user/DyWA/output/ttt/dataset_ttt_teacher.pkl}
mkdir -p "$(dirname "$TTT_OUT")"

# Object set + parallelism. Defaults reproduce the current multi-bottle setting;
# override for a single-bottle run, e.g.:
#   TTT_FILTER=/input/DGN/bottle_one.json TTT_NUM_EPISODES=50 \
#   TTT_OUT=.../dataset_one_bottle.pkl bash collect_ttt.sh 1
TTT_FILTER=${TTT_FILTER:-/input/DGN/bottle_train.json}
TTT_NUM_ENV=${TTT_NUM_ENV:-256}

# Physics isolation. With TTT_FIX_PHYS=1, pin every per-episode physics variable
# EXCEPT friction (object + table friction stay randomized) so friction is the
# sole hidden dynamics variable a TTT belief could adapt to. Size is fixed via a
# deterministic min==max scale target; mass and restitution likewise. Initial
# object pose and goal pose still vary (task conditions, not physics). Default
# (unset/0) leaves the full domain randomization untouched.
FIX_PHYS_ARGS=()
if [ "${TTT_FIX_PHYS:-0}" = "1" ]; then
  S=${TTT_FIX_SCALE:-0.08}; M=${TTT_FIX_MASS:-0.3}; R=${TTT_FIX_RESTITUTION:-0.0}
  FIX_PHYS_ARGS=(
    ++env.single_object_scene.min_scale=${S} ++env.single_object_scene.max_scale=${S}
    ++env.single_object_scene.min_mass=${M} ++env.single_object_scene.max_mass=${M}
    ++env.single_object_scene.min_object_restitution=${R}
    ++env.single_object_scene.max_object_restitution=${R}
  )
  echo "[fix_phys] size=${S}m mass=${M}kg restitution=${R} (object+table friction stay randomized)"
fi

PYTORCH_JIT=0 python3 collect_teacher_dataset.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
  ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
  ++env.single_object_scene.filter_file=${TTT_FILTER} \
  ++env.num_env=${TTT_NUM_ENV} \
  "${FIX_PHYS_ARGS[@]}"
