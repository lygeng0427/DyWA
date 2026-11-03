#!/bin/bash
# Usage: ROLL_AGENT=teacher bash eval_rollout.sh [GPU]
#
# Closed-loop rollout success rate. PHASE 1 (teacher) is a HARNESS TEST: it must
# reproduce the corpus's own numbers (in-DR 0.837 / out-DR 0.718) before any
# phase-2 number for our policy means anything.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train
export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$TORCH_EXTENSIONS_DIR"
GPU=${1:-0}
export ROLL_AGENT=${ROLL_AGENT:-teacher}
export ROLL_EPISODES=${ROLL_EPISODES:-320}
export ROLL_MODE=${ROLL_MODE:-blind}
echo "[roll] agent=$ROLL_AGENT mode=$ROLL_MODE episodes=$ROLL_EPISODES gpu=$GPU"
if [ "$ROLL_AGENT" = "policy" ] && [ -z "$ROLL_CKPT" ]; then
  echo "ROLL_CKPT is required with ROLL_AGENT=policy"; exit 1; fi
# Hydra args copied VERBATIM from collect_gap_tracks.sh, plus the same pinned
# physics collect_fric_tracks.sh uses for the mass channel. Anything else and
# the rollout is not the same experiment the corpus came from, so its success
# rate could not be compared against it.
FILTER=${ROLL_FILTER:-/input/DGN/bottle_one.json}
PIN=(
  ++env.single_object_scene.min_scale=0.08 ++env.single_object_scene.max_scale=0.08
  ++env.single_object_scene.min_object_restitution=0.0
  ++env.single_object_scene.max_object_restitution=0.0
  ++env.single_object_scene.min_object_friction=0.6
  ++env.single_object_scene.max_object_friction=0.6
  ++env.single_object_scene.min_table_friction=0.55
  ++env.single_object_scene.max_table_friction=0.55
  ++env.franka.min_hand_friction=1.25 ++env.franka.max_hand_friction=1.25
)
PYTORCH_JIT=0 python3 -u eval_rollout_policy.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
  ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
  ++env.single_object_scene.filter_file=${FILTER} \
  ++env.num_env=${ROLL_NUM_ENV:-32} \
  ++env.seed=${ROLL_SEED:-12345} \
  "${PIN[@]}" "${@:2}"
