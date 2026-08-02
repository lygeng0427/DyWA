#!/bin/bash
# Usage: bash eval_ttt.sh <k_inner_steps> [GPU]
#   k_inner_steps = 0 -> no-TTT baseline (q == q0)
#                   1,2 -> TTT-adapted
#
# Evaluate a meta-trained TTT student on the held-out bottle. Per-object success
# is written by CountCategoricalSuccess under output/test_rma/dywa/result/.
#
# Prereq: /input/DGN/bottle_test.json (held-out bottle) and a meta-trained ckpt.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
# Reduce allocator fragmentation during the TTT inner-loop point-attention.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

K=${1:?usage: eval_ttt.sh <k_inner_steps> [GPU]}
GPU=${2:-0}

# Must match the alpha the checkpoint was meta-trained with, otherwise the belief
# adapts at a different step size than it was trained for. Defaults to the
# alpha_<a>/best.ckpt written by meta_train_ttt.sh for the same TTT_ALPHA.
export TTT_ALPHA=${TTT_ALPHA:-0.1}

# Policy trunk MUST match the checkpoint's, otherwise `strict=False` silently
# leaves the whole policy random. The run-dir naming mirrors meta_train_ttt.py:
# {alpha, policy_trunk, recon_weight} all select the checkpoint directory.
export TTT_POLICY_TRUNK=${TTT_POLICY_TRUNK:-dywa}
export TTT_RECON_WEIGHT=${TTT_RECON_WEIGHT:-5}
if [ "${TTT_POLICY_TRUNK}" = "dywa" ]; then
  RUN_NAME="alpha_${TTT_ALPHA}"
else
  RUN_NAME="${TTT_POLICY_TRUNK}_alpha_${TTT_ALPHA}"
fi
RUN_NAME="${RUN_NAME}_recon${TTT_RECON_WEIGHT}"
# Checkpoint base dir. MUST match TTT_SAVE_DIR used by meta_train_ttt.sh, since
# run_name encodes only {trunk,alpha,recon} — not the object set. A single-bottle
# run trained into a distinct TTT_SAVE_DIR is found by setting the same var here.
TTT_SAVE_DIR=${TTT_SAVE_DIR:-/home/user/DyWA/output/ttt/ckpt}
STUDENT_CKPT=${TTT_STUDENT_CKPT:-${TTT_SAVE_DIR}/${RUN_NAME}/best.ckpt}
# Object set + parallelism. Defaults reproduce the current held-out-bottle eval;
# override for single-bottle eval, e.g. TTT_FILTER=/input/DGN/bottle_one.json.
TTT_FILTER=${TTT_FILTER:-/input/DGN/bottle_test.json}
TTT_NUM_ENV=${TTT_NUM_ENV:-60}

# Physics isolation (see collect_ttt.sh). MUST match the collect setting, or the
# student is evaluated on a different dynamics distribution than it was trained on.
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
echo "[eval_ttt] ttt_alpha=${TTT_ALPHA}  policy_trunk=${TTT_POLICY_TRUNK}  recon=${TTT_RECON_WEIGHT}  ckpt=${STUDENT_CKPT}"

# Optional per-episode video recording. TTT_RECORD=1 saves successful and failed
# episodes into <record_dir>/succ/ and <record_dir>/fail/ (episode_type=both),
# capped at TTT_RECORD_MAX videos per type. Default off → eval behaves as before.
REC_ARGS=()
if [ "${TTT_RECORD:-0}" = "1" ]; then
  REC_DIR=${TTT_RECORD_DIR:-/home/user/DyWA/output/ttt/videos}
  REC_MAX=${TTT_RECORD_MAX:-6}
  REC_USE_COL=${TTT_RECORD_USE_COL:-False}   # False => RGB visual mesh
  REC_GHOST=${TTT_RECORD_GHOST:-True}        # draw object at goal pose
  REC_ARGS=(
    ++use_nvdr_record_episode=True
    ++nvdr_record_episode.episode_type=both
    ++nvdr_record_episode.record_dir=${REC_DIR}
    ++nvdr_record_episode.max_per_type=${REC_MAX}
    ++nvdr_record_episode.use_col=${REC_USE_COL}
    ++nvdr_record_episode.draw_goal_ghost=${REC_GHOST}
  )
  echo "[eval_ttt] recording succ/fail videos -> ${REC_DIR}  (max ${REC_MAX}/type, rgb=$([ "${REC_USE_COL}" = "False" ] && echo yes || echo no), goal_ghost=${REC_GHOST})"
fi

PYTORCH_JIT=0 python3 eval_ttt.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  ++student.ttt.latent_dim=${TTT_LATENT_DIM:-64} \
  ++student.ttt.k_inner_steps=${K} \
  ++student.ttt.ttt_alpha=${TTT_ALPHA} \
  ++student.ttt.policy_trunk=${TTT_POLICY_TRUNK} \
  ++student.ttt.policy_layers=${TTT_POLICY_LAYERS:-4} \
  ++student.ttt.window_k=${TTT_WINDOW_K:-5} \
  ++load_student="${STUDENT_CKPT}" \
  ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
  ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
  ++env.single_object_scene.filter_file=${TTT_FILTER} \
  ++env.single_object_scene.mode=valid \
  ++env.num_env=${TTT_NUM_ENV} \
  "${FIX_PHYS_ARGS[@]}" \
  "${REC_ARGS[@]}"
# NOTE: eval_ttt.py tallies per-object success via CountCategoricalSuccess
# unconditionally (it never reads log_categorical_results, and that key is not in
# eval_ttt.py's Config schema — passing it raises ConfigKeyError). Success rate is
# printed to stdout ("Global average success rate: ...") and saved under
# output/test_rma/dywa/result/. k=0 and k=1 write the SAME dir, so read the rate
# from each run's stdout, not the file (the later run overwrites it).
