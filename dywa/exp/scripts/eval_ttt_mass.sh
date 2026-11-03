#!/bin/bash
# Usage: bash eval_ttt_mass.sh <k_inner_steps> [GPU]
#   k_inner_steps = 0 -> no-TTT baseline (q == q0)
#                   1,2,5 -> TTT-adapted
#
# MASS-varying variant of eval_ttt.sh, on ONE bottle. The eval physics MUST
# match what collect_ttt_mass.sh collected: mass varies per env, everything else
# (including object identity) pinned.
#
#   ckpts   <- output/ttt/ckpt_mass_one_bottle/<run_name>/best.ckpt
#   results -> output/test_rma/dywa/result/dgn-mass-one/  (via TTT_RESULT_TAG)
#
# NOTE ON GENERALIZATION: with one bottle there is no held-out OBJECT, so this
# evaluates on the same bottle that was collected — the held-out variable is
# MASS (each episode resamples it), not shape. That is the right test for "does
# the belief adapt to mass", and deliberately says nothing about cross-object
# generalization. Use the 19/5-bottle split for that (TTT_FILTER + a matching
# collect).
#
# The result tag matters: CountCategoricalSuccess keys its output dir on
# base_set alone, so without it a mass eval would silently overwrite the
# friction eval's per-object results.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}
export PYTHONBREAKPOINT=${PYTHONBREAKPOINT:-0}

K=${1:?usage: eval_ttt_mass.sh <k_inner_steps> [GPU]}
GPU=${2:-0}

# Must match meta_train_ttt_mass.sh — these three compose the checkpoint dir name.
export TTT_ALPHA=${TTT_ALPHA:-0.5}
export TTT_POLICY_TRUNK=${TTT_POLICY_TRUNK:-film_transformer}
export TTT_RECON_WEIGHT=${TTT_RECON_WEIGHT:-5}
if [ "${TTT_POLICY_TRUNK}" = "dywa" ]; then
  RUN_NAME="alpha_${TTT_ALPHA}"
else
  RUN_NAME="${TTT_POLICY_TRUNK}_alpha_${TTT_ALPHA}"
fi
RUN_NAME="${RUN_NAME}_recon${TTT_RECON_WEIGHT}"
# 3D RoPE. MUST match meta_train_ttt_mass.sh: RoPE adds no state_dict keys, so a
# rope-on ckpt loads into a rope-off student silently and would be evaluated as a
# differently-wired network. The run-dir suffix is the only guard — keep this
# block byte-identical to meta_train_ttt.py's run_name construction.
TTT_ROPE=${TTT_ROPE:-0}
TTT_ROPE_OMEGA_MIN=${TTT_ROPE_OMEGA_MIN:-0.4}
TTT_ROPE_OMEGA_MAX=${TTT_ROPE_OMEGA_MAX:-30}
ROPE_ARGS=(
  ++student.ttt.rope=$([ "${TTT_ROPE}" = "1" ] && echo True || echo False)
  ++student.ttt.rope_omega_min=${TTT_ROPE_OMEGA_MIN}
  ++student.ttt.rope_omega_max=${TTT_ROPE_OMEGA_MAX}
)
if [ "${TTT_ROPE}" = "1" ]; then
  RUN_NAME="${RUN_NAME}_rope${TTT_ROPE_OMEGA_MIN}-${TTT_ROPE_OMEGA_MAX}"
fi
# Keep the mass results of the two arms apart, mirroring how the ckpt dirs split.
export TTT_RESULT_TAG=${TTT_RESULT_TAG:-mass-one$([ "${TTT_ROPE}" = "1" ] && echo "-rope")}

TTT_SAVE_DIR=${TTT_SAVE_DIR:-/home/user/DyWA/output/ttt/ckpt_mass_one_bottle}
STUDENT_CKPT=${TTT_STUDENT_CKPT:-${TTT_SAVE_DIR}/${RUN_NAME}/best.ckpt}
# Same single bottle as collect — mass is the held-out variable, not shape.
TTT_FILTER=${TTT_FILTER:-/input/DGN/bottle_one.json}
TTT_NUM_ENV=${TTT_NUM_ENV:-60}

# (TTT_RESULT_TAG is set above, alongside the rope suffix, so the rope-on and
#  rope-off arms write to separate result dirs and neither lands in the
#  friction run's directory.)

# Scene mode. The multi-bottle eval uses mode=valid to cycle deterministically
# through the held-out objects, but with ONE bottle that path also pins the
# object's ORIENTATION to a fixed validation pose: measured quat std across
# resets is ~0.003 under mode=valid vs ~0.49 under mode=train. Since collect runs
# mode=train, a mode=valid eval would score the student on a far narrower pose
# distribution than it was trained on. Default to train so collect and eval match;
# set TTT_MODE=valid if you deliberately want the fixed-pose variant.
TTT_MODE=${TTT_MODE:-train}

# Physics isolation, MASS variant — MUST match collect_ttt_mass.sh exactly, or
# the student is evaluated on a different dynamics distribution than it saw.
S=${TTT_FIX_SCALE:-0.08}
R=${TTT_FIX_RESTITUTION:-0.0}
OF=${TTT_FIX_OBJ_FRICTION:-0.5}
TF=${TTT_FIX_TBL_FRICTION:-0.5}
HF=${TTT_FIX_HAND_FRICTION:-1.0}
MASS_MIN=${TTT_MASS_MIN:-0.1}
MASS_MAX=${TTT_MASS_MAX:-0.5}

echo "[eval_ttt_mass] k=${K}  alpha=${TTT_ALPHA}  trunk=${TTT_POLICY_TRUNK}  recon=${TTT_RECON_WEIGHT}"
echo "[eval_ttt_mass] ckpt=${STUDENT_CKPT}"
echo "[eval_ttt_mass] mass [${MASS_MIN},${MASS_MAX}] kg; pinned scale=${S} rest=${R} obj_fr=${OF} tbl_fr=${TF} hand_fr=${HF}"
echo "[eval_ttt_mass] results -> output/test_rma/dywa/result/dgn-${TTT_RESULT_TAG}/"

REC_ARGS=()
if [ "${TTT_RECORD:-0}" = "1" ]; then
  REC_DIR=${TTT_RECORD_DIR:-/home/user/DyWA/output/ttt/videos_mass}
  REC_MAX=${TTT_RECORD_MAX:-6}
  REC_USE_COL=${TTT_RECORD_USE_COL:-False}
  REC_GHOST=${TTT_RECORD_GHOST:-True}
  REC_ARGS=(
    ++use_nvdr_record_episode=True
    ++nvdr_record_episode.episode_type=both
    ++nvdr_record_episode.record_dir=${REC_DIR}
    ++nvdr_record_episode.max_per_type=${REC_MAX}
    ++nvdr_record_episode.use_col=${REC_USE_COL}
    ++nvdr_record_episode.draw_goal_ghost=${REC_GHOST}
  )
  echo "[eval_ttt_mass] recording succ/fail videos -> ${REC_DIR}"
fi

PYTORCH_JIT=0 python3 eval_ttt.py \
  "${ROPE_ARGS[@]}" \
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
  ++env.single_object_scene.mode=${TTT_MODE} \
  ++env.num_env=${TTT_NUM_ENV} \
  ++env.single_object_scene.min_mass=${MASS_MIN} \
  ++env.single_object_scene.max_mass=${MASS_MAX} \
  ++env.single_object_scene.min_scale=${S} ++env.single_object_scene.max_scale=${S} \
  ++env.single_object_scene.min_object_restitution=${R} \
  ++env.single_object_scene.max_object_restitution=${R} \
  ++env.single_object_scene.min_object_friction=${OF} \
  ++env.single_object_scene.max_object_friction=${OF} \
  ++env.single_object_scene.min_table_friction=${TF} \
  ++env.single_object_scene.max_table_friction=${TF} \
  ++env.franka.min_hand_friction=${HF} \
  ++env.franka.max_hand_friction=${HF} \
  "${REC_ARGS[@]}"
# Success rate is printed to stdout ("Global average success rate: ..."); each k
# overwrites the same file under output/test_rma/dywa/result/dgn-mass/, so read
# the rate from each run's stdout.
