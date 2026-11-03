#!/bin/bash
# Usage: bash collect_ttt_mass.sh [GPU]
#
# MASS-varying variant of collect_ttt.sh, on ONE bottle. Collects a
# teacher-rollout dataset in which MASS is the hidden dynamics parameter and
# everything else (object identity, scale, restitution, object friction, table
# friction, hand friction) is PINNED — the mirror image of the friction-varying
# single-bottle pipeline.
#
# Why mass instead of friction (measured 2026-08-04, GPU 1):
#   * headroom  — teacher success falls 0.749 -> 0.167 across mass 0.05-1.5 kg,
#                 vs 0.64-0.71 (flat) across friction 0.05-1.5. Friction has
#                 nothing for an adaptive belief to recover; mass does.
#                 (_probe_param_headroom.py)
#   * true label — the teacher's phys_params mass entry reads corr=+1.000 against
#                 the actual sim mass, because AddPhysParams reads it once from
#                 get_actor_rigid_body_properties and the reset-time DR never
#                 rewrites mass. The friction entry decorrelates to ~0.05 after
#                 the first reset (mid-sim friction writes are inert on the GPU
#                 pipeline), so the friction imitation target was label noise.
#
# NOTE ON RANGE: this deliberately keeps icra_base's trained mass range
# (0.1-0.5 kg). Within that range the measured headroom is only ~7.7 points
# (0.804 -> 0.727), which is within ~2sigma of noise — so a flat TTT curve is a
# plausible outcome here too. Widening min/max_mass would expose the real
# 58-point effect but needs a teacher retrained on the wider range.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
# The repo carries an uncommitted breakpoint() in env/common.py's DR path.
export PYTHONBREAKPOINT=${PYTHONBREAKPOINT:-0}

GPU=${1:-0}

# collector knobs (read from env by collect_teacher_dataset.py). Output path is
# distinct from the friction run's dataset_one_bottle*.pkl.
export TTT_NUM_EPISODES=${TTT_NUM_EPISODES:-500}
export TTT_MAX_STEPS=${TTT_MAX_STEPS:-20000}
export TTT_OUT=${TTT_OUT:-/home/user/DyWA/output/ttt/dataset_mass_one_bottle.pkl}
mkdir -p "$(dirname "$TTT_OUT")"

# ONE bottle: the same object in every env, so mass is the only thing that
# differs between envs. Object identity would otherwise be a second hidden
# variable, and the belief could explain the outcome with shape instead of mass.
# Override with TTT_FILTER=/input/DGN/bottle_train.json for the 19-bottle set.
TTT_FILTER=${TTT_FILTER:-/input/DGN/bottle_one.json}
TTT_NUM_ENV=${TTT_NUM_ENV:-32}

# Physics isolation, MASS variant: pin scale, restitution and ALL THREE friction
# coefficients (object, table, hand) so mass is the sole hidden variable. Mass
# keeps icra_base's trained range and is randomized per env at scene setup.
#
# Note the asymmetry vs the friction pipeline: mass is assigned ONCE at scene
# creation (apply_domain_randomization with enable_mass=True) and the reset-time
# DR never rewrites it, so mass is genuinely constant within an env for the whole
# run. That is exactly what makes it a well-posed belief target.
S=${TTT_FIX_SCALE:-0.08}
R=${TTT_FIX_RESTITUTION:-0.0}
OF=${TTT_FIX_OBJ_FRICTION:-0.5}
TF=${TTT_FIX_TBL_FRICTION:-0.5}
HF=${TTT_FIX_HAND_FRICTION:-1.0}
MASS_MIN=${TTT_MASS_MIN:-0.1}
MASS_MAX=${TTT_MASS_MAX:-0.5}

echo "[collect_ttt_mass] object set: ${TTT_FILTER}"
echo "[collect_ttt_mass] mass varies [${MASS_MIN},${MASS_MAX}] kg per env"
echo "[collect_ttt_mass] pinned: scale=${S} restitution=${R} obj_fr=${OF} tbl_fr=${TF} hand_fr=${HF}"
echo "[collect_ttt_mass] -> ${TTT_OUT}"

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
  ++env.franka.max_hand_friction=${HF}
