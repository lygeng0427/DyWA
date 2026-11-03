#!/bin/bash
# Usage: bash collect_fric_tracks.sh [GPU]
#
# Collect GAP-style 3D point-flow tracks with ONE friction channel swept over a
# per-env grid and every other dynamics variable pinned. Wraps
# collect_fric_tracks.py, which in turn drives collect_gap_tracks.py unchanged.
#
# The pinned values are IDENTICAL to collect_gap_tracks.sh's GAP_FIX_PHYS=fixed
# defaults, so the resulting dataset differs from output/gap/tracks_act_*.hdf5 in
# exactly one variable. That is what makes the "does conditioning recover the
# loss?" comparison interpretable.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1          # vendored DINOv2 deps are absent; fallback is fine
mkdir -p "$TORCH_EXTENSIONS_DIR"

GPU=${1:-0}

# ── the swept channel ─────────────────────────────────────────────────────
# table  : table friction swept, object friction pinned  (the experiment)
# object : object friction swept, table friction pinned  (the fallback, taken
#          only if the Step-A probe shows the table channel is inert)
# mass   : object mass swept, both frictions pinned
export FRIC_CHANNEL=${FRIC_CHANNEL:-table}
# Range and spacing default per channel inside collect_fric_tracks.py (friction
# 0.1-1.5 linear, mass log). Only export them if the caller overrode them, or an
# empty value would beat the python-side default.
if [ -n "${FRIC_LO}" ];      then export FRIC_LO;      fi
if [ -n "${FRIC_HI}" ];      then export FRIC_HI;      fi
if [ -n "${FRIC_SPACING}" ]; then export FRIC_SPACING; fi

# ── everything else: the GAP collector's own knobs ────────────────────────
export GAP_OUT=${GAP_OUT:-/home/user/DyWA/output/fric/tracks.hdf5}
export GAP_NUM_EPISODES=${GAP_NUM_EPISODES:-1250}
export GAP_MAX_STEPS=${GAP_MAX_STEPS:-400000}
export GAP_MAX_EP_LEN=${GAP_MAX_EP_LEN:-0}
export GAP_MIN_EP_LEN=${GAP_MIN_EP_LEN:-4}
# Success-only defaults ON for friction and OFF for mass. Mass strongly affects
# whether the teacher reaches the goal, so keeping only successes conditions on
# an effect of the swept variable -- a collider that preferentially deletes the
# episodes where mass mattered most. Collect everything (episode_success is
# stored) and filter at train time with FRIC_SUCCESS_ONLY instead.
if [ "$FRIC_CHANNEL" = "mass" ]; then
  export GAP_SUCCESS_ONLY=${GAP_SUCCESS_ONLY:-0}
else
  export GAP_SUCCESS_ONLY=${GAP_SUCCESS_ONLY:-1}
fi
export GAP_TRACK_HORIZON=${GAP_TRACK_HORIZON:-16}
export GAP_DINO=${GAP_DINO:-1}
export GAP_STORE_IMAGE=${GAP_STORE_IMAGE:-1}
mkdir -p "$(dirname "$GAP_OUT")"

GAP_FILTER=${GAP_FILTER:-/input/DGN/bottle_one.json}

# num_env is also the GRID SIZE -- one friction value per env. 32 is the
# measured throughput knee for the collector and the number of discrete friction
# values this experiment is specified around; changing it changes both at once.
GAP_NUM_ENV=${GAP_NUM_ENV:-32}
# Varies object poses/goals across parallel GPUs. It does NOT vary the friction
# grid: that is a linspace, deliberately identical on every GPU (see the module
# docstring) so the two output files share one 32-value axis.
GAP_SEED=${GAP_SEED:-0}

# Pinned physics -- same numbers as collect_gap_tracks.sh's GAP_FIX_PHYS=fixed.
FIX_S=${GAP_FIX_SCALE:-0.08}            # object scale (m)
FIX_M=${GAP_FIX_MASS:-0.3}              # object mass (kg)
FIX_R=${GAP_FIX_RESTITUTION:-0.0}
FIX_OF=${GAP_FIX_OBJ_FRICTION:-0.6}
FIX_TF=${GAP_FIX_TABLE_FRICTION:-0.55}
FIX_HF=${GAP_FIX_HAND_FRICTION:-1.25}   # no per-env path anyway; pinned for the record

PIN_ARGS=(
  ++env.single_object_scene.min_scale=${FIX_S} ++env.single_object_scene.max_scale=${FIX_S}
  ++env.single_object_scene.min_mass=${FIX_M} ++env.single_object_scene.max_mass=${FIX_M}
  ++env.single_object_scene.min_object_restitution=${FIX_R}
  ++env.single_object_scene.max_object_restitution=${FIX_R}
  ++env.franka.min_hand_friction=${FIX_HF} ++env.franka.max_hand_friction=${FIX_HF}
)
# The swept channel's config range is pinned too. This is not redundant with the
# bake: the config draw runs FIRST (scene setup) and the bake then overwrites the
# swept body, so pinning here makes the NON-swept body's value definite and keeps
# the phys_params assertion in models/fric/dataset.py meaningful.
case "$FRIC_CHANNEL" in
  table)
    PIN_ARGS+=(
      ++env.single_object_scene.min_object_friction=${FIX_OF}
      ++env.single_object_scene.max_object_friction=${FIX_OF})
    echo "[fric] TABLE friction swept over [${FRIC_LO:-0.1}, ${FRIC_HI:-1.5}]; object pinned at ${FIX_OF}"
    ;;
  object)
    PIN_ARGS+=(
      ++env.single_object_scene.min_table_friction=${FIX_TF}
      ++env.single_object_scene.max_table_friction=${FIX_TF})
    echo "[fric] OBJECT friction swept over [${FRIC_LO:-0.1}, ${FRIC_HI:-1.5}]; table pinned at ${FIX_TF}"
    ;;
  mass)
    # BOTH frictions pinned. Leaving table friction on its config range would
    # make it vary across envs, and train_fric_track.py's assert_phys_grid would
    # (correctly) refuse the file -- the sweep must be one variable, not two.
    PIN_ARGS+=(
      ++env.single_object_scene.min_object_friction=${FIX_OF}
      ++env.single_object_scene.max_object_friction=${FIX_OF}
      ++env.single_object_scene.min_table_friction=${FIX_TF}
      ++env.single_object_scene.max_table_friction=${FIX_TF})
    # min_mass/max_mass stay pinned at FIX_M above ON PURPOSE. The bake runs
    # after scene setup and overwrites them, so a bake that silently failed
    # shows up as FIX_M in every env and the readback assert fires. Leaving the
    # config DR live instead would hide that behind 32 plausible random masses.
    echo "[fric] MASS swept over [${FRIC_LO:-0.05}, ${FRIC_HI:-1.0}] kg;"\
         "frictions pinned at table=${FIX_TF} object=${FIX_OF}"
    ;;
  *) echo "unknown FRIC_CHANNEL='${FRIC_CHANNEL}' (want: table|object|mass)"; exit 1 ;;
esac

echo "[collect_fric_tracks] out=$GAP_OUT episodes=$GAP_NUM_EPISODES "\
"envs=$GAP_NUM_ENV seed=$GAP_SEED horizon=$GAP_TRACK_HORIZON dino=$GAP_DINO"

PYTORCH_JIT=0 python3 -u collect_fric_tracks.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
  ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
  ++env.single_object_scene.filter_file=${GAP_FILTER} \
  ++env.num_env=${GAP_NUM_ENV} \
  ++env.seed=${GAP_SEED} \
  "${PIN_ARGS[@]}"
