#!/bin/bash
# Usage: bash collect_gap_tracks.sh [GPU]
#
# Generate GAP-style ground-truth 3D point-flow tracks from teacher rollouts.
# Method ported from third_party/GAP/scripts/generate_gt_tracks.py; see
# _plan_gap_tracks.md for the measurements this relies on.
#
# Prereqs:
#   * DINOv2 weights (1.2 GB), fetched on the HOST into the mounted cache:
#       mkdir -p ~/.cache/dywa/dinov2 && cd ~/.cache/dywa/dinov2
#       curl -L -C - -o model.safetensors \
#         https://huggingface.co/timm/vit_large_patch14_reg4_dinov2.lvd142m/resolve/main/model.safetensors
#     (host ~/.cache/dywa == container /home/user/.cache/pkm)
#   * Set GAP_DINO=0 to skip feature extraction entirely.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1          # vendored DINOv2 deps are absent; fallback is fine
mkdir -p "$TORCH_EXTENSIONS_DIR"

GPU=${1:-0}

export GAP_OUT=${GAP_OUT:-/home/user/DyWA/output/gap/tracks.hdf5}
export GAP_NUM_EPISODES=${GAP_NUM_EPISODES:-50}
export GAP_MAX_STEPS=${GAP_MAX_STEPS:-20000}
# Length/outcome policy is deliberately IDENTICAL to collect_ttt.sh's collector
# (collect_teacher_dataset.py): a MIN floor, NO upper cap, success-only.
# An upper cap is NOT a neutral memory knob -- episodes hitting the task
# timeout (300 steps -> ~283 frames) are exactly what it discards, which
# silently biases the dataset toward short/easy rollouts.
export GAP_MAX_EP_LEN=${GAP_MAX_EP_LEN:-0}
export GAP_MIN_EP_LEN=${GAP_MIN_EP_LEN:-4}
export GAP_SUCCESS_ONLY=${GAP_SUCCESS_ONLY:-1}
export GAP_TRACK_HORIZON=${GAP_TRACK_HORIZON:-16}
export GAP_DINO=${GAP_DINO:-1}
export GAP_STORE_IMAGE=${GAP_STORE_IMAGE:-1}
mkdir -p "$(dirname "$GAP_OUT")"

# Frames are ~1.1 MB each with features+image, and a whole episode is buffered
# per env before being written, so keep num_env modest.
GAP_FILTER=${GAP_FILTER:-/input/DGN/bottle_one.json}

# ── fixed-physics variants ────────────────────────────────────────────────
# GAP_FIX_PHYS selects which single dynamics variable is left free; everything
# else is pinned by setting min==max. Mirrors TTT_FIX_PHYS in collect_ttt.sh.
#   friction : vary object + table + hand friction; pin scale/mass/restitution
#   mass     : vary object mass;                    pin scale/restitution/all friction
#   fixed    : pin EVERYTHING -- one single dynamics config for every episode.
#              The only variation left is the initial object pose / goal draw,
#              so use GAP_SEED to keep parallel GPUs from duplicating rollouts.
#   (unset)  : full icra_base domain randomisation
FIX_S=${GAP_FIX_SCALE:-0.08}            # object scale (m)
FIX_M=${GAP_FIX_MASS:-0.3}              # object mass (kg)
FIX_R=${GAP_FIX_RESTITUTION:-0.0}
FIX_OF=${GAP_FIX_OBJ_FRICTION:-0.6}     # within default 0.2-1.0
FIX_TF=${GAP_FIX_TABLE_FRICTION:-0.55}  # within icra_base 0.3-0.8
FIX_HF=${GAP_FIX_HAND_FRICTION:-1.25}   # within icra_base 1.0-1.5

PIN_SCALE=(++env.single_object_scene.min_scale=${FIX_S} ++env.single_object_scene.max_scale=${FIX_S})
PIN_MASS=(++env.single_object_scene.min_mass=${FIX_M} ++env.single_object_scene.max_mass=${FIX_M})
PIN_REST=(++env.single_object_scene.min_object_restitution=${FIX_R} ++env.single_object_scene.max_object_restitution=${FIX_R})
PIN_FRIC=(
  ++env.single_object_scene.min_object_friction=${FIX_OF} ++env.single_object_scene.max_object_friction=${FIX_OF}
  ++env.single_object_scene.min_table_friction=${FIX_TF} ++env.single_object_scene.max_table_friction=${FIX_TF}
  ++env.franka.min_hand_friction=${FIX_HF} ++env.franka.max_hand_friction=${FIX_HF}
)

FIX_ARGS=()
case "${GAP_FIX_PHYS:-none}" in
  friction)
    FIX_ARGS=("${PIN_SCALE[@]}" "${PIN_MASS[@]}" "${PIN_REST[@]}")
    echo "[fix_phys] FRICTION varies; pinned scale=${FIX_S} mass=${FIX_M} restitution=${FIX_R}"
    ;;
  mass)
    FIX_ARGS=("${PIN_SCALE[@]}" "${PIN_REST[@]}" "${PIN_FRIC[@]}")
    echo "[fix_phys] MASS varies; pinned scale=${FIX_S} restitution=${FIX_R} \
obj_fric=${FIX_OF} table_fric=${FIX_TF} hand_fric=${FIX_HF}"
    ;;
  fixed)
    FIX_ARGS=("${PIN_SCALE[@]}" "${PIN_MASS[@]}" "${PIN_REST[@]}" "${PIN_FRIC[@]}")
    echo "[fix_phys] NOTHING varies; scale=${FIX_S} mass=${FIX_M} restitution=${FIX_R} \
obj_fric=${FIX_OF} table_fric=${FIX_TF} hand_fric=${FIX_HF}"
    ;;
  none) ;;
  *) echo "unknown GAP_FIX_PHYS='${GAP_FIX_PHYS}' (want: friction|mass|fixed)"; exit 1 ;;
esac
# num_env drives throughput almost linearly: nearly everything per step (physics,
# nvdr render, wrapper stack) is fixed overhead amortised over num_env frames.
# The old default of 8 was set for RAM -- a whole episode is buffered per env
# before it is written -- but at ~1.1 MB/frame and ~300-frame episodes even 64
# envs is ~21 GB against ~100 GB free, so that ceiling is not the binding one.
GAP_NUM_ENV=${GAP_NUM_ENV:-32}
# Without this every GPU draws the SAME object poses and goals, so a 2-GPU
# "parallel collection" would just produce the same dataset twice.
GAP_SEED=${GAP_SEED:-0}

echo "[collect_gap_tracks] out=$GAP_OUT episodes=$GAP_NUM_EPISODES "\
"envs=$GAP_NUM_ENV seed=$GAP_SEED horizon=$GAP_TRACK_HORIZON dino=$GAP_DINO"

PYTORCH_JIT=0 python3 -u collect_gap_tracks.py \
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
  "${FIX_ARGS[@]}"
