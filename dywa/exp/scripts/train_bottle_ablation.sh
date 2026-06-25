#!/bin/bash
# Usage: bash train_bottle_ablation.sh {full|no_adapt|no_hist_film} [GPU]
#
# Distill a bottle-only student from the pretrained (general) teacher.
#   full          : baseline  -- history + FiLM + adaptation loss
#   no_adapt      : ablation 1 -- drop the adaptation contrastive loss (loss_coef=0)
#   no_hist_film  : ablation 2 -- no HistoryEncoder and no FiLM decoder
#                   (this also removes the adaptation loss, since there is no
#                    history `cond` to supervise)
#
# Trains only on the 19 training bottles (bottle_train.json). The teacher is
# reused as-is; the ablations are student-side only.
#
# MUST be run inside the DyWA Docker container (needs Isaac Gym).
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

VARIANT=${1:-}
GPU=${2:-0}
if [ -z "$VARIANT" ]; then
  echo "usage: $(basename "$0") {full|no_adapt|no_hist_film} [GPU]"; exit 1
fi

TRAIN_STEP=20000                       # decent for bottle-only (see CLAUDE.md)
SAVE_STEP=2000                         # checkpoint save frequency (cfg.save_step)

root="/home/user/DyWA/output/bottle_ablation"
name="bottle_${VARIANT}"

case "$VARIANT" in
  full)         ABL="++student.decoder.film_mlp=1" ;;
  no_adapt)     ABL="++student.decoder.film_mlp=1 ++student.constraint.loss_coef=0" ;;
  no_hist_film) ABL="++student.use_history=False ++student.decoder.decoder_type=mlp" ;;
  *) echo "unknown variant '$VARIANT' (expected full|no_adapt|no_hist_film)"; exit 1 ;;
esac

mkdir -p "${root}/${name}"

PYTORCH_JIT=0 python3 train_rma.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++name="$name" \
  ++path.root="${root}/${name}" \
  ++env.num_env=1024 \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  ++add_teacher_state=1 \
  ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
  ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
  ++env.single_object_scene.filter_file=/input/DGN/bottle_train.json \
  ++train_step=${TRAIN_STEP} \
  ++save_step=${SAVE_STEP} \
  ${ABL}
# &> "${root}/${name}/out.out"
