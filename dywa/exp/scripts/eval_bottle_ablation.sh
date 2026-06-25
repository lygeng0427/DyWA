#!/bin/bash
# Usage: bash eval_bottle_ablation.sh {full|no_adapt|no_hist_film} [GPU]
#
# Evaluate a bottle-only student (trained by train_bottle_ablation.sh) on the
# single held-out evaluation bottle (bottle_test.json: 1 mesh, 5 scales).
#
# IMPORTANT:
#   * `+load_student` points at the NEWLY TRAINED ablation checkpoint, not the
#     pretrained one -- otherwise you would silently evaluate the pretrained model.
#   * the per-variant architecture overrides MUST match those used in training:
#     test_rma loads with strict=False, so a mismatch leaves weights randomly
#     initialised without erroring (see verification note below).
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

train_root="/home/user/DyWA/output/bottle_ablation"
name="bottle_${VARIANT}"
RUN="run-000"                            # <-- set to the actual training run subdir
# Training sets path.root=${train_root}/${name}; RunPath then creates
# ${path.root}/run-NNN/{ckpt,stat}. (++name does NOT add a path subdir.)
student_ckpt="${train_root}/${name}/${RUN}/ckpt/last.ckpt"

root="/home/user/DyWA/output/test_rma/bottle_ablation"

case "$VARIANT" in
  full)         ABL="++student.decoder.film_mlp=1" ;;
  no_adapt)     ABL="++student.decoder.film_mlp=1 ++student.constraint.loss_coef=0" ;;
  no_hist_film) ABL="++student.use_history=False ++student.decoder.decoder_type=mlp" ;;
  *) echo "unknown variant '$VARIANT' (expected full|no_adapt|no_hist_film)"; exit 1 ;;
esac

if [ ! -f "$student_ckpt" ]; then
  echo "student checkpoint not found: $student_ckpt"
  echo "  -> set RUN to the actual run-NNN subdir produced by training."
  exit 1
fi

mkdir -p "${root}/${name}"

PYTORCH_JIT=0 python3 test_rma.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++name="$name" \
  ++path.root="${root}/${name}" \
  ++env.num_env=60 \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  +load_student="${student_ckpt}" \
  ++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
  ++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
  ++plot_pc=0 \
  ++dagger_train_env.anneal_step=1 \
  ++add_teacher_state=1 \
  ++env.single_object_scene.filter_file=/input/DGN/bottle_test.json \
  ++env.single_object_scene.mode=valid \
  ++monitor.num_env_record=60 \
  ++log_categorical_results=True \
  ${ABL}
# &> "${root}/${name}/out.out"
