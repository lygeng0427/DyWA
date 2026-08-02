#!/bin/bash
# Usage: bash meta_train_ttt.sh [GPU]
#
# Offline TTT meta-training on the teacher-rollout dataset (from collect_ttt.sh).
# Does NOT instantiate Isaac Gym, but needs the DyWA point encoder (CUDA).
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

GPU=${1:-0}

# meta-train knobs (read from env by meta_train_ttt.py)
export TTT_DATA=${TTT_DATA:-/home/user/DyWA/output/ttt/dataset_ttt_teacher.pkl}
export TTT_EPOCHS=${TTT_EPOCHS:-800}
export TTT_ACCUM=${TTT_ACCUM:-32}
export TTT_SAVE_DIR=${TTT_SAVE_DIR:-/home/user/DyWA/output/ttt/ckpt}
mkdir -p "$TTT_SAVE_DIR"

# Metrics go to WandB (project=$TTT_WANDB_PROJECT, entity=$TTT_WANDB_ENTITY, run
# name = the per-run dir e.g. alpha_1_recon5). Override or disable logging with:
#   TTT_WANDB_PROJECT=... TTT_WANDB_ENTITY=... TTT_WANDB_MODE=online|offline|disabled
# (`wandb login` once first; use TTT_WANDB_MODE=disabled for quick debug runs).

# Inner-loop step size. Each alpha writes to  $TTT_SAVE_DIR/alpha_<a>/  so an
# alpha sweep never overwrites itself, and each becomes a distinct WandB run:
#   for a in 0.1 0.5 1.0 2.0; do TTT_ALPHA=$a bash meta_train_ttt.sh 1; done
export TTT_ALPHA=${TTT_ALPHA:-1}

# Policy trunk: 'dywa' (TokenDecoder+Aggregator+q_head) or 'film_transformer'
# (doors-style FiLMPolicyTrunk). A/B them with:
#   for t in dywa film_transformer; do TTT_POLICY_TRUNK=$t bash meta_train_ttt.sh 1; done
export TTT_POLICY_TRUNK=${TTT_POLICY_TRUNK:-dywa}

PYTORCH_JIT=0 python3 meta_train_ttt.py \
  +platform=debug \
  +env=abs_goal_1view \
  +run=teacher_base \
  +student=dywa/base \
  ++global_device=cuda:${GPU} \
  ++student.norm="ln" \
  ++student.ttt.latent_dim=${TTT_LATENT_DIM:-64} \
  ++student.ttt.k_inner_steps=${TTT_K:-1} \
  ++student.ttt.ttt_alpha=${TTT_ALPHA} \
  ++student.ttt.policy_trunk=${TTT_POLICY_TRUNK} \
  ++student.ttt.policy_layers=${TTT_POLICY_LAYERS:-4} \
  ++student.ttt.window_k=${TTT_WINDOW_K:-5} \
  ++student.ttt.detach_q=False
