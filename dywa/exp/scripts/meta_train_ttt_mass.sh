#!/bin/bash
# Usage: bash meta_train_ttt_mass.sh [GPU]
#
# MASS-varying variant of meta_train_ttt.sh, on ONE bottle. Identical
# meta-training; only the dataset, the checkpoint dir and the WandB project
# differ, so a mass run can never overwrite or be confused with a friction run.
#
#   dataset   -> output/ttt/dataset_mass_one_bottle.pkl   (collect_ttt_mass.sh)
#   ckpts     -> output/ttt/ckpt_mass_one_bottle/<run_name>/
#   wandb     -> project 'ttt_dywa_mass'
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"
export PYTHONBREAKPOINT=${PYTHONBREAKPOINT:-0}

GPU=${1:-0}

export TTT_DATA=${TTT_DATA:-/home/user/DyWA/output/ttt/dataset_mass_one_bottle.pkl}
export TTT_EPOCHS=${TTT_EPOCHS:-500}
export TTT_ACCUM=${TTT_ACCUM:-32}
export TTT_SAVE_DIR=${TTT_SAVE_DIR:-/home/user/DyWA/output/ttt/ckpt_mass_one_bottle}
mkdir -p "$TTT_SAVE_DIR"

# Separate WandB project so the mass sweep does not land among the friction runs.
export TTT_WANDB_PROJECT=${TTT_WANDB_PROJECT:-ttt_dywa_mass}

export TTT_ALPHA=${TTT_ALPHA:-0.5}
export TTT_POLICY_TRUNK=${TTT_POLICY_TRUNK:-film_transformer}
export TTT_RECON_WEIGHT=${TTT_RECON_WEIGHT:-5}

# 3D RoPE (axis-split, Q/K only) in OutcomeEncoder / DynamicsModel /
# TransformerPointEncoder. TTT_ROPE=1 turns it on; meta_train_ttt.py then appends
# _rope<omega_min>-<omega_max> to the run dir. That suffix is the ONLY guard:
# RoPE is parameter-free, so a rope-on ckpt loads into a rope-off student with no
# error and `_assert_ckpt_covers_policy` cannot see the difference. Write the
# values as python's %g renders them ("30", not "30.0").
TTT_ROPE=${TTT_ROPE:-0}
TTT_ROPE_OMEGA_MIN=${TTT_ROPE_OMEGA_MIN:-0.4}
TTT_ROPE_OMEGA_MAX=${TTT_ROPE_OMEGA_MAX:-30}
ROPE_ARGS=(
  ++student.ttt.rope=$([ "${TTT_ROPE}" = "1" ] && echo True || echo False)
  ++student.ttt.rope_omega_min=${TTT_ROPE_OMEGA_MIN}
  ++student.ttt.rope_omega_max=${TTT_ROPE_OMEGA_MAX}
)

echo "[meta_train_ttt_mass] data=${TTT_DATA}"
echo "[meta_train_ttt_mass] save=${TTT_SAVE_DIR}  wandb_project=${TTT_WANDB_PROJECT}"
echo "[meta_train_ttt_mass] rope=${TTT_ROPE} (omega ${TTT_ROPE_OMEGA_MIN}..${TTT_ROPE_OMEGA_MAX})"

PYTORCH_JIT=0 python3 meta_train_ttt.py \
  "${ROPE_ARGS[@]}" \
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
