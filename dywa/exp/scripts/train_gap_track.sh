#!/bin/bash
# Usage: bash train_gap_track.sh <trunk> [gpu] [action_mode]
#
# Train one GAP-style track model on the single-friction dataset. Runs WITHOUT
# Isaac Gym (reads the collector's HDF5), so it can share the box with a
# running collection.
#
#   trunk        transformer | dit
#   action_mode  realized (default) | command | both
#                realized = the EEF pose actually reached each step (7-D)
#                command  = the policy's own 20-D action (dpos, dori, KP, KD)
#                both     = concatenated. Only `command` is available at
#                           planning time, when nothing has been executed yet.
#
# No latent belief: one pinned friction config means there is no hidden
# dynamics variable to infer. Curves go to WandB (GAP_WANDB_PROJECT,
# default 'dywa-gap-track'); GAP_WANDB_MODE=disabled turns it off.
#
# MUST run inside the DyWA Docker container.
set -e
TRUNK=${1:?trunk: transformer|dit}
GPU=${2:-0}
MODE=${3:-realized}

export GAP_TRUNK=$TRUNK
export GAP_ACTION_MODE=$MODE
export GAP_DEVICE=cuda:${GPU}
export GAP_DATA=${GAP_DATA:-/home/user/DyWA/output/gap/tracks_act_a.hdf5,/home/user/DyWA/output/gap/tracks_act_b.hdf5}
export GAP_SAVE_DIR=${GAP_SAVE_DIR:-/home/user/DyWA/output/gap/ckpt}
export GAP_EPOCHS=${GAP_EPOCHS:-40}
export GAP_BATCH=${GAP_BATCH:-16}
export GAP_D_MODEL=${GAP_D_MODEL:-512}
export GAP_WORKERS=${GAP_WORKERS:-8}
export GAP_OBJ_WEIGHT=${GAP_OBJ_WEIGHT:-1}
export GAP_HIST_K=${GAP_HIST_K:-0}      # past frames -> velocity
export GAP_HEAD=${GAP_HEAD:-free}       # free | rigid (per-body SE(3))
export GAP_WANDB_PROJECT=${GAP_WANDB_PROJECT:-dywa-gap-track}
export GAP_WANDB_MODE=${GAP_WANDB_MODE:-online}

cd /home/user/DyWA/dywa/exp/train
echo "[train_gap_track] trunk=$TRUNK action=$MODE gpu=$GPU wandb=$GAP_WANDB_MODE"
PYTORCH_JIT=0 python3 -u train_gap_track.py
