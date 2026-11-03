#!/bin/bash
# Usage: bash train_stream_tracks.sh <trunk> <phys_cond> <action_mode> [GPU]
#   trunk       : transformer | dit
#   phys_cond   : none | token | film | both
#   action_mode : command | command_pose | realized | realized_gain | both
#
# Trains on the FIFO shard window that collect_stream_tracks.sh is filling.
# Runs until STREAM_TOTAL_STEPS. Resumable (STREAM_RESUME=1) and warm-startable
# (STREAM_INIT_FROM=<ckpt>), because the point is to be able to keep pushing a
# run that has not yet converged instead of restarting it.
#
# REQUIRED, no defaults -- both decide what the numbers mean:
#   STREAM_EVAL_DATA     the frozen eval corpus (never streamed, never trained on)
#   STREAM_VAL_IDS_FROM  a reference ckpt whose val_ids pin the held-out episodes
#
# The headline is `gap/window_vs_val` in WandB: 1.89x offline, and the run is
# doing its job if it walks toward 1.0 WITHOUT val/mv_object rising to meet it.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$TORCH_EXTENSIONS_DIR"

export STREAM_TRUNK=${1:-transformer}
export STREAM_PHYS_COND=${2:-film}
export STREAM_ACTION_MODE=${3:-command}
GPU=${4:-0}
export STREAM_DEVICE=cuda:${GPU}

export STREAM_CHANNEL=${STREAM_CHANNEL:-mass}
export STREAM_DIR=${STREAM_DIR:-/home/user/DyWA/output/${STREAM_CHANNEL}/stream}
export STREAM_SAVE_DIR=${STREAM_SAVE_DIR:-/home/user/DyWA/output/${STREAM_CHANNEL}/ckpt}
export STREAM_WINDOW_SHARDS=${STREAM_WINDOW_SHARDS:-30}
export STREAM_TOTAL_STEPS=${STREAM_TOTAL_STEPS:-1680000}
export STREAM_BLOCK_STEPS=${STREAM_BLOCK_STEPS:-20000}
export STREAM_DINO_DIM=${STREAM_DINO_DIM:-64}
export STREAM_PROBE_FRAMES=${STREAM_PROBE_FRAMES:-12000}
export STREAM_FULL_VAL_EVERY=${STREAM_FULL_VAL_EVERY:-10}

if [ -z "${STREAM_EVAL_DATA}" ]; then
  echo "STREAM_EVAL_DATA is required (the frozen eval corpus)."; exit 1; fi
if [ -z "${STREAM_VAL_IDS_FROM}" ]; then
  echo "STREAM_VAL_IDS_FROM is required (reference ckpt pinning the val ids)."; exit 1; fi

echo "[train_stream] trunk=$STREAM_TRUNK phys_cond=$STREAM_PHYS_COND"\
" action=$STREAM_ACTION_MODE gpu=$GPU dino=$STREAM_DINO_DIM"
echo "[train_stream] stream=$STREAM_DIR window=$STREAM_WINDOW_SHARDS shards"
echo "[train_stream] steps=$STREAM_TOTAL_STEPS block=$STREAM_BLOCK_STEPS"\
" lr=${STREAM_LR:-<auto>} anneal=${STREAM_ANNEAL_STEPS:-0}"
echo "[train_stream] eval=$STREAM_EVAL_DATA"
echo "[train_stream] val_ids_from=$STREAM_VAL_IDS_FROM"
echo "[train_stream] init_from=${STREAM_INIT_FROM:-<scratch>} resume=${STREAM_RESUME:-0}"

PYTORCH_JIT=0 python3 -u train_stream_tracks.py
