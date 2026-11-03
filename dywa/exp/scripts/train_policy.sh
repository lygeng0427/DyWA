#!/bin/bash
# Usage: bash train_policy.sh <cond> [GPU]
#   cond : none | mass | latent
#
# Trains the joint action+track DiT on the FIFO shard window, with NO inner
# loop. This is the HEADROOM GATE: run `none` and `mass` (and `mass` with
# POLICY_PHYS_SHUFFLE=1 as the negative control) before the meta-trainer,
# because a latent can only recover what the true value carries.
#
# Read `val/action`, not `val/loss`: the total is dominated by a track term
# that is ~87% arm, and the arm's track is the action's own consequence.
#
# REQUIRED, no defaults -- both decide what the numbers mean:
#   POLICY_EVAL_DATA     the frozen eval corpus (never streamed, never trained on)
#   POLICY_VAL_IDS_FROM  a reference ckpt whose val_ids pin the held-out episodes
#
# The shard window is train_stream_tracks.ShardWindow, so its knobs are the
# STREAM_* ones (STREAM_DIR, STREAM_WINDOW_SHARDS, STREAM_MIN_SHARDS,
# STREAM_DINO_DIM, STREAM_WORKERS).
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$TORCH_EXTENSIONS_DIR"

export POLICY_COND=${1:-none}
GPU=${2:-0}
export POLICY_DEVICE=cuda:${GPU}

export STREAM_CHANNEL=${STREAM_CHANNEL:-mass}
export STREAM_DIR=${STREAM_DIR:-/home/user/DyWA/output/mass/meta_stream}
export STREAM_WINDOW_SHARDS=${STREAM_WINDOW_SHARDS:-30}
export STREAM_MIN_SHARDS=${STREAM_MIN_SHARDS:-4}
export STREAM_DINO_DIM=${STREAM_DINO_DIM:-64}
export STREAM_WORKERS=${STREAM_WORKERS:-6}
# Eviction DELETES the evicted shard. Default to keeping, because the gate runs
# two arms against ONE static shard directory and either would otherwise delete
# the other's data. Set 0 explicitly for a long run with a live collector, where
# bounded disk is the whole point.
export STREAM_KEEP_SHARDS=${STREAM_KEEP_SHARDS:-1}
export POLICY_SAVE_DIR=${POLICY_SAVE_DIR:-/home/user/DyWA/output/mass/ckpt_policy}
export POLICY_TOTAL_STEPS=${POLICY_TOTAL_STEPS:-200000}
export POLICY_BLOCK_STEPS=${POLICY_BLOCK_STEPS:-10000}

if [ -z "${POLICY_EVAL_DATA}" ]; then
  echo "POLICY_EVAL_DATA is required (the frozen eval corpus)."; exit 1; fi
if [ -z "${POLICY_VAL_IDS_FROM}" ]; then
  echo "POLICY_VAL_IDS_FROM is required (reference ckpt pinning the val ids)."; exit 1; fi

echo "[policy] cond=$POLICY_COND shuffle=${POLICY_PHYS_SHUFFLE:-0} gpu=$GPU"
echo "[policy] stream=$STREAM_DIR window=$STREAM_WINDOW_SHARDS shards"
echo "[policy] steps=$POLICY_TOTAL_STEPS block=$POLICY_BLOCK_STEPS lr=${POLICY_LR:-1e-4}"
echo "[policy] eval=$POLICY_EVAL_DATA"

PYTORCH_JIT=0 python3 -u train_policy.py
