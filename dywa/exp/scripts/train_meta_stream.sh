#!/bin/bash
# Usage: bash train_meta_stream.sh [GPU]
#
# Meta-training of the latent track model on a FIFO window of streamed shards,
# warm-started from the offline arm. Run the collector FIRST, on the OTHER GPU:
#
#   STREAM_CHANNEL=mass STREAM_DIR=/home/user/DyWA/output/mass/meta_stream \
#   STREAM_EVAL_DATA=<the two frozen corpus files> GAP_SEED=200 \
#   bash dywa/exp/scripts/collect_stream_tracks.sh 1
#
# Sharing one GPU between collector and trainer turns the validation pass from
# well under a minute into several.
#
# REQUIRED, no defaults -- each decides what the numbers mean:
#   META_INIT_FROM       the offline meta checkpoint to warm-start from
#   META_EVAL_DATA       the frozen eval corpus (never streamed, never trained on)
#   META_VAL_IDS_FROM    reference ckpt pinning the held-out 419 episodes
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$TORCH_EXTENSIONS_DIR"

GPU=${1:-0}
export META_DEVICE=cuda:${GPU}

export META_EVAL_DATA=${META_EVAL_DATA:-/home/user/DyWA/output/mass/tracks_m_a.hdf5,/home/user/DyWA/output/mass/tracks_m_b.hdf5}
export META_VAL_IDS_FROM=${META_VAL_IDS_FROM:-/home/user/DyWA/output/mass/ckpt/transformer_both_command/best.ckpt}
export META_INIT_FROM=${META_INIT_FROM:-/home/user/DyWA/output/mass/ckpt/meta_z8_k1_m8_e16_command/best.ckpt}
export META_SAVE_DIR=${META_SAVE_DIR:-/home/user/DyWA/output/mass/ckpt}

# Model/TTT shape. The trainer ASSERTS every one of these against the warm-start
# checkpoint's own cfg, so a mismatch fails in seconds rather than producing a
# quietly incomparable run.
export META_Z_DIM=${META_Z_DIM:-8}
# K=5 INNER STEPS, not 1. Measured on r2b @630k at e1 cadence, 30 episodes:
#   K=1 +0.165 | K=2 +0.288 | K=3 +0.366 | K=5 +0.394 | K=10 +0.103 mm
# so the trained single step is ~2.4x short of the optimum. The `damped` control
# (K steps of alpha/K, total travel capped at alpha) reproduces K=1 to three
# decimals at every K, so the inner DIRECTION is converged after one step and
# the extra steps buy DISTANCE -- the learned alpha (~0.077) is ~5x too small
# and meta-training cannot fix it itself (dL/dalpha ~ 0 on the flat surface near
# q0). Alpha is left alone deliberately; K is the knob being moved.
#
# m_max 8 -> 4 PAYS FOR THE MEMORY. Every inner step retains its graph through
# the double backward, so K=5 multiplies inner memory ~5x and 4x8x8 already
# peaks at 15.2 GB. Halving the support is the cheapest place to find it:
# measured cost of m=4 vs m=8 is -0.019 +- 0.018 mm (paired, 159 episodes),
# i.e. nothing. Raise it back and drop META_BATCH_EP instead if outer batch
# size turns out to matter more.
export META_K=${META_K:-5}
export META_M_MAX=${META_M_MAX:-4}
export META_EVERY=${META_EVERY:-16}
export META_QUERY=${META_QUERY:-8}
export META_DINO_DIM=${META_DINO_DIM:-64}
export META_ACTION_MODE=${META_ACTION_MODE:-command}
export META_DROPOUT=${META_DROPOUT:-0.3}

# Warm-started, so a LOW constant LR with a short warmup -- not OneCycle, which
# needs total_steps up front and makes the run un-extendable.
export META_BATCH_EP=${META_BATCH_EP:-4}
export META_TOTAL_STEPS=${META_TOTAL_STEPS:-400000}
export META_BLOCK_STEPS=${META_BLOCK_STEPS:-5000}
export META_LR=${META_LR:-3e-5}
export META_WARMUP=${META_WARMUP:-2000}
export META_VAL_EPS=${META_VAL_EPS:-40}

# Consumed by train_stream_tracks.py's ShardWindow at import.
export STREAM_DIR=${STREAM_DIR:-/home/user/DyWA/output/mass/meta_stream}
export STREAM_WINDOW_SHARDS=${STREAM_WINDOW_SHARDS:-25}
export STREAM_MIN_SHARDS=${STREAM_MIN_SHARDS:-4}
export STREAM_DINO_DIM=${STREAM_DINO_DIM:-64}
export STREAM_SPLIT_SEED=${STREAM_SPLIT_SEED:-0}
export STREAM_KEEP_SHARDS=${STREAM_KEEP_SHARDS:-0}
export STREAM_POLL_S=${STREAM_POLL_S:-20}
export STREAM_STATIC=${STREAM_STATIC:-}
mkdir -p "$STREAM_DIR"

for v in META_INIT_FROM META_EVAL_DATA META_VAL_IDS_FROM; do
  if [ -z "${!v}" ]; then echo "$v is required."; exit 1; fi
done

echo "[meta_stream] gpu=$GPU z=$META_Z_DIM K=$META_K m_max=$META_M_MAX"\
" every=$META_EVERY Q=$META_QUERY batch_ep=$META_BATCH_EP"
echo "[meta_stream] stream=$STREAM_DIR window=$STREAM_WINDOW_SHARDS"\
" min=$STREAM_MIN_SHARDS block=$META_BLOCK_STEPS lr=$META_LR"
echo "[meta_stream] init_from=$META_INIT_FROM"

PYTORCH_JIT=0 python3 -u train_meta_stream.py
