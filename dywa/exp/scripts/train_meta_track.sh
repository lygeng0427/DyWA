#!/bin/bash
# Usage: bash train_meta_track.sh <z_dim> [GPU]
#
# Second-order meta-training of a latent-conditioned track model: the latent
# replaces the privileged mass input and is fitted at test time from the
# episode's own elapsed motion. See train_meta_track.py's docstring.
#
# REQUIRED, no default -- it decides what the numbers mean:
#   META_VAL_IDS_FROM   a reference ckpt whose val_ids pin the held-out episodes.
#                       Every mass number in this project is on those 419
#                       episodes; without it the run is not comparable to the
#                       `film` ceiling (11.18 mm) or the `none` floor (11.96).
#
# The headline is `val/gain_mm` (pre - post) in WandB, measured WITHIN one model
# on identical frames. A between-model comparison here carries a ~0.13 mm seed
# spread, the same size as the effect. The in-training value is a SUBSET probe;
# quote only eval_meta_track.py.
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$TORCH_EXTENSIONS_DIR"

export META_Z_DIM=${1:-8}
GPU=${2:-0}
export META_DEVICE=cuda:${GPU}

# All six mass files, DINO-64 on disk. Not a guess: the checkpoints this is
# compared against record exactly this list in ck['paths'].
export META_DATA=${META_DATA:-/home/user/DyWA/output/mass/tracks_m_a.hdf5,/home/user/DyWA/output/mass/tracks_m_b.hdf5,/home/user/DyWA/output/mass/tracks_m64_0.hdf5,/home/user/DyWA/output/mass/tracks_m64_1.hdf5,/home/user/DyWA/output/mass/tracks_m64_b0.hdf5,/home/user/DyWA/output/mass/tracks_m64_b1.hdf5}
export META_SAVE_DIR=${META_SAVE_DIR:-/home/user/DyWA/output/mass/ckpt}
export META_DINO_DIM=${META_DINO_DIM:-64}
export META_ACTION_MODE=${META_ACTION_MODE:-command}

# TTT shape. `every` is BOTH the test-time refit cadence and a diagnostic: if
# performance is flat in it the latent is a persistent physics belief, and if it
# degrades sharply the latent is a time-local correction.
export META_H=${META_H:-16}
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
export META_DROPOUT=${META_DROPOUT:-0.3}
export META_DETACH_Q=${META_DETACH_Q:-0}

# MEASURED (_probe_meta_mem.py, RTX 4090 24 GB): peak memory is ~0.16 GB per
# frame and support frames dominate, because they are retained through a DOUBLE
# backward while query frames are ordinary. B=4,m=8,Q=8 -> 64 frames, 15.2 GB
# peak, 0.304 s/step. 8x8x8 OOMs. Query frames are the cheap axis but they come
# from one `every`-wide window, so raising Q past ~8 buys correlated samples.
export META_BATCH_EP=${META_BATCH_EP:-4}
export META_STEPS=${META_STEPS:-200000}
export META_LR=${META_LR:-1e-4}
export META_SEED=${META_SEED:-0}
export META_VAL_EVERY=${META_VAL_EVERY:-2000}
export META_VAL_EPS=${META_VAL_EPS:-24}

if [ -z "${META_VAL_IDS_FROM}" ]; then
  echo "META_VAL_IDS_FROM is required (reference ckpt pinning the val ids)."
  echo "  e.g. /home/user/DyWA/output/mass/ckpt/transformer_both_command/best.ckpt"
  exit 1
fi

echo "[train_meta] z_dim=$META_Z_DIM gpu=$GPU K=$META_K m_max=$META_M_MAX"\
" every=$META_EVERY Q=$META_QUERY drop=$META_DROPOUT fomaml=$META_DETACH_Q"
echo "[train_meta] steps=$META_STEPS batch_ep=$META_BATCH_EP lr=$META_LR"
echo "[train_meta] val_ids_from=$META_VAL_IDS_FROM"

PYTORCH_JIT=0 python3 -u train_meta_track.py
