#!/bin/bash
# Usage: bash train_meta_policy.sh [GPU]
#
# Second-order meta-training of a dynamics belief against a POLICY outer loss.
# Inner: LatentTrackModel (object-only track error on elapsed windows).
# Outer: JointPolicyDiT (action chunk + scene tracks, conditioned on the belief).
#
# RUN THE GATE FIRST (train_policy.sh none / mass). A latent can only recover
# what the true value carries, so if cond=mass does not beat cond=none on
# val/action there is nothing here to learn.
#
# REQUIRED, no defaults -- both decide what the numbers mean:
#   METAP_EVAL_DATA     the frozen eval corpus (never streamed, never trained on)
#   METAP_VAL_IDS_FROM  a reference ckpt whose val_ids pin the held-out episodes
#
# Useful:
#   METAP_INIT_POLICY=<gate none/best.ckpt>  warm-start the policy (FiLM heads
#                                            stay at init; asserted)
#   METAP_INIT_FROM=<metapolicy best.ckpt>   CONTINUE both models (phase 2)
#   METAP_FILM_SCALE / METAP_ALPHA_SET / METAP_ALPHA_FREEZE
#                                            the belief's leverage. Measure with
#                                            _probe_film_scale.py before changing;
#                                            alpha is a TRAVEL DISTANCE, and if it
#                                            is left free the meta-loss shrinks it
#                                            to escape an adaptation that is not
#                                            yet helping.
#   METAP_HIST_DROP=1.0                      the no-adaptation control
#   METAP_SHUFFLE_Q=1                        score with the WRONG episode's belief
#   METAP_DETACH_Q=1                         FOMAML instead of full second order
#
# The shard window is train_stream_tracks.ShardWindow, so its knobs are STREAM_*.
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE
mkdir -p "$TORCH_EXTENSIONS_DIR"

GPU=${1:-0}
export METAP_DEVICE=cuda:${GPU}

export STREAM_DIR=${STREAM_DIR:-/home/user/DyWA/output/mass/meta_stream}
export STREAM_WINDOW_SHARDS=${STREAM_WINDOW_SHARDS:-30}
export STREAM_MIN_SHARDS=${STREAM_MIN_SHARDS:-4}
export STREAM_DINO_DIM=${STREAM_DINO_DIM:-64}
export STREAM_WORKERS=${STREAM_WORKERS:-6}
# Eviction DELETES the evicted shard. Set 0 explicitly when a collector is live
# and bounded disk is the point.
export STREAM_KEEP_SHARDS=${STREAM_KEEP_SHARDS:-1}

export METAP_SAVE_DIR=${METAP_SAVE_DIR:-/home/user/DyWA/output/mass/ckpt_metapolicy}
export METAP_TOTAL_STEPS=${METAP_TOTAL_STEPS:-400000}
export METAP_BLOCK_STEPS=${METAP_BLOCK_STEPS:-5000}

if [ -z "${METAP_EVAL_DATA}" ]; then
  echo "METAP_EVAL_DATA is required (the frozen eval corpus)."; exit 1; fi
if [ -z "${METAP_VAL_IDS_FROM}" ]; then
  echo "METAP_VAL_IDS_FROM is required (reference ckpt pinning the val ids)."; exit 1; fi

echo "[metapolicy] gpu=$GPU z_dim=${METAP_Z_DIM:-32} K=${METAP_K:-1}"\
" m_max=${METAP_M_MAX:-8} every=${METAP_EVERY:-16} query=${METAP_QUERY:-4}"
echo "[metapolicy] stream=$STREAM_DIR window=$STREAM_WINDOW_SHARDS shards"
echo "[metapolicy] steps=$METAP_TOTAL_STEPS block=$METAP_BLOCK_STEPS"\
" hist_drop=${METAP_HIST_DROP:-0.3} detach_q=${METAP_DETACH_Q:-0}"
echo "[metapolicy] init_policy=${METAP_INIT_POLICY:-<scratch>}"\
" init_from=${METAP_INIT_FROM:-<none>} film=${METAP_FILM_SCALE:-10}"\
" alpha=${METAP_ALPHA_SET:-${METAP_ALPHA0:-0.89}} frozen=${METAP_ALPHA_FREEZE:-0}"
echo "[metapolicy] keep_shards=$STREAM_KEEP_SHARDS (0 = bounded disk, needs a live collector)"

PYTORCH_JIT=0 python3 -u train_meta_policy.py
