#!/bin/bash
# Usage: METAP_CKPT=<ckpt> bash eval_meta_policy.sh [GPU]
#
# Four columns on the pinned val episodes: blind / ttt / ceil / shuf.
# `ceil` uses the future and is an upper bound, never a result.
# `shuf` must be clearly WORSE than blind, or the model is ignoring the belief.
#
# METAP_K_EVAL=1,3,5   sweep the inner step count
# METAP_EVAL_EPS=40    subsample the val episodes (0 = all)
#
# MUST run inside the DyWA Docker container.
cd /home/user/DyWA/dywa/exp/train
export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE
GPU=${1:-0}
export METAP_DEVICE=cuda:${GPU}
export METAP_EVAL_DATA=${METAP_EVAL_DATA:-/home/user/DyWA/output/mass/tracks_m_a.hdf5,/home/user/DyWA/output/mass/tracks_m_b.hdf5}
if [ -z "${METAP_CKPT}" ]; then echo "METAP_CKPT is required."; exit 1; fi
echo "[eval] ckpt=$METAP_CKPT gpu=$GPU K=${METAP_K_EVAL:-<ckpt>}"
PYTORCH_JIT=0 python3 -u eval_meta_policy.py
