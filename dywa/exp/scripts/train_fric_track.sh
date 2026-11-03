#!/bin/bash
# Usage: bash train_fric_track.sh <trunk> <phys_cond> [GPU] [extra env already exported]
#   trunk      : transformer | dit
#   phys_cond  : none | token | film | both
# Controls are env vars: FRIC_PHYS_SHUFFLE=1, FRIC_HELDOUT_MU=8
cd /home/user/DyWA/dywa/exp/train
export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
export FRIC_TRUNK=${1:-transformer}
export FRIC_PHYS_COND=${2:-both}
GPU=${3:-0}
export FRIC_DEVICE=cuda:${GPU}
export FRIC_DATA=${FRIC_DATA:-/home/user/DyWA/output/fric/tracks_tf_a.hdf5,/home/user/DyWA/output/fric/tracks_tf_b.hdf5}
export FRIC_SAVE_DIR=${FRIC_SAVE_DIR:-/home/user/DyWA/output/fric/ckpt}
export FRIC_EPOCHS=${FRIC_EPOCHS:-10}
# Echo the resolved paths. FRIC_DATA defaults to the FRICTION dataset, so a mass
# run launched without overriding it would train happily on the wrong data and
# only show up as a channel name buried in the phys line.
echo "[train_fric] trunk=$FRIC_TRUNK phys_cond=$FRIC_PHYS_COND gpu=$GPU shuffle=${FRIC_PHYS_SHUFFLE:-0} heldout=${FRIC_HELDOUT_MU:-0} success_only=${FRIC_SUCCESS_ONLY:-0}"
echo "[train_fric] data=$FRIC_DATA"
echo "[train_fric] save=$FRIC_SAVE_DIR  epochs=$FRIC_EPOCHS  action=${FRIC_ACTION_MODE:-realized_gain}"
PYTORCH_JIT=0 python3 -u train_fric_track.py
