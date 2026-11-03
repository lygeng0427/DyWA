#!/bin/bash
# Usage: bash eval_fric_track.sh <ckpt_dir_or_file> [GPU]
cd /home/user/DyWA/dywa/exp/train
export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:/home/user/DyWA/dywa/src:$PYTHONPATH
CK=${1:?need a checkpoint dir or file}
[ -d "$CK" ] && CK="$CK/best.ckpt"
export FRIC_CKPT="$CK"
export FRIC_DEVICE=cuda:${2:-0}
PYTORCH_JIT=0 python3 -u eval_fric_track.py
