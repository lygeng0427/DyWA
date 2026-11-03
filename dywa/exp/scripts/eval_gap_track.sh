#!/bin/bash
# Usage: bash eval_gap_track.sh <ckpt_dir_or_file> [gpu]
#
# Held-out tracking error in mm, over three subsets, each against a
# CONSTANT-ZERO-PREDICTION baseline. Read the `moving` row: `valid` is GAP's
# own group, a validity mask that includes the static table, so a model that
# emits zeros everywhere scores well on it.
set -e
CKPT=${1:?path to a .ckpt or a run dir}
GPU=${2:-0}
# best.ckpt: with no latent, val loss is a plain held-out number and is a valid
# selection signal again. (It was disabled while a per-episode belief existed,
# because val latent rows never received a gradient.)
[ -d "$CKPT" ] && CKPT="$CKPT/best.ckpt"
export GAP_CKPT=$CKPT GAP_DEVICE=cuda:${GPU}
cd /home/user/DyWA/dywa/exp/train
PYTORCH_JIT=0 python3 -u eval_gap_track.py
