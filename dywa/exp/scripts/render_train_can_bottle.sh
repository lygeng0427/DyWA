#!/bin/bash
# Render one scene image per TRAINING can/bottle object
# (can_bottle_train_gallery.json: 2 cans + 19 bottles = 21 objects).
#
# Produces <gallery_dir>/<objkey>.png for each object plus a
# _gallery_montage.png overview. No policy is loaded; the env is stepped with
# zero actions and each settled object is grabbed once (arm hidden).
#
# MUST be run inside the DyWA Docker container (needs Isaac Gym).
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

name='dywa'
root="/home/user/DyWA/output/test_rma"
gallery_dir="${root}/${name}/gallery/train_can_bottle"

GPU=${1:-0}

mkdir -p "$gallery_dir"

PYTORCH_JIT=0 python3 render_train_objects.py \
+platform=debug \
+env=abs_goal_1view \
+run=teacher_base \
+student=dywa/base \
++name="$name" \
++path.root="${root}/${name}" \
++env.num_env=64 \
++global_device=cuda:${GPU} \
++student.norm="ln" \
++load_ckpt=null \
++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
++plot_pc=0 \
++env.single_object_scene.filter_file=/input/DGN/can_bottle_train_gallery.json \
++env.single_object_scene.mode=valid \
++gallery_dir="${gallery_dir}" \
++gallery_max_steps=1500 \
++gallery_settle=25 \
# &> "$root/$name/gallery.out"
