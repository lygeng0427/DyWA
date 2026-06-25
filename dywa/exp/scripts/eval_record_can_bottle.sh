#!/bin/bash
# Record success + failure rollout mp4 videos for the held-out (TEST) can and
# bottle instances. Runs the same student/teacher DAgger eval as
# eval_student_unseen_obj.sh, but:
#   * restricts the object set to the test can+bottle (can_bottle_test.json)
#   * wraps the env with NvdrRecordEpisode (episode_type=both)
#
# Output mp4s land under <record_dir>/{succ,fail}/env_XXXX/episode_*_<objkey>.mp4
# so each clip is tagged with its object key (can/bottle + scale).
#
# MUST be run inside the DyWA Docker container (needs Isaac Gym).
cd /home/user/DyWA/dywa/exp/train

export PYTHONPATH=/opt/isaacgym/python:/home/user/DyWA:$PYTHONPATH
export TORCH_EXTENSIONS_DIR=/tmp/docker/torch_extensions
mkdir -p "$TORCH_EXTENSIONS_DIR"

name='dywa'
root="/home/user/DyWA/output/test_rma"
record_dir="${root}/${name}/rollouts/can_bottle"

GPU=${1:-0}

mkdir -p "$record_dir"

PYTORCH_JIT=0 python3 test_rma.py \
+platform=debug \
+env=abs_goal_1view \
+run=teacher_base \
+student=dywa/base \
++name="$name" \
++path.root="${root}/${name}" \
++env.num_env=12 \
++global_device=cuda:${GPU} \
++student.norm="ln" \
+load_student=/input/pretrained/Dywa_abs_1view/ckpt/last.ckpt \
++load_ckpt=/input/pretrained/Dywa_abs_1view/ckpt/teacher-last.ckpt \
++icp_obs.icp.ckpt=/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920 \
++plot_pc=0 \
++dagger_train_env.anneal_step=1 \
++add_teacher_state=1 \
++student.decoder.film_mlp=1 \
++env.single_object_scene.filter_file=/input/DGN/can_bottle_test.json \
++env.single_object_scene.mode=valid \
++log_categorical_results=True \
++record_episode=True \
++record_episode_cfg.episode_type=both \
++record_episode_cfg.record_dir="${record_dir}" \
++test_step=3000 \
# &> "$root/$name/record.out"
