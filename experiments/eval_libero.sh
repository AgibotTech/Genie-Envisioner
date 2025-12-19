#!/usr/bin/bash

gpu=${1:-0}

config_file=configs/ltx_model_libero/action_model_libero_goal_pdstate.yaml

output_dir=evaluation_results/libero

ckpt_path_goal=PATH/TO/TRAINED/LIBERO_GOAL_CHECKPOINT.safetensor
ckpt_path_obj=PATH/TO/TRAINED/LIBERO_OBJECT_CHECKPOINT.safetensor
ckpt_path_10=PATH/TO/TRAINED/LIBERO_10_CHECKPOINT.safetensor
ckpt_path_spa=PATH/TO/TRAINED/LIBERO_SPATIAL_CHECKPOINT.safetensor


EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_goal \
    --exec_step 8 \
    --task_suite_name  libero_goal \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 20


EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_10 \
    --exec_step $exec_step \
    --task_suite_name  libero_10 \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 20


EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_obj \
    --exec_step $exec_step \
    --task_suite_name  libero_object \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 30

EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_spa \
    --exec_step $exec_step \
    --task_suite_name  libero_spatial \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 30