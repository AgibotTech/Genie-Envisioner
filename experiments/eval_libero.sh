#!/usr/bin/bash

gpu=${1:-0}

config_file=configs/ltx_model/libero/action_model_libero.yaml

output_dir=evaluation_results/libero

# base line checkpoints
# ckpt_path_goal=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/Genie-Envisioner/ge_act_libero_goal.safetensors
# ckpt_path_10=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/Genie-Envisioner/ge_act_libero_10.safetensors
# ckpt_path_obj=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/Genie-Envisioner/ge_act_libero_object.safetensors
# ckpt_path_spa=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/Genie-Envisioner/ge_act_libero_spatial.safetensors

# no video adaption and trained on full data sinel checkpoint - Sajjad
ckpt_path_goal=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/output/2026_01_13_19_53_44/step_110000/diffusion_pytorch_model.safetensors
ckpt_path_10=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/output/2026_01_13_19_53_44/step_110000/diffusion_pytorch_model.safetensors
ckpt_path_obj=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/output/2026_01_13_19_53_44/step_110000/diffusion_pytorch_model.safetensors
ckpt_path_spa=/shared_work/physical_intelligence/sajjad/projects/Genie-Envisioner/output/2026_01_13_19_53_44/step_110000/diffusion_pytorch_model.safetensors

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
    --exec_step 8 \
    --task_suite_name  libero_10 \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 20


EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_obj \
    --exec_step 8 \
    --task_suite_name  libero_object \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 30

EGL_DEVICE_ID=$gpu python  experiments/eval_libero.py \
    --config_file $config_file \
    --output_dir  $output_dir \
    --ckpt_path $ckpt_path_spa \
    --exec_step 8 \
    --task_suite_name  libero_spatial \
    --device $gpu \
    --num_trails_per_task 50 \
    --threshold 30