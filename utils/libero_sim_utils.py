import os
import time

import torch
import numpy as np

import imageio
import math

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
import robosuite.utils.transform_utils as T

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"{rollout_dir}/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def get_libero_state(obs):
    state = np.concatenate((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]) )
    return state

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_env(task, image_height=256, image_width=256, control_freq=20, floor_color=-1, wall_color=-1, camera_names=["agentview", "robot0_eye_in_hand"]):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": image_height, "camera_widths": image_width, "control_freq": control_freq,  "camera_names": camera_names}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def get_libero_image(obs):
    agtview_img = obs["agentview_image"]
    agtview_img = agtview_img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    wrist_img = obs["robot0_eye_in_hand_image"]
    wrist_img = wrist_img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    
    return agtview_img, wrist_img

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None, extra_info=None):
    """Saves an MP4 replay of an episode."""
    if extra_info is None:
        rollout_dir = f"{rollout_dir}/{DATE}" 
    else:
        rollout_dir = f"{rollout_dir}/{extra_info}/{DATE}" 
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path
