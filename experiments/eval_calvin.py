import os
import shutil
import json
import glob
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import time
import sys
root_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, root_dir)

from collections import deque, Counter


from web_infer_utils.MVActor import MVActor

import cv2
import av
import torch

import argparse
import traceback
import hydra
from tqdm.auto import tqdm


from utils.calvin_env_wrapper import CalvinEnvWrapperRaw

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
from pytorch_lightning import seed_everything


MAX_STEP = 360
CALVIN_ROOT = "/path/to/the/path/of/git-cloned/Calvin" ### git clone --recurse-submodules https://github.com/mees/calvin.git
CALVIN_DATASET = "/path/to/the/root/of/calvin/dataset"


def print_and_save(results, sequences, eval_result_path, epoch=None, sidx=0, eidx=1001):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = np.mean(results)
    chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": cnt_success[task], "total": total[task]}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

    current_data[epoch] = data

    with open(os.path.join(eval_result_path, f's{sidx}e{eidx}.json'), "w") as file:
        json.dump(chain_sr, file)

    print()
    previous_data = {}
    json_data = {**previous_data, **current_data}
    with open(os.path.join(eval_result_path, f'result_s{sidx}e{eidx}.json'), "w") as file:
        json.dump(json_data, file)
    print(
        f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
        f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
    )



def save_vid(imgs, output_filename, fps=30):

    # 1. Open the output file in write mode
    output = av.open(output_filename, 'w')

    # 2. Add a video stream with the H.264 codec
    stream = output.add_stream('h264', fps)
    stream.width = imgs[0].shape[1]
    stream.height = imgs[0].shape[0]
    stream.pix_fmt = 'yuv420p'
    # stream.options = {'crf': '17'}
    
    # 3. Process and encode imgs
    for img in imgs:
        frame = av.VideoFrame.from_ndarray(img, format='bgr24')
        for packet in stream.encode(frame):
            output.mux(packet)

    # 4. Flush the encoder and close the file
    for packet in stream.encode():
        output.mux(packet)

    output.close()



def get_action_func(obs, policy, prompt, num_inference_steps=5, execution_step=10):
    
    image_size = policy.args.data['val']["sample_size"]

    image_obs = np.stack([
        cv2.resize(obs['rgb_obs']['rgb_static'], dsize=image_size),
        cv2.resize(obs['rgb_obs']['rgb_gripper'], dsize=image_size),
    ], axis=0)

    state = np.array(obs['robot_obs']).astype(np.float32)
    ndim_state = state.shape[0]
    ndim_action = policy.action_dim-ndim_state
    state = np.concatenate([
        np.zeros(ndim_action),
        state,
    ])

    actions = policy.play(
        image_obs, prompt, num_inference_steps=num_inference_steps, execution_step=execution_step, state=state, state_zeropadding=(ndim_action,0), ndim_action=ndim_action
    )

    return actions


def rollout(env, policy, task_oracle, subtask, val_annotations, subtask_i, sequence_i, ep_len, num_inference_steps=10, execution_step=10):

    obs = env.get_obs()

    lang_annotation = val_annotations[subtask][0].split("\n")[0].replace("\u2019", "'")

    policy.reset()
    start_info = env.get_info()

    action_queue = deque()
    for step in range(ep_len):
        
        # get action chunk
        if len(action_queue) == 0:
            action_queue.extend(
                get_action_func(
                    obs, policy, lang_annotation,
                    num_inference_steps=num_inference_steps,
                    execution_step=execution_step,
                )
            )
        action = action_queue.popleft()
        if action[-1] < 0:
            action[-1] = -1
        else:
            action[-1] = 1
        obs, _, _, current_info = env.step(action)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            return True

    return False



def run_eval(
    env,
    policy,
    res_path,
    execution_step=8,
    device="cuda:0",
    num_total_rollouts = 1000,
    sidx=None,
    eidx=None,
):

    if sidx is None:
        sidx = 0
    if eidx is None:
        eidx = num_total_rollouts + 1

    os.makedirs(res_path, exist_ok=True)


    # eval_sequences = get_sequences(num_total_rollouts)

    with open('experiments/calvin_eval_sequences.json', 'r') as f:
        eval_sequences = json.load(f)
    
    assert(num_total_rollouts == len(eval_sequences))

    eval_sequences = eval_sequences[sidx:eidx]
    eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    ### git clone --recurse-submodules https://github.com/mees/calvin.git
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    sequence_i = 0
    results = []
    for initial_state, eval_sequence in eval_sequences:
        
        robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

        ### execute a sequence of multiple tasks
        success_counter = 0
        for subtask_i, subtask in enumerate(eval_sequence):
            success = rollout(
                env, policy, task_oracle, subtask, val_annotations, subtask_i, sequence_i, MAX_STEP,
                execution_step=execution_step,
            )
            if success:
                success_counter += 1
            else:
                break
        results.append(success_counter)
        sequence_i += 1

        success_list = count_success(results)
        eval_sequences.set_description(
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|| {:05d}/{:05d}".format(sequence_i, len(eval_sequences))
        )

    print_and_save(results, eval_sequences, res_path, None, sidx=sidx, eidx=eidx)


def load_policy(config_file, transformer_file, device, num_inference_steps=10, threshold=1):
    policy = MVActor(
        config_file=config_file,
        transformer_file=transformer_file,
        load_weights=True,
        threshold=threshold,
        domain_name="calvin",
        num_inference_steps=num_inference_steps,
        action_dim=22,
        gripper_dim=1,
        device=device,
        norm_type="minmax"
    )
    return policy



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file')
    parser.add_argument('-w', '--weight')
    parser.add_argument('-r', '--res_path')
    parser.add_argument('-d', '--device', default=0)
    parser.add_argument('-s', '--sidx', type=int, default=0)
    parser.add_argument('-e', '--eidx', type=int, default=1000)

    args = parser.parse_args()

    args.device = int(args.device)

    seed_everything(7)
    
    policy = load_policy(
        config_file=args.config_file, transformer_file=args.weight, device=f"cuda:{args.device}"
    )

    observation_space = {
        'rgb_obs': ['rgb_static', 'rgb_gripper'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']
    }
    env = CalvinEnvWrapperRaw(
        CALVIN_DATASET + "/validation", observation_space, device=args.device,
    )

    policy.reset()

    run_eval(
        env=env,
        policy=policy,
        res_path=args.res_path,
        device=f"cuda:{args.device}",
        sidx=args.sidx, eidx=args.eidx,
    )

