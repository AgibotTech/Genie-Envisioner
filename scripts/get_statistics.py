import os
import numpy as np
import pandas as pd
import tqdm
import json
import argparse


def load_data(data_path, key="action"):
    data = pd.read_parquet(data_path)
    data = np.stack([data[key][i] for i in range(data[key].shape[0])])
    # data = np.stack([data[key][i][0] for i in range(data[key].shape[0])])
    return data 


def cal_statistic(data, _filter=True):
    q99 = np.percentile(data, 99, axis=0)
    q01 = np.percentile(data,  1, axis=0)
    if _filter:
        data_mask = (data>=q01) & (data <= q99)
        data_mask = data_mask.min(axis=1)
        data = data[data_mask, :]
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return means, stds, q99, q01


def get_statistics(data_root, data_name, data_type, save_path, action_key="action", state_key="observation.state", nrnd=50000, _filter=True,):
    
    assert(data_type in ["joint", "eef"])

    data_path_list = os.listdir(data_root)
    data_path_list.sort()
    if nrnd <= len(data_path_list):
        data_path_list = np.random.choice(data_path_list, nrnd)

    data_list = []
    state_list = []
    delta_data_list = []
    for data_path in tqdm.tqdm(data_path_list):
        data = load_data(os.path.join(data_root, data_path), action_key)
        data_list.append(data)
        delta_data = data[1:] - data[:-1]
        delta_data_list.append(delta_data)
        state = load_data(os.path.join(data_root, data_path), state_key)
        state_list.append(state)

    data_list = np.concatenate(data_list, axis=0)
    assert(len(data_list.shape)==2)
    means, stds, q99, q01 = cal_statistic(data_list, _filter=_filter)

    delta_data_list = np.concatenate(delta_data_list, axis=0)
    assert(len(delta_data_list.shape)==2)
    delta_means, delta_stds, delta_q99, delta_q01 = cal_statistic(delta_data_list, _filter=_filter)

    state_list = np.concatenate(state_list, axis=0)
    assert(len(state_list.shape)==2)
    state_means, state_stds, state_q99, state_q01 = cal_statistic(state_list, _filter=_filter)

    ### example:
    ### data_name=agibotworld, data_type="joint"/"eef"
    ### 
    ### StatisticInfo = {
    ###     "agibotworld_joint": {
    ###         "mean": [
    ###             ...
    ###         ]
    ###         "std": [
    ###             ...
    ###         ]
    ###     "agibotworld_delta_joint": {
    ###         "mean": [
    ###             ...
    ###         ]
    ###         "std": [
    ###             ...
    ###         ]
    ### }
    ###     "agibotworld_state_joint": {
    ###         "mean": [
    ###             ...
    ###         ]
    ###         "std": [
    ###             ...
    ###         ]
    ### }

    statistics_info = dict({
        data_name+"_"+data_type:dict({
            "mean": means.tolist(),
            "std": stds.tolist(),
            "q99": q99.tolist(),
            "q01": q01.tolist(),
        }),
        data_name+"_delta_"+data_type:dict({
            "mean": delta_means.tolist(),
            "std": delta_stds.tolist(),
            "q99": delta_q99.tolist(),
            "q01": delta_q01.tolist(),
        }),
        data_name+"_state_"+data_type:dict({
            "mean": state_means.tolist(),
            "std": state_stds.tolist(),
            "q99": state_q99.tolist(),
            "q01": state_q01.tolist(),
        }),
    })

    # if os.path.exists(save_path):
    #     with open(save_path, "r") as f:
    #         exist_info = json.load(f)
    # else:
    #     exist_info = dict()
    # for k in statistics_info.keys():
    #     assert k not in exist_info

    exist_info = dict()
    exist_info.update(statistics_info)
    
    with open(save_path, "w") as f:
        json.dump(exist_info, f, indent=4)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default="PATH/TO/YOUR/DATASET")
    parser.add_argument('--data_name', default="YOUR_CUSTOM_DATASET")
    parser.add_argument('--data_type', default="joints")
    parser.add_argument('--action_key', default="action")
    parser.add_argument('--state_key', default="observation.state")
    parser.add_argument('--save_path', default="PATH/OF/JSON/FILE")

    args = parser.parse_args()

    get_statistics(
        args.data_root, args.data_name, args.data_type, args.save_path, action_key=args.action_key, state_key=args.state_key
    )
