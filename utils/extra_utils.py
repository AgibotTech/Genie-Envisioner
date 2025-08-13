import numpy as np
import torch
import random


def act_metric(preds, gts, prefix='val', start_stop_interval=[(0,1),(1,9),(9,25),(25,57)]):
    """
    inputs:
        preds    : b, t, nc_act
        gts      : b, t, nc_act
        start_stop_interval: how to split action predictions along the temporal dimension, like [(0, t-1), (t-1, t)]
    outputs:
        MSE of actions
    """
    assert preds.shape == gts.shape
    assert start_stop_interval[0][0] == 0 and start_stop_interval[-1][-1] == preds.shape[1]
    logs = {}
    for i in range(preds.shape[-1]):
        dim_delta = (preds[:,:,i] - gts[:,:,i]) ** 2
        dim_mean = dim_delta.mean(axis=0)
        dim_std = dim_delta.std(axis=0)
        for h_start, h_stop in start_stop_interval:
            logs[f'{prefix}/{h_start}_{h_stop}_dim_{i}_diff'] = np.mean(dim_mean[h_start:h_stop])
            logs[f'{prefix}/{h_start}_{h_stop}_dim_{i}_std'] = np.mean(dim_std[h_start:h_stop])

    return logs