import numpy as np
import os
import h5py
from scipy.spatial.transform import Rotation


def normalize_angles(radius):
    radius_normed = np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)
    return radius_normed


def get_actions_eef(gripper, all_ends_p=None, all_ends_o=None, slices=None, delta_act_sidx=None):

    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        ### the first frame is repeated to fill memory
        n = all_ends_p.shape[0]-1+delta_act_sidx
        slices = [0,]*(delta_act_sidx-1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_left_rpy = []
    all_right_rpy = []

    for i in slices:
        rot_l = Rotation.from_quat(all_ends_o[i, 0])
        left_rpy = np.concatenate((all_ends_p[i,0], rot_l.as_euler("xyz", degrees=False)), axis=0)
        rot_r = Rotation.from_quat(all_ends_o[i, 1])
        right_rpy = np.concatenate((all_ends_p[i,1], rot_r.as_euler("xyz", degrees=False)), axis=0)
        all_left_rpy.append(left_rpy)
        all_right_rpy.append(right_rpy)

    ### xyz, rpy
    all_left_rpy = np.stack(all_left_rpy)
    all_right_rpy = np.stack(all_right_rpy)

    ### xyz, xyzw, gripper
    all_abs_actions = np.zeros([n, 14])
    ### xyz, rpy, gripper
    all_delta_actions = np.zeros([n-delta_act_sidx, 14])
    for i in range(0, n):
        all_abs_actions[i, 0:6] = all_left_rpy[i, :6]
        all_abs_actions[i, 6] = gripper[slices[i], 0]
        all_abs_actions[i, 7:13] = all_right_rpy[i, :6]
        all_abs_actions[i, 13] = gripper[slices[i], 1]
        if i >= delta_act_sidx:
            all_delta_actions[i-delta_act_sidx, 0:6] = all_left_rpy[i, :6] - all_left_rpy[i-1, :6]
            all_delta_actions[i-delta_act_sidx, 3:6] = normalize_angles(all_delta_actions[i-delta_act_sidx, 3:6])
            all_delta_actions[i-delta_act_sidx, 6] = gripper[slices[i], 0]
            all_delta_actions[i-delta_act_sidx, 7:13] = all_right_rpy[i, :6] - all_right_rpy[i-1, :6]
            all_delta_actions[i-delta_act_sidx, 10:13] = normalize_angles(all_delta_actions[i-delta_act_sidx, 10:13])
            all_delta_actions[i-delta_act_sidx, 13] = gripper[slices[i], 1]

    return all_abs_actions, all_delta_actions


def get_actions_joint(gripper, all_joints=None, slices=None, delta_act_sidx=None, n_arm_joints=7):

    if delta_act_sidx is None:
        delta_act_sidx = 1

    if slices is None:
        ### the first frame is repeated to fill memory
        n = all_ends_p.shape[0]-1+delta_act_sidx
        slices = [0,]*(delta_act_sidx-1) + list(range(all_ends_p.shape[0]))
    else:
        n = len(slices)

    all_abs_actions = np.zeros([n, n_arm_joints*2+2])
    all_delta_actions = np.zeros([n-delta_act_sidx, n_arm_joints*2+2])
    for i in range(0, n):
        i_joint_l = all_joints[slices[i]][:n_arm_joints]
        i_joint_r = all_joints[slices[i]][n_arm_joints:]
        all_abs_actions[i, :n_arm_joints] = i_joint_l
        all_abs_actions[i, n_arm_joints] = gripper[slices[i], 0]
        all_abs_actions[i, n_arm_joints+1:2*n_arm_joints+1] = i_joint_r
        all_abs_actions[i, 2*n_arm_joints+1] = gripper[slices[i], 1]   
        if i >= delta_act_sidx:
            all_delta_actions[i-delta_act_sidx, :n_arm_joints] = i_joint_l - all_joints[slices[i]-1][:n_arm_joints]
            all_delta_actions[i-delta_act_sidx, n_arm_joints] = gripper[slices[i], 0]
            all_delta_actions[i-delta_act_sidx, n_arm_joints+1:2*n_arm_joints+1] = i_joint_r - all_joints[slices[i]-1][n_arm_joints:]
            all_delta_actions[i-delta_act_sidx, 2*n_arm_joints+1] = gripper[slices[i], 1]

    return all_abs_actions, all_delta_actions


def parse_h5(h5_file, slices=None, delta_act_sidx=1, action_space="eef", n_arm_joints=7):
    """
    read and parse .h5 file, and obtain the absolute actions and the action differences
    """
    with h5py.File(h5_file, "r") as fid:
        
        all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)

        if action_space == "eef":
            all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
            all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)
            all_abs_actions, all_delta_actions = get_actions_eef(
                gripper=all_abs_gripper,
                slices=slices,
                delta_act_sidx=delta_act_sidx,
                all_ends_p=all_ends_p,
                all_ends_o=all_ends_o,
            )
        elif action_space == "joint":
            all_joints = np.array(fid["state/joint/position"])
            all_abs_actions, all_delta_actions = get_actions_joint(
                gripper=all_abs_gripper,
                slices=slices,
                delta_act_sidx=delta_act_sidx,
                all_joints=all_joints,
                n_arm_joints=n_arm_joints
            )
        else:
            raise NotImplementedError

    return all_abs_actions, all_delta_actions
