import os
import argparse
import numpy as np
import h5py
import json
import av
import cv2



def reorganize_gesim_inputs(data_root, task_id, episode_id, save_root, sidx=0, eidx=-1, n_mem=4, cams=["head", "hand_left", "hand_right"]):
    """ This function reorganize the data in AgiBotWorld to fit the script infer_gesim.py:
        AgiBotWorld:
            ...
            observations:
                296:
                    655235:
                        videos:
                            head_color.mp4
                            ...
            parameters:
                296:
                    655235:
                        parameters:
                            camera:
                                head_extrinsic_params_aligned.json 
                                ...
                ...
            proprio_stats:
                296:
                    655235:
                        proprio_stats.h5
                ...
    """
    
    os.makedirs(save_root, exist_ok=True)

    ### get actions
    h5_file = os.path.join(data_root, "proprio_stats", task_id, episode_id, "proprio_stats.h5")
    with h5py.File(h5_file, "r") as fid:
        all_abs_gripper = np.array(fid[f"state/effector/position"], dtype=np.float32)[sidx:eidx]
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)[sidx:eidx]
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)[sidx:eidx]

    ### actions: t, 16
    ### 7-quat_l, 1-gripper_l, 7-quat_r, 1-gripper_r)
    actions = np.concatenate((all_ends_p[:,0,], all_ends_o[:,0,], all_abs_gripper[:,:1], all_ends_p[:,1,], all_ends_o[:,1,], all_abs_gripper[:,1:]), axis=1)
    print(actions.shape)
    np.save(os.path.join(save_root, "actions.npy"), actions)


    ### get extrinsics and intrinsics
    episode_root = os.path.join(data_root, "parameters", task_id, episode_id)
    for cam in cams:
        c2ws = []
        with open(os.path.join(episode_root, "parameters", "camera", f"{cam}_extrinsic_params_aligned.json")) as f:
            ex_infos = json.load(f)
            for i, ex in enumerate(ex_infos):
                c2w = np.eye(4)
                R = ex["extrinsic"]["rotation_matrix"]
                T = ex["extrinsic"]["translation_vector"]
                c2w[:3,:3] = R
                c2w[:3,3] = T
                c2ws.append(c2w)
        ### t,4,4
        c2ws = np.stack(c2ws, axis=0)[sidx:eidx]
        np.save(os.path.join(save_root, f"extrinsic_{cam}.npy"), c2ws)

        with open(os.path.join(episode_root, "parameters", "camera", f"{cam}_intrinsic_params.json")) as f:
            intrinsic_info = json.load(f)["intrinsic"]
            intrinsic = np.eye(3)
            intrinsic[0,0] = intrinsic_info["fx"]
            intrinsic[1,1] = intrinsic_info["fy"]
            intrinsic[0,2] = intrinsic_info["ppx"]
            intrinsic[1,2] = intrinsic_info["ppy"]
        ### 3,3
        np.save(os.path.join(save_root, f"intrinsic_{cam}.npy"), intrinsic)


    ### get frames
    for cam in cams:
        os.makedirs(os.path.join(save_root, f"{cam}_color"), exist_ok=True)
        video_path = os.path.join(data_root, "observations", task_id, episode_id, "videos", f"{cam}_color.mp4")
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            cnt = 0
            for i, frame in enumerate(container.decode(video_stream)):
                if i>=sidx and i<sidx+n_mem: ### The first frame is used as memories.
                    frame_ndarray = frame.to_ndarray(format='bgr24')
                    cv2.imwrite(os.path.join(save_root, f"{cam}_color", f"{cnt}.png"), frame_ndarray)
                    cnt+=1
                if i>=sidx+n_mem:
                    break


def args_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--data_root')
    parser.add_argument('--task_id')
    parser.add_argument('--episode_id')
    parser.add_argument('--save_root')
    parser.add_argument('--valid_start', default=0)
    parser.add_argument('--valid_end', default=300)
    parser.add_argument('--n_mem', default=4)

    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = args_parser()

    reorganize_gesim_inputs(args.data_root, args.task_id, args.episode_id, args.save_root, int(args.valid_start), int(args.valid_end), int(args.n_mem))
