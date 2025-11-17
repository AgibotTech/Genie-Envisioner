# Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation

<div id="top" align="center">

![Overview](figs/overview.png)

 <a href='https://arxiv.org/abs/2508.05635'><img src='https://img.shields.io/badge/arXiv-2508.05635-b31b1b.svg'></a> &nbsp; <a href='https://genie-envisioner.github.io/'><img src='https://img.shields.io/badge/Site-GenieEnvisioner-blue'></a> &nbsp;
 
Join our [WeChat Group](figs/join_wechat_group.jpg)


</div>

This repo is the official implementation of Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation.


## News

- [2025.10.22] 🚀 Pretrained Weights of [GE-Sim(Cosmos2-based version)](https://modelscope.cn/models/agibot_world/Genie-Envisioner/file/view/master/ge_sim_cosmos_v0.1.safetensors) have been released. The released GE-Sim model is pretrained on [AgiBotWorld](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta).
 
- [2025.10.22] 🚀 Example results and codes of GE-Sim (the latest version based on [Cosmos2](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)) have been released. Detailed usage can be found in [GE-Sim](#ge-sim-inference) and the example results can be found in [Example results of GE-sim](#example-results-of-ge-sim).
  
- [2025.10.17] 📄 The technical report [Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation](https://arxiv.org/abs/2508.05635) has been updated. More experimental results for GE-Act are provided. 

- [2025.08.14] 🚀 Weights of [GE_base](https://huggingface.co/agibot-world/Genie-Envisioner) have been released.

- [2025.08.13] 🚀 Codes of Genie Envisioner has been released.

- [2025.08.08] 📄 The technical report [Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation](https://arxiv.org/abs/2508.05635) has been released.

- [2025.05.16] 🚀 [EWMB (Embodied World Model Benchmark)](https://github.com/AgibotTech/EWMBench) has been released.


## TODO
- [x] Release inference & training code
- [x] Release model weights
- [x] Support more backbone models




## Getting started

### Setup

```
git clone https://github.com/AgibotTech/Genie-Envisioner.git
conda create -n genie_envisioner python=3.10.4
conda activate genie_envisioner
pip install -r requirements.txt
```

### Training

#### GE-Act Post-Training

1. Download the pretrained weights of [GE-base](https://huggingface.co/agibot-world/Genie-Envisioner/tree/main) and the weights of tokenizer and vae used in LTX_Video from [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/tree/main), and modify the model weight config in `configs/ltx_model/video_model.yaml`:
    ```
    pretrained_model_name_or_path: PATH/TO/PRETRAINED_WEIGHTS_OF_VAE_AND_TOKENIZER
    diffusion_model:
    model_path: PATH/TO/GE_base_{version}.safetensors
    ```
    Note: If you are only performing the post training phase, you do not need to download the complete LTX model weights. You only need to download the weights for the [text_encoder](https://huggingface.co/Lightricks/LTX-Video/tree/main/text_encoder), [tokenizer](https://huggingface.co/Lightricks/LTX-Video/tree/main/tokenizer) and [VAE](https://huggingface.co/Lightricks/LTX-Video/tree/main/vae), as well as the [model_index.json](https://huggingface.co/Lightricks/LTX-Video/blob/main/model_index.json), and place them in the same directory.

2. Build your own LeRoBot dataset following the instruction in [LeRoBot](https://github.com/huggingface/lerobot).

    File Structure Example:

    ```
    ROOT_PATH_TO_YOUR_DATASETS/
    ├── DATASETNAME/
    │   ├── data/
    │   │   ├── episode_000000.parquet
    │   │   ├── episode_000001.parquet
    │   │   ├── ...
    │   │   └── episode_{:06d}.parquet
    │   ├── meta/
    │   │   ├── episodes_stats.jsonl
    │   │   ├── episodes.jsonl
    │   │   ├── tasks.json
    │   │   └── info.json
    │   └── videos/
    │       ├── chunk-000/
    │       |   ├── observation.images.top_head
    │       |   |   ├── episode_000000.mp4
    │       |   |   ├── episode_000001.mp4
    │       |   |   ├── ...
    │       |   |   └── episode_{:06d}.mp4
    │       |   ├── observation.images.hand_left
    │       |   |   ├── episode_000000.mp4
    │       |   |   └── ...
    │       |   └── observation.images.hand_right
    │       |   |   ├── episode_000000.mp4
    │       |       └── ...
    |       └── ...
    └── ...
    ```

3. Calculate the action statistics. We provide an example script for LeRoBot-like datasets ``scripts/get_statistics.py`` and you can run the script as bellow:
    
    ```
    python scripts/get_statistics.py --data_root PATH/TO/YOUR/DATASET --data_name $DATASETNAM --data_type joint --action_key action --state_key observation.state --save_path PATH/OF/FILE.json
    ```

    After running the script, you can get a json file of statistics. You can specific the path of json file in configs:
    
    ```
    data:
        train:
            ...
            stat_file: PATH/OF/FILE.json
        val:
            ...
            stat_file: PATH/OF/FILE.json
    ```

    Content of the json file:
    ```
    {
        "DATASETNAME_joint": {
            "mean": [
                0,
                ...
            ],
            "std":[
                1,
                ...
            ]
        },
        "DATASETNAME_delta_joint": {
            "mean": [
                0,
                ...
            ],
            "std":[
                1,
                ...
            ]
        }
        "DATASETNAME_state_joint": {
            "mean": [
                0,
                ...
            ],
            "std":[
                1,
                ...
            ]
        }
    }
    ```
    

4. Task-specific video adaption
    
    As mentioned in our paper, although GE-base has zero-shot capability, for the unseen robots or customized new tasks, we recommend performing this step of video adaptation to achieve better performance.

    1. Modify the config in ``configs/ltx_model/video_model_lerobot.yaml``. More details of dataset can be found in ``data/utils/*_dataset.py``:
    ```
    data:
        train / val:
            data_roots:   [ROOT_PATH_TO_YOUR_DATASETS, ]
            domains:      [DATASETNAME, ]
            # rewrite to the camera names used in your dataset
            valid_cam:    ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
            ...
    ```

    2. Disable action-model as bellow in `configs/ltx_model/video_model_lerobot.yaml`:
    ```
    return_action: False
    return_video: True
    train_mode: 'video_only'
    diffusion_model:
        config:
            action_expert: False
    ```

    3. Run
    ```
    bash scripts/train.sh main.py configs/ltx_model/video_model_lerobot.yaml
    ```

5. Action Post-Training

    1. Modify the config in ``configs/ltx_model/policy_model_lerobot.yaml``
    ```
    diffusion_model:
        model_path: PATH_TO_VIDEO_POST_TRAINING_CHECKPOINT_SAFETENSOR
    data:
        train / val:
            data_roots:   [ROOT_PATH_TO_YOUR_DATASETS, ]
            domains:      [DATASETNAME, ]
            # rewrite to the camera names used in your dataset
            valid_cam:    ["observation.images.top_head", "observation.images.hand_left", "observation.images.hand_right"]
            # rewrite to the keys used in your dataset
            action_key:   "action"
            state_key:    "observation.state" 
            action_type:  "absolute"  # "absolute", "delta" or "relative"
            action_space: "joint"
            ...
    ```
    More details of dataset can be found in data/utils/*_dataset.py

    2. Enable action-model as bellow in `configs/ltx_model/policy_model_lerobot.yaml`:
    ```
    return_action: True
    return_video: False
    train_mode: 'action_full'
    diffusion_model:
        config:
            action_expert: True
    ```

    3. Run
    ```
    bash scripts/train.sh main.py configs/ltx_model/policy_model_lerobot.yaml
    ```


#### GE-base Pre-Training

You can also train GE-base on your own database. Here, we take training on AgiBotWorld as an example:

1. Download [🤗AgiBotWorld](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)

2. Modify dataset config in ``configs/ltx_model/video_model.yaml``:
    ```
    data:
        train / val:
            data_roots: ["path/to/agibot-world/AgiBotWorld-Beta", ]
            task_info_root: ["path/to/agibot-world/AgiBotWorld-Beta/task_info", ]
            domains: ["agibotworld", ]
            ...
            dataset_info_cache_path: "path/to/save/dataset_meta_info_cache"
    ```

3. Download the weights of tokenizer and vae used in LTX_Video from [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/tree/main) and the pretrained [ltx-video-2b-v0.9](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.safetensors), and modify the model weight config in `configs/ltx_model/video_model.yaml`:
    ```
    pretrained_model_name_or_path: PATH/TO/PRETRAINED_WEIGHTS_OF_VAE_AND_TOKENIZER
    diffusion_model:
     model_path: PATH/TO/PRETRAINED_MODEL.safetensor
    ```

4. Pre-train Video-Model
    ```
    bash scripts/train.sh main.py configs/ltx_model/video_model.yaml
    ``` 


### Validation

Predict actions and draw an open-loop verification diagram

```
bash scripts/infer.sh main.py \
    configs/ltx_model/policy_model_lerobot.yaml \
    path/to/trained/checkpoint.safetensors \
    path/to/save/outputs \
    DATASETNAME
```


### GE-Act Deployment

We provide a simple example of deploying GE-Act server based on [openpi](https://github.com/Physical-Intelligence/openpi):

```
# GE-Act server
# modify $IP_ADDRESS_OF_SERVER to your ip address and modify $DOMAIN_NAME to DATASETNAME
bash web_infer_scripts/run_server.sh

# A simple client that send random observations
bash web_infer_scripts/run_simple_client.sh
```


### Video Generation

You can generate videos as bellow:
```
bash scripts/infer.sh main.py \
    configs/ltx_model/video_model_infer_slow.yaml \
    path/to/trained/checkpoint.safetensors \
    path/to/save/outputs \
    DATASETNAME
```

We also provide two examples in ``video_gen_examples`` and a simple script to generate videos. As described in our paper, the video generation model takes sparse memory frames as input. Therefore, each sample in ``video_gen_examples`` includes four multi-view images sampled from history frames.

```
python video_gen_examples/infer.py \
    --config_file configs/ltx_model/video_model_infer_slow.yaml \
    --image_root video_gen_examples/sample_0 \
    --prompt_txt_file video_gen_examples/sample_0/prompt.txt \
    --output_path path/to/save/results
```

As detailed in our paper, we provide two pre-trained video generation models:

- [GE-Base-slow](https://huggingface.co/agibot-world/Genie-Envisioner/tree/main) (Mid-Range frequency video generation, synchronized with action dynamics)
- [GE-Base-fast](https://huggingface.co/agibot-world/Genie-Envisioner/tree/main) (Low-Frequency video generation optimized for low-latency applications)

When utilizing these models, please select the appropriate configuration file and ensure the ``diffusion_model.model_path`` parameter correctly points to your chosen model weights





### GE-Sim Inference

We provide an example script ``gesim_video_gen_examples/infer_gesim.py`` for GE-Sim inference. For simplicity, this script directly load extrinsics, intrinsics and actions from .npy files.

We also provide an example data-conversion script ``gesim_video_gen_examples/get_example_gesim_inputs.py`` that reorganizes the data in [AgiBotWorld](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) to fit the data format used in ``gesim_video_gen_examples/infer_gesim.py``.


```

# 1. Convert an episode to .npy files or build your custom data
#    If you only want to use the provided example data in gesim_video_gen_examples/sample_0, you can skip this step.

python gesim_video_gen_examples/get_example_gesim_inputs.py --data_root=${YOUR_AGIBOTWORLD_ROOT} --task_id=${TASK_id} --episode_id=${EPI_ID} --save_root=gesim_video_gen_examples/sample_0 --valid_start=0 --valid_end=300

# 2. Download the weights of GE-Sim(cosmos-based version) from https://modelscope.cn/models/agibot_world/Genie-Envisioner/file/view/master/ge_sim_cosmos_v0.1.safetensors

# 3. Download the scheduler config and the weights of text_encoder, tokenizers and vae of nvidia/Cosmos-Predict2-2B-Video2World from https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World

# 4. Modify the PATH in configs/cosmos_model/acwm_cosmos.yaml

# 5. Run the following command

python gesim_video_gen_examples/infer_gesim.py \
    --config_file=configs/cosmos_model/acwm_cosmos.yaml \
    --image_root=gesim_video_gen_examples/sample_0 \
    --extrinsic_root=gesim_video_gen_examples/sample_0 \
    --intrinsic_root=gesim_video_gen_examples/sample_0 \
    --action_path=gesim_video_gen_examples/sample_0/actions.npy \
    --output_path=gesim_video_gen_examples/sample_0_res
```


We provide an example function of obtaining camera-to-base extrinsics of all frames when only the action sequence and the camera-to-base extrinsc of the first frame are available. Detailed usage is provided in [``gesim_video_gen_examples/get_example_gesim_inputs.py``](https://github.com/AgibotTech/Genie-Envisioner/blob/01adfce22c2b8f9e53cb12a36ec1c1ef91420be9/gesim_video_gen_examples/get_example_gesim_inputs.py#L147C13-L147C23)


```
import scipy
from scipy.spatial.transform import Rotation

def get_cam2base(poses, init_pose=None, init_c2b=None, c2e=None):
    """
    poses:    T*7 ndarray. The following end-effection poses: T*{xyz+quat(xyzw)}
    c2e:      4x4 ndarray. The camera-to-end extrinsic
    init_pose:  7 ndarray. The initial pose: {xyz+quat(xyzw)}
    init_c2b: 4x4 ndarray. The camera-to-base extrinsic of the first frame
    """

    ### when c2e is not provided, we need to obtain c2e from init_pose and init_c2b first
    assert((init_c2b is not None and init_pose is not None) or (c2e is not None))

    ###    cam2base = end2base @ cam2end = pose @ cam2end
    ### -> cam2end = pose^-1 @ cam2base

    if c2e is None:
        ### the first pose matrix (= end-to-base) of left or right end-effector         
        pose_mat = np.eye(4)
        pose_mat[:3,:3] = Rotation.from_quat(init_pose[3:7]).as_matrix()
        pose_mat[:3,3] = init_pose[:3]

        ### Get cam2end from the first pose matrix and the first cam2base matrix
        c2e = np.dot(np.linalg.inv(pose_mat), init_c2b)

    ### Get cam2base extrinsics of each frame
    c2bs = []
    for _i in range(poses.shape[0]):
        pose_mat = np.eye(4)
        pose_mat[:3,:3] = Rotation.from_quat(poses[_i, 3:7]).as_matrix()
        pose_mat[:3,3] = poses[_i, :3]
        c2b = np.dot(pose_mat, c2e)
        c2bs.append(c2b)
    c2bs = np.stack(c2bs, axis=0)
    return c2bs

```



## Example results of GE-sim

### Example results of interaction with objects

<div align="center">
  <video src="https://github.com/user-attachments/assets/2ce55fe0-3a30-4291-8ce1-e0cc95fc5f80" width="70%"> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/e644a39f-def6-42bd-a065-c0b06f937fd2" width="70%"> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/265d7f83-b1ca-426c-af62-7be12045051f" width="70%"> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/5b77fbd5-7b2d-4450-ae97-11da345ee623" width="70%"> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/ae4cd41a-4606-4425-a82e-6d45e74249b8" width="70%"> </video>
</div>


### Example results of artificial trajectories

<div align="center">
  <video src="https://github.com/user-attachments/assets/bbbc68a7-1e0c-4080-8111-921238c66993" width="70%"> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/ce025a54-8d98-4b83-8e12-114d6d914d78" width="70%"> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/0547de3a-be9c-445a-8b00-96cbbd907491" width="70%"> </video>
</div>

<div align="center">
  <video src="https://github.com/user-attachments/assets/e81f7f5b-2b52-4c20-8c1c-c17351aa33a1" width="70%" > </video> 
</div>






## Citation
```bib
@article{liao2025genie,
  title={Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation},
  author={Liao, Yue and Zhou, Pengfei and Huang, Siyuan and Yang, Donglin and Chen, Shengcong and Jiang, Yuxin and Hu, Yue and Cai, Jingbin and Liu, Si and Luo, Jianlan, Chen Liliang, Yan Shuicheng, Yao Maoqing, Ren Guanghui},
  journal={arXiv preprint arXiv:2508.05635},
  year={2025}
}
```

## Acknowledgment

- The Genie-Envisioner team 🤗 for building Genie Envisioner [Paper](https://arxiv.org/abs/2508.05635).

- The previous version EnerVerse of Genie-Envisioner. [Paper](https://arxiv.org/abs/2501.01895)

- The previous version EnerVerse-AC of GE-Sim. [Paper](https://arxiv.org/abs/2505.09723) [Github](https://github.com/AgibotTech/EnerVerse-AC)

- The Embodied World Model BenchMark. [Paper](https://arxiv.org/abs/2505.09694) [Github](https://github.com/AgibotTech/EWMBench)

- The [AgiBotWorld Dataset](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)

- The LTX-Video Model [Paper](https://arxiv.org/abs/2501.00103) [Github](https://github.com/Lightricks/LTX-Video)

- The Cosmos Model [Github](https://github.com/nvidia-cosmos)



## License

Codes in the directory ``models/ltx_models``,  ``models/cosmos_models``, ``models/pipeline`` and ``web_infer_utils/openpi_client`` are modified from [Diffusers](https://github.com/huggingface/diffusers/), [LTX-Video](https://github.com/Lightricks/LTX-Video), [Cosmos](https://github.com/nvidia-cosmos) and [openpi](https://github.com/Physical-Intelligence/openpi), which means these codes under [Apache License 2.0](https://github.com/huggingface/diffusers/blob/main/LICENSE).

Other data and codes within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
