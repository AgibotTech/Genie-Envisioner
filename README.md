# Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation

<div id="top" align="center">

![Overview](figs/overview.png)

 <a href='https://arxiv.org/abs/2508.05635'><img src='https://img.shields.io/badge/arXiv-2508.05635-b31b1b.svg'></a> &nbsp; <a href='https://genie-envisioner.github.io/'><img src='https://img.shields.io/badge/Site-GenieEnvisioner-blue'></a> &nbsp;


</div>

This repo is the official implementation of Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation.


## News

- [2025.08.14] 🚀 Weights of [GE-Base](https://huggingface.co/agibot-world/Genie-Envisioner) has been released.

- [2025.08.13] 🚀 Codes of Genie Envisioner has been released.

- [2025.08.08] 📄 The technical report [Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation](https://arxiv.org/abs/2508.05635) has been released.

- [2025.05.16] 🚀 [EWMB (Embodied World Model Benchmark)](https://github.com/AgibotTech/EWMBench) has been released.


## TODO
- [x] Release inference & training code
- [x] Release model weights
- [ ] Support more backbone models



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

1. Download the pretrained weights of [GE-base](https://huggingface.co/agibot-world/Genie-Envisioner/blob/main/GE_base_fast_v0.1.safetensors) and the weights of tokenizer and vae used in LTX_Video from [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/tree/main), and modify the model weight config in `configs/ltx_model/video_model.yaml`:
    ```
    pretrained_model_name_or_path: PATH/TO/PRETRAINED_WEIGHTS_OF_VAE_AND_TOKENIZER
    diffusion_model:
    model_path: PATH/TO/GE_base_{version}.safetensors
    ```

2. Build your own LeRoBot dataset following the instruction in [LeRoBot](https://github.com/huggingface/lerobot) and [a conversion script of AgiBotWorld](https://github.com/OpenDriveLab/AgiBot-World/blob/main/scripts/convert_to_lerobot.py).

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

3. Calculate the action statistics and add them to ``data/utils/statistics.py``.
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

4. Modified the config in ``configs/ltx_model/policy_model_lerobot.yaml``:
    ```
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


5. Enable action-model as bellow in `configs/ltx_model/policy_model_lerobot.yaml`:
    ```
    return_action: True
    return_video: False
    diffusion_model:
        config:
            action_expert: True
    ```
    Run
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

3. Download the weights of tokenizer and vae used in LTX_Video from [HuggingFace](https://huggingface.co/Lightricks/LTX-Video/tree/main) and the pretrained weights of [GE-Base](https://huggingface.co/agibot-world/Genie-Envisioner/blob/main/GE_base_fast_v0.1.safetensors), and modify the model weight config in `configs/ltx_model/video_model.yaml`:
    ```
    pretrained_model_name_or_path: PATH/TO/PRETRAINED_WEIGHTS_OF_VAE_AND_TOKENIZER
    diffusion_model:
    model_path: PATH/TO/GE_base_{version}.safetensors
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

- The previous version EnerVerse of Genie-Envisioner. [Paper](https://arxiv.org/abs/2508.05635)

- The previous version EnerVerse-AC of GE-Sim. [Paper](https://arxiv.org/abs/2505.09723) [Github](https://github.com/AgibotTech/EnerVerse-AC)

- The Embodied World Model BenchMark. [Paper](https://arxiv.org/abs/2505.09694) [Github](https://github.com/AgibotTech/EWMBench)

- The [AgiBotWorld Dataset](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)

- The LTX-Video Model [Paper](https://arxiv.org/abs/2501.00103) [Github](https://github.com/Lightricks/LTX-Video)



## License

All the data and code within this repo are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
