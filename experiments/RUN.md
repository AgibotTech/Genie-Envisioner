# Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation

## Performance of GE-Act on Simulation Benchmarks

### LIBERO

|  Split | Goal | Object | 10   | Spatial | Average |
|  ----  | ---- | ----   | ---- | ----    | ----    |
| Success Rate | 0.958 | 0.976 | 0.944 | 0.982 | 0.965 |


## Calvin

|  Split | Len-1 | Len-2 | Len-3 | Len-4 | Len-5 | Avg. Subtasks 
|  ----  | ---- | ----   | ---- | ----    | ----    | ----    |
| Success Rate | 0.950 | 0.898 | 0.857 | 0.808 | 0.747 | 4.260 


## Evaluating on LIBERO

1. Download [the weights of GE-Act trained on Calvin dataset](https://modelscope.cn/models/agibot_world/Genie-Envisioner/files)

2. Modify the path of checkpoints in ``experiments/eval_libero.sh``

3. Modify the path ``pretrained_model_name_or_path`` in ``configs/ltx_model/libero/action_model_libero.yaml``

3. Run
    ```
    bash experiments/eval_libero.sh
    ```

## Evaluating on Calvin

1. Download [the weights of GE-Act trained on Calvin dataset](https://modelscope.cn/models/agibot_world/Genie-Envisioner/files)

2. Modify the path of checkpoints in ``experiments/eval_calvin.sh``

3. Modify the path ``pretrained_model_name_or_path`` in ``configs/ltx_model/calvin/action_model_calvin.yaml``

3. Run
    ```
    bash experiments/eval_calvin.sh
    ```



## Training on LIBERO

### Prepare Dataset

1. Download LIBERO dataset from
    ```
    https://huggingface.co/datasets/openvla/modified_libero_rlds
    ```

2. Convert LIBERO Dataset to LEROBOT-format following
    ```
    https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py
    ```


### Prepare action and state statistics

1. You can use the statistic file provided in ``configs/ltx_model/libero/libero_all.json``. 

2. (Optional) If you want to create your own statistic file, you can run the script as bellow.
    ```
    python scripts/get_statistics.py --data_root PATH/TO/YOUR/DATASET --data_name libero --data_type eef --action_key actions --state_key state --save_path PATH/OF/FILE.json
    ```

### Task-specific video adaption

1. Modify the path in ``configs/ltx_model/libero/video_model_lerobot.yaml``.

2. Run
    ```
    bash scripts/train.sh main.py configs/ltx_model/libero/video_model_lerobot.yaml
    ```

### Action Post-Training

1. Modify the path in ``configs/ltx_model/libero/action_model_lerobot.yaml``.

2. Run
    ```
    bash scripts/train.sh main.py configs/ltx_model/action_model_lerobot.yaml
    ```

3. After training, the model should fit the **training data** well on the open-loop validation plot. For example, the figure below shows the open-loop accuracy of GE-Act trained on libero-10 for 50,000 steps with total batch size of 128.

![open-loop-example](./openloop_evaluation_libero_10_train.png)

