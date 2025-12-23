import logging
import os
from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
import torch

from calvin_env.envs.play_table_env import get_env
from calvin_env.utils.utils import EglDeviceNotFoundError, get_egl_device_id

logger = logging.getLogger(__name__)


class CalvinEnvWrapperRaw(gym.Wrapper):
    def __init__(self, abs_datasets_dir, observation_space, device, show_gui=False, **kwargs):
        """Environment wrapper which returns raw observations.

        Args:
            abs_datasets_dir: absolute datset directory
            observation_sapce: {'rgb_obs': ['rgb_static', 'rgb_gripper'], 'depth_obs': [], 'state_obs': ['robot_obs'], 'actions': ['rel_actions'], 'language': ['language']}
        """
        self.set_egl_device(device)
        env = get_env(
            abs_datasets_dir, show_gui=show_gui, obs_space=observation_space, **kwargs
        )
        super(CalvinEnvWrapperRaw, self).__init__(env)
        self.observation_space_keys = observation_space
        self.device = device
        self.relative_actions = "rel_actions" in self.observation_space_keys["actions"]
        logger.info(f"Initialized PlayTableEnv for device {self.device}")

    @staticmethod
    def set_egl_device(device):
        if "EGL_VISIBLE_DEVICES" in os.environ:
            logger.warning("Environment variable EGL_VISIBLE_DEVICES is already set. Is this intended?")
        # modified: cuda_id = device.index if device.type == "cuda" else 0
        cuda_id = torch.cuda.current_device()
        try:
            egl_id = get_egl_device_id(cuda_id)
        except EglDeviceNotFoundError:
            logger.warning(
                "Couldn't find correct EGL device. Setting EGL_VISIBLE_DEVICE=0. "
                "When using DDP with many GPUs this can lead to OOM errors. "
                "Did you install PyBullet correctly? Please refer to calvin env README"
            )
            egl_id = 0
        os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
        logger.info(f"EGL_DEVICE_ID {egl_id} <==> CUDA_DEVICE_ID {cuda_id}")

    def step(
        self, action_tensor: torch.Tensor
    ) -> Tuple[Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]], int, bool, Dict]:
        if self.relative_actions:
            action = action_tensor#.squeeze().cpu().detach().numpy()
            assert len(action) == 7
        else:
            if action_tensor.shape[-1] == 7:
                slice_ids = [3, 6]
            elif action_tensor.shape[-1] == 8:
                slice_ids = [3, 7]
            else:
                logger.error("actions are required to have length 8 (for euler angles) or 9 (for quaternions)")
                raise NotImplementedError
            action = np.split(action_tensor, slice_ids)
        o, r, d, i = self.env.step(action)

        obs = o # use raw observation
        return obs, r, d, i

    def reset(
        self,
        reset_info: Dict[str, Any] = None,
        batch_idx: int = 0,
        seq_idx: int = 0,
        scene_obs: Any = None,
        robot_obs: Any = None,
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        if reset_info is not None:
            obs = self.env.reset(
                robot_obs=reset_info["robot_obs"][batch_idx, seq_idx],
                scene_obs=reset_info["scene_obs"][batch_idx, seq_idx],
            )
        elif scene_obs is not None or robot_obs is not None:
            obs = self.env.reset(scene_obs=scene_obs, robot_obs=robot_obs)
        else:
            obs = self.env.reset()

        return obs # use raw observation

    def get_info(self):
        return self.env.get_info()

    def get_obs(self):
        obs = self.env.get_obs()
        return obs # use raw observation