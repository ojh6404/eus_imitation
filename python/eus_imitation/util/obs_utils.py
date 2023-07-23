from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F

import eus_imitation.util.tensor_utils as TensorUtils


IMAGE_CHANNEL_DIMS = {1, 3}


class Modality(ABC):
    @abstractmethod
    def process_obs(self, obs):
        return obs

    @abstractmethod
    def unprocess_obs(self, processed_obs):
        return processed_obs

    @abstractmethod
    def _default_process_obs(self, obs):
        return obs

    @abstractmethod
    def _default_unprocess_obs(self, obs):
        return obs

    @classmethod
    def set_obs_processor(cls, processor=None):
        cls._custom_obs_processor = processor

    @classmethod
    def set_obs_unprocessor(cls, unprocessor=None):
        cls._custom_obs_unprocessor = unprocessor

    @classmethod
    def process_obs(cls, obs):
        if hasattr(cls, "_custom_obs_processor"):
            return cls._custom_obs_processor(obs)
        else:
            return cls._default_process_obs(obs)

    @classmethod
    def unprocess_obs(cls, processed_obs):
        if hasattr(cls, "_custom_obs_unprocessor"):
            return cls._custom_obs_unprocessor(processed_obs)
        else:
            return cls._default_unprocess_obs(processed_obs)


class ImageModality(Modality):
    name = "image"
    num_channels = 3
    height = 224
    width = 224

    def _default_process_obs(
        self, obs: np.ndarray, batched: bool = False
    ) -> torch.Tensor:
        if batched:
            obs = obs.transpose(0, 3, 1, 2)
        else:
            obs = obs.transpose(2, 0, 1)
        obs = TensorUtils.to_tensor(obs)

        return obs

    def _default_unprocess_obs(self, processed_obs):
        return TensorUtils.to_numpy(processed_obs)

    def process_obs(self, obs: np.ndarray, batched: bool = False) -> torch.Tensor:
        return self._default_process_obs(obs, batched)

    def unprocess_obs(self, processed_obs):
        return TensorUtils.to_numpy(processed_obs)

    def process_frame(cls, frame):
        return TensorUtils.to_tensor(frame)


class FloatVectorModality(Modality):
    name = "float_vector"

    def __init__(self, scale=1.0):
        self.scale = scale

    def _default_process_obs(self, obs):
        return TensorUtils.to_tensor(obs)

    def _default_unprocess_obs(self, processed_obs):
        return TensorUtils.to_numpy(processed_obs)

    def process_obs(self, obs):
        return TensorUtils.to_tensor(obs)

    def unprocess_obs(self, processed_obs):
        return TensorUtils.to_numpy(processed_obs)
