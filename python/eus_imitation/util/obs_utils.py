from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn.functional as F

import eus_imitation.util.tensor_utils as TensorUtils

from typing import Union, List, Tuple, Dict, Any, Optional


IMAGE_CHANNEL_DIMS = {1, 3}


class Modality(ABC):
    def __init__(self, name: str):
        self.name = name

    def set_scaler(
        self,
        mean: Union[float, np.ndarray, torch.Tensor],
        std: Union[float, np.ndarray, torch.Tensor],
    ) -> None:
        self.mean = TensorUtils.to_tensor(mean)
        self.std = TensorUtils.to_tensor(std)

    @abstractmethod
    def _default_process_obs(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        return obs

    @abstractmethod
    def _default_unprocess_obs(self, obs: torch.Tensor) -> np.ndarray:
        return obs

    @classmethod
    def set_obs_processor(cls, processor=None):
        cls._custom_obs_processor = processor

    @classmethod
    def set_obs_unprocessor(cls, unprocessor=None):
        cls._custom_obs_unprocessor = unprocessor

    def process_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if hasattr(self, "_custom_obs_processor"):
            return self._custom_obs_processor(obs)
        else:
            return self._default_process_obs(obs)

    def unprocess_obs(self, obs: torch.Tensor) -> np.ndarray:
        if hasattr(self, "_custom_obs_unprocessor"):
            return self._custom_obs_unprocessor(obs)
        else:
            return self._default_unprocess_obs(obs)


class ImageModality(Modality):
    def __init__(self, name: str):
        super().__init__(name)

        self.num_channels: int = 3
        self.height: int = 224
        self.width: int = 224
        self.mean: Optional[torch.Tensor] = 0.0
        self.std: Optional[torch.Tensor] = 255.0

    def _default_process_obs(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Images like (B, T, H, W, C) or (B, H, W, C) or (H, W, C) torch tensor or numpy ndarray of uint8.
        Processing obs into a form that can be fed into the encoder like (B*T, C, H, W) or (B, C, H, W) torch tensor of float32.
        """
        assert len(obs.shape) == 5 or len(obs.shape) == 4 or len(obs.shape) == 3
        obs = TensorUtils.to_float(TensorUtils.to_tensor(obs))  # to torch float tensor
        obs = TensorUtils.to_device(obs, "cuda:0")  # to cuda
        # to Batched 4D tensor
        obs = obs.view(-1, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        # to BHWC to BCHW and contigious
        obs = TensorUtils.contiguous(obs.permute(0, 3, 1, 2))
        # normalize
        obs = (
            obs - self.mean
        ) / self.std  # to [0, 1] of [B, C, H, W] torch float tensor
        return obs

    def _default_unprocess_obs(self, processed_obs: torch.Tensor) -> np.ndarray:
        """
        Images like (B, C, H, W) torch tensor.
        Unprocessing obs into a form that can be fed into the decoder like (B, H, W, C) numpy ndarray of uint8.
        """
        assert len(processed_obs.shape) == 4
        # to [0, 255] of [B, C, H, W] torch float tensor
        unprocessed_obs = TensorUtils.to_numpy(processed_obs * 255.0)
        # to BCHW to BHWC
        unprocessed_obs = unprocessed_obs.transpose(0, 2, 3, 1)
        # to numpy ndarray of uint8
        unprocessed_obs = unprocessed_obs.astype(np.uint8)
        return unprocessed_obs


class FloatVectorModality(Modality):
    def __init__(self, name: str):
        super(FloatVectorModality, self).__init__(name)
        self.mean: Optional[torch.Tensor] = 0.0
        self.std: Optional[torch.Tensor] = 1.0

    def _default_process_obs(
        self, obs: Union[np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Vector like (B, T, D) or (B, D) or (D) torch tensor or numpy ndarray of float.
        Processing obs into a form that can be fed into the encoder like (B*T, D) or (B, D) torch tensor of float32.
        """
        assert len(obs.shape) == 3 or len(obs.shape) == 2 or len(obs.shape) == 1
        obs = TensorUtils.to_float(TensorUtils.to_tensor(obs))  # to torch float tensor
        obs = TensorUtils.to_device(obs, "cuda:0")  # to cuda
        obs = TensorUtils.contiguous(obs)  # to contigious
        obs = obs.view(-1, obs.shape[-1])
        obs = (obs - self.mean) / self.std  # normalize
        return obs

    def _default_unprocess_obs(self, processed_obs: torch.Tensor) -> np.ndarray:
        """
        Vector like (B, D) torch tensor.
        Unprocessing obs into a form that can be fed into the decoder like (B, D) numpy ndarray of float32.
        """
        assert len(processed_obs.shape) == 2
        unprocessed_obs = processed_obs * self.std + self.mean
        unprocessed_obs = TensorUtils.to_numpy(unprocessed_obs)
        return unprocessed_obs
