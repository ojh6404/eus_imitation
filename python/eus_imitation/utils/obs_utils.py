from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import yaml

import torch
import torch.nn as nn

import eus_imitation.utils.tensor_utils as TensorUtils
from eus_imitation.models.base_nets import *

from typing import Union, List, Tuple, Dict, Any, Optional


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


class ModalityEncoderBase(nn.Module):
    modality: Union[ImageModality, FloatVectorModality] = None
    # nets: Union[nn.ModuleDict, nn.ModuleList] = None


# function that gets first element inputs like (0, 1, 2) or (0,1) or (0,(1,2)) or 0
# and returns 0
def get_first_element(inputs):
    if isinstance(inputs, tuple):
        return get_first_element(inputs[0])
    else:
        return inputs

class ImageModalityEncoder(ModalityEncoderBase):
    def __init__(self, cfg: Dict, obs_name: str) -> None:
        super(ImageModalityEncoder, self).__init__()

        self.cfg = cfg
        self.pretrained = cfg.pretrained
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.has_decoder = cfg.has_decoder
        self.encoder_model = cfg.model


        # self.model = eval(cfg.type)(**cfg_dict)
        self.model = eval(self.encoder_model)()
        if self.pretrained:
            self.model.load_state_dict(torch.load(cfg.model_path))

        self.modality = ImageModality(obs_name)
        self.normalize = cfg.get("normalize", False)
        if self.normalize:
            pass  # TODO: add custom normalization
        else:
            self.modality.set_scaler(mean=0.0, std=1.0)

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = self.model.nets["encoder"]
        if self.has_decoder:
            self.nets["decoder"] = self.model.nets["decoder"]

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Images like (B, T, H, W, C) or (B, H, W, C) or (H, W, C) torch tensor or numpy ndarray of uint8.
        return latent like (B*T, D) or (B, D) or (D) torch tensor.
        """
        assert len(obs.shape) == 5 or len(obs.shape) == 4 or len(obs.shape) == 3
        if len(obs.shape) == 5:
            batch_size, seq_len, height, width, channel = obs.shape
        elif len(obs.shape) == 4:
            batch_size, height, width, channel = obs.shape
        else:  # len(obs.shape) == 3
            height, width, channel = obs.shape
        processed_obs = self.modality.process_obs(obs) # to [0, 1] of [-1, C, H, W] torch float tensor

        latent, _, _ = self.nets["encoder"](processed_obs) # (B, T, D) or (B, D) or (D)
        if len(obs.shape) == 5:
            latent = latent.view(batch_size, seq_len, -1)  # (B, T, D)
        elif len(obs.shape) == 4:
            latent = latent.view(batch_size, -1)  # (B, D)
        else:  # len(obs.shape) == 3
            latent = latent.view(-1)  # (D)
        return latent


class FloatVectorModalityEncoder(nn.Module):
    def __init__(self, cfg: Dict, obs_name: str) -> None:
        super(FloatVectorModalityEncoder, self).__init__()

        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.layer_dims = cfg.layer_dims
        self.activation = eval("nn." + cfg.get("activation", "ReLU"))

        self.modality = FloatVectorModality(obs_name)
        self.normalize = cfg.get("normalize", False)
        if self.normalize:
            with open("./config/normalize.yaml", "r") as f:
                normalizer_cfg = dict(yaml.load(f, Loader=yaml.SafeLoader))
            max = torch.Tensor(normalizer_cfg["obs"][self.modality.name]["max"]).to(
                "cuda:0"
            )
            min = torch.Tensor(normalizer_cfg["obs"][self.modality.name]["min"]).to(
                "cuda:0"
            )
            # self.register_buffer("max", max)
            # self.register_buffer("min", min)

            self.modality.set_scaler(mean=(max + min) / 2, std=(max - min) / 2)

        self.nets = (
            MLP(
                input_dim=self.input_dim,
                layer_dims=self.layer_dims,
                output_dim=self.output_dim,
                activation=self.activation,
            )
            if self.layer_dims
            else nn.Identity()
        )



    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Vector like (B, T, D) or (B, D) or (D) torch tensor or numpy ndarray of float.
        return Vector like (B*T, D) or (B, D) or (D) torch tensor.
        """
        assert len(obs.shape) == 3 or len(obs.shape) == 2 or len(obs.shape) == 1
        if len(obs.shape) == 3:
            batch_size, seq_len, dim = obs.shape
        elif len(obs.shape) == 2:
            batch_size, dim = obs.shape
        else:  # len(obs.shape) == 1
            dim = obs.shape[0]
        processed_obs = self.modality.process_obs(obs)
        vector = self.nets(processed_obs)
        if len(obs.shape) == 3:
            vector = vector.view(batch_size, seq_len, -1)
        elif len(obs.shape) == 2:
            vector = vector.view(batch_size, -1)
        else:  # len(obs.shape) == 1
            vector = vector.view(-1)
        return vector


class ObservationEncoder(nn.Module):
    """
    Encodes observations into a latent space.
    """

    def __init__(self, cfg: Dict) -> None:
        super(ObservationEncoder, self).__init__()
        self.cfg = cfg
        self._build_encoder()

    def _build_encoder(self) -> None:
        self.nets = nn.ModuleDict()
        for key in self.cfg.keys():
            modality_encoder = eval(self.cfg[key]["modality"] + "Encoder")
            self.nets[key] = modality_encoder(self.cfg[key]["obs_encoder"], key)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = []
        for key in self.cfg.keys():
            obs_latents.append(self.nets[key](obs_dict[key]))
        obs_latents = torch.cat(obs_latents, dim=-1)
        return obs_latents
