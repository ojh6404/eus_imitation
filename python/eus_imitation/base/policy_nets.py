#!/usr/bin/env python3

from abc import abstractmethod
import textwrap
import numpy as np
from collections import OrderedDict

from typing import Dict, List, Optional, Tuple, Union, Callable
import yaml
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchvision import models as vision_models
from torchvision import transforms

from eus_imitation.base.base_nets import MLP, RNN, AutoEncoder
import eus_imitation.util.tensor_utils as TensorUtils

"""
Policy Networks
flow : ObservationEncoder -> ActorCore[MLP, RNN, ...] -> MLPDecoder
"""

class ModalityEncoderBase(nn.Module):
    def __init__(self):
        super(ModalityEncoderBase, self).__init__()

    @abstractmethod
    def process_obs(self, obs):
        """
        Process observations into a form that can be fed into the encoder.
        obs input is expected to be a numpy ndarrays.
        """
        return TensorUtils.to_tensor(obs)


class ImageModalityEncoder(ModalityEncoderBase):
    def __init__(self, cfg: Dict) -> None:
        super(ImageModalityEncoder, self).__init__()


        self.cfg = cfg



        self.pretrained = cfg.pretrained
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim

        has_decoder = cfg.has_decoder

        autoencoder = AutoEncoder( # for test
            input_size=cfg.input_dim[:2],
            input_channel=cfg.input_dim[2],
            latent_dim=16,
            normalization=nn.BatchNorm2d,
        )


        self.nets = nn.ModuleDict()

        self.nets["encoder"] = autoencoder.encoder
        if has_decoder:
            self.nets["decoder"] = autoencoder.decoder

    def process_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Images like (B, T, H, W, C) or (B, H, W, C) or (H, W, C) torch tensor or numpy ndarray of uint8.
        Processing obs into a form that can be fed into the encoder like (B*T, C, H, W) or (B, C, H, W) torch tensor.
        """
        assert len(obs.shape) == 5 or len(obs.shape) == 4 or len(obs.shape) == 3
        obs = TensorUtils.to_float(TensorUtils.to_tensor(obs)) # to torch float tensor
        obs = TensorUtils.to_device(obs, "cuda:0") # to cuda
        # to Batched 4D tensor
        obs = obs.view(-1, obs.shape[-3], obs.shape[-2], obs.shape[-1])
        # to BHWC to BCHW and contigious
        obs = TensorUtils.contiguous(obs.permute(0, 3, 1, 2))
        # normalize
        obs /= 255.0 # to [0, 1] of [B, C, H, W] torch float tensor
        return obs

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Images like (B, T, H, W, C) or (B, H, W, C) or (H, W, C) torch tensor or numpy ndarray of uint8.
        return latent like (B*T, D) or (B, D) or (D) torch tensor.
        """
        assert len(obs.shape) == 5 or len(obs.shape) == 4 or len(obs.shape) == 3
        if len(obs.shape) ==5:
            batch_size, seq_len, height, width, channel = obs.shape
        elif len(obs.shape) == 4:
            batch_size, height, width, channel = obs.shape
        else: # len(obs.shape) == 3
            height, width, channel = obs.shape
        processed_obs = self.process_obs(obs)
        latent = self.nets["encoder"](processed_obs)
        if len(obs.shape) == 5:
            latent = latent.view(batch_size, seq_len, -1) # (B, T, D)
        elif len(obs.shape) == 4:
            latent = latent.view(batch_size, -1) # (B, D)
        else: # len(obs.shape) == 3
            latent = latent.view(-1) # (D)
        return latent


class FloatVectorModalityEncoder(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super(FloatVectorModalityEncoder, self).__init__()

        self.cfg = cfg
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        self.layer_dims = cfg.layer_dims
        self.activation = eval("nn." + cfg.get("activation", "ReLU"))

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

    # input numpy ndaarray or torch tensor
    def process_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Vector like (B, T, D) or (B, D) or (D) torch tensor or numpy ndarray of float.
        Processing obs into a form that can be fed into the encoder like (B*T, D) or (B, D) torch tensor.
        """
        assert len(obs.shape) == 3 or len(obs.shape) == 2 or len(obs.shape) == 1
        obs = TensorUtils.to_float(TensorUtils.to_tensor(obs)) # to torch float tensor
        obs = TensorUtils.to_device(obs, "cuda:0") # to cuda
        obs = TensorUtils.contiguous(obs) # to contigious
        obs = obs.view(-1, obs.shape[-1]) # to BD
        return obs


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
        else: # len(obs.shape) == 1
            dim = obs.shape[0]
        processed_obs = self.process_obs(obs)
        vector = self.nets(processed_obs)
        if len(obs.shape) == 3:
            vector = vector.view(batch_size, seq_len, -1)
        elif len(obs.shape) == 2:
            vector = vector.view(batch_size, -1)
        else: # len(obs.shape) == 1
            vector = vector.view(-1)
        return vector


class ObservationEncoder(nn.Module):
    """
    Encodes observations into a latent space.
    """

    def __init__(self, cfg) -> None:
        super(ObservationEncoder, self).__init__()
        self.cfg = cfg
        self._build_network()

    def _build_network(self) -> None:
        self.nets = nn.ModuleDict()
        for key in self.cfg.keys():
            modality_encoder = eval(self.cfg[key]["modality"] + "Encoder")
            self.nets[key] = modality_encoder(self.cfg[key]["obs_encoder"])

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


class Actor(nn.Module):
    """
    Base class for actor networks.
    """

    @abstractmethod
    def _build_network(self):
        pass


class RNNActor(Actor):
    pass

    def __init__(self, cfg: Dict) -> None:
        super(RNNActor, self).__init__()
        self.cfg = cfg

        # print("cfg: ", cfg)

        self.obs = cfg.obs

        self.policy_type = cfg.policy.type
        self.rnn_type = cfg.policy.rnn.type
        self.rnn_num_layers = cfg.policy.rnn.rnn_num_layers
        self.rnn_hidden_dim = cfg.policy.rnn.rnn_hidden_dim
        self.rnn_input_dim = sum(
            [self.obs[key]["obs_encoder"]["output_dim"] for key in self.obs.keys()]
        )
        self.rnn_kwargs = cfg.policy.rnn.get("kwargs", {})
        self.action_dim = cfg.actions.dim

        self.mlp_layer_dims = cfg.policy.mlp_layer_dims
        self.mlp_activation = eval("nn." + cfg.policy.get("mlp_activation", "ReLU"))

        self.nets = nn.ModuleDict()

        self._build_network()

    def _build_network(self) -> None:
        """
        Build the network.
        inputs passed to obs_encoder -> rnn -> mlp_decoder
        """
        self.nets["obs_encoder"] = ObservationEncoder(self.obs)
        self.nets["mlp_decoder"] = MLP(
            input_dim=self.rnn_hidden_dim,
            layer_dims=self.mlp_layer_dims,
            output_dim=self.action_dim,
            activation=self.mlp_activation,
        )
        self.nets["rnn"] = RNN(
            rnn_input_dim=self.rnn_input_dim,
            rnn_hidden_dim=self.rnn_hidden_dim,
            rnn_num_layers=self.rnn_num_layers,
            rnn_type=self.rnn_type,
            rnn_kwargs=self.rnn_kwargs,
            per_step_net=self.nets["mlp_decoder"],
        )

        # self.policy_nets = nn.Sequential(
        #     self.nets["obs_encoder"],
        #     self.nets["rnn"],
        # )

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        rnn_state: Optional[torch.Tensor] = None,
        return_rnn_state: bool = False,
    ):
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = self.nets["obs_encoder"](obs_dict)
        outputs, rnn_state = self.nets["rnn"](inputs=obs_latents, rnn_state=rnn_state, return_rnn_state=True)
        if return_rnn_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_step(
        self, obs_dict: Dict[str, torch.Tensor], rnn_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = self.nets["obs_encoder"](obs_dict)
        output, rnn_state = self.nets["rnn"].forward_step(obs_latents, rnn_state)
        return output, rnn_state


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        actor_cfg = edict(yaml.safe_load(f)).actor

    batch_size = 32

    # test actor
    actor = RNNActor(actor_cfg)

    test_obs_dict = {
        "image": torch.randn(batch_size, 10, 3, 224, 224).to("cuda:0"),
        "robot_ee_pos": torch.randn(batch_size, 10, 4).to("cuda:0"),
    }

    actor.to("cuda:0")

    output, rnn_state = actor(test_obs_dict)

    input("test")
