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

from eus_imitation.base.base_nets import MLP, RNN
import eus_imitation.util.tensor_utils as TensorUtils

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
resnet = vision_models.resnet18(weights=vision_models.ResNet18_Weights.DEFAULT)


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
        self.resnet = resnet
        self.resnet.fc = MLP(
            input_dim=512,
            layer_dims=[256],
            output_dim=768,
            activation=nn.ReLU,
        )
        self.resnet.train()

        self.nets = nn.Sequential(
            self.resnet,
        )

    def process_obs(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Batched Time sequence  of images like (5, 10, 3, 224, 224) torch tensor is expected.
        Processing obs into a form that can be fed into the encoder.
        """
        obs = TensorUtils.to_tensor(obs)
        batch_size, seq_len, channels, height, width = obs.shape
        obs = obs.view(-1, channels, height, width)
        obs /= 255.0
        return obs

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Batched Time sequence  of images like (5, 10, 224, 224, 3) torch tensor is expected.
        return Batched Time sequence of latent vectors like (5, 10, 768) torch tensor.
        """
        batch_size, seq_len, height, width, channels = obs.shape
        obs = self.process_obs(obs)
        obs = self.nets(obs)
        obs = obs.view(batch_size, seq_len, -1)
        return obs


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
        Batched Time sequence  of float vectors like (5, 10, 4) numpy ndarray or torch tensor is expected.
        Processing obs into a form that can be fed into the encoder.
        """
        obs = TensorUtils.to_tensor(obs)
        batch_size, seq_len, dim = obs.shape
        obs = obs.view(-1, dim)
        return obs

    def forward(self, obs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Batched Time sequence  of float vectors like (5, 10, 4) numpy ndarray or torch tensor is expected.
        return Batched Time sequence of latent vectors like (5, 10, 4) torch tensor.
        """
        batch_size, seq_len, dim = obs.shape
        obs = self.process_obs(obs)
        obs = self.nets(obs)
        obs = obs.view(batch_size, seq_len, -1)
        return obs


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
        return_rnn_states: bool = False,
    ):
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """

        obs_latents = self.nets["obs_encoder"](obs_dict)
        output, rnn_state = self.nets["rnn"](obs_latents, rnn_state)
        return output, rnn_state if return_rnn_states else output

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
