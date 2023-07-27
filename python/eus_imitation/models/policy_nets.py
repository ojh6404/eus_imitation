#!/usr/bin/env python3

from abc import abstractmethod
import textwrap
import numpy as np
from collections import OrderedDict

from typing import Dict, Optional, Tuple, Union, List
import yaml
from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchvision import models as vision_models
from torchvision import transforms

from eus_imitation.models.base_nets import MLP, RNN
import eus_imitation.utils.tensor_utils as TensorUtils
from eus_imitation.utils.obs_utils import ObservationEncoder, ImageModality, FloatVectorModality


"""
Policy Networks
flow : ObservationEncoder -> ActorCore[MLP, RNN, ...] -> MLPDecoder
"""


class Actor(nn.Module):
    """
    Base class for actor networks.
    """

    @abstractmethod
    def _build_network(self):
        pass


class MLPActor(Actor):
    def __init__(self, cfg: Dict) -> None:
        super(MLPActor, self).__init__()
        self.cfg = cfg
        self.obs = cfg.obs
        self.policy_type = cfg.policy.type

        self.action_dim = cfg.actions.dim
        self.mlp_layer_dims = cfg.policy.mlp_layer_dims
        self.mlp_activation = eval("nn." + cfg.policy.get("mlp_activation", "ReLU"))

        self.rnn_hidden_dim = cfg.policy.rnn.rnn_hidden_dim

        self.nets = nn.ModuleDict()

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

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
    ):
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = self.nets["obs_encoder"](obs_dict)
        actions = self.nets["mlp_decoder"](obs_latents)
        return actions


class RNNActor(Actor):
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
        self.action_modality = eval(cfg.actions.modality)

        self.normalize_cfg = edict(yaml.safe_load(open("./config/normalize.yaml", "r")))

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

    def get_rnn_init_state(self, batch_size: int, device: torch.device) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.nets["rnn"].get_rnn_init_state(batch_size, device)


    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        rnn_state: Optional[torch.Tensor] = None,
        return_rnn_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs_dict is expected to be a dictionary with keys of self.obs_keys
        like {"image": image_obs, "robot_ee_pos": robot_ee_pos_obs}
        """
        obs_latents = self.nets["obs_encoder"](obs_dict)
        outputs, rnn_state = self.nets["rnn"](
            inputs=obs_latents, rnn_state=rnn_state, return_rnn_state=True
        )

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


class TransformerActor(Actor):
    """
    Actor with Transformer encoder and MLP decoder
    """

    def __init__(self,)->None:
        pass # TODO


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        actor_cfg = edict(yaml.safe_load(f)).actor

    from eus_imitation.util.datasets import SequenceDataset
    from torch.utils.data import DataLoader

    obs_keys = ["image", "robot_ee_pos"]
    dataset_keys = ["actions"]
    dataset = SequenceDataset(
        hdf5_path=args.dataset,
        obs_keys=obs_keys,  # observations we want to appear in batches
        dataset_keys=dataset_keys,  # keys we want to appear in batches
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        # hdf5_cache_mode="all",  # cache dataset in memory to avoid repeated file i/o
        hdf5_cache_mode=None,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
    )

    test_obs = dataset[0]["obs"]  # [T, ...]

    test_obs = TensorUtils.to_device(test_obs, "cuda:0")
    test_obs = TensorUtils.to_batch(test_obs)  # [B, T, ...]

    # test actor
    actor = RNNActor(actor_cfg)

    actor.to("cuda:0")

    output = actor(test_obs)  # [B, T, D]

    input("test")

    del dataset
