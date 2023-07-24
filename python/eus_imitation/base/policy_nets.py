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
from eus_imitation.util.obs_utils import ImageModality, FloatVectorModality


"""
Policy Networks
flow : ObservationEncoder -> ActorCore[MLP, RNN, ...] -> MLPDecoder
"""


class ModalityEncoderBase(nn.Module):
    modality: Union[ImageModality, FloatVectorModality] = None
    # nets: Union[nn.ModuleDict, nn.ModuleList] = None


class ImageModalityEncoder(ModalityEncoderBase):
    def __init__(self, cfg: Dict, obs_name: str) -> None:
        super(ImageModalityEncoder, self).__init__()

        self.cfg = cfg
        self.pretrained = cfg.pretrained
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        has_decoder = cfg.has_decoder

        self.modality = ImageModality(obs_name)
        self.normalize = cfg.get("normalize", False)
        if self.normalize:
            pass  # TODO: add custom normalization
        else:
            self.modality.set_scaler(mean=0.0, std=1.0)

        autoencoder = AutoEncoder(  # for test
            input_size=[224, 224],
            input_channel=3,
            latent_dim=16,
            normalization=nn.BatchNorm2d,
        )

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = autoencoder.encoder
        if has_decoder:
            self.nets["decoder"] = autoencoder.decoder

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
        processed_obs = self.modality.process_obs(obs)
        latent = self.nets["encoder"](processed_obs)
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
                normalizer_cfg = edict(yaml.load(f, Loader=yaml.SafeLoader))
            max = torch.Tensor(normalizer_cfg[self.modality.name]["obs_max"]).to(
                "cuda:0"
            )
            min = torch.Tensor(normalizer_cfg[self.modality.name]["obs_min"]).to(
                "cuda:0"
            )

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
        return_rnn_state: bool = False,
    ):
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
