#!/usr/bin/env python3

import sys
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

from typing import (
    List,
    Dict,
    Tuple,
    Union,
    Optional,
    Any,
    Callable,
    Iterable,
    Type,
    Sequence,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from eus_imitation.utils.python_utils import extract_class_init_kwargs_from_dict
import eus_imitation.util.tensor_utils as TensorUtils
import eus_imitation.util.obs_utils as ObsUtils
from eus_imitation.base.base_nets import (
    Module,
    Sequential,
    MLP,
    RNN_Base,
    ResNet18Conv,
    SpatialSoftmax,
    FeatureAggregator,
)
from eus_imitation.models.obs_core import VisualCore, Randomizer


class RNN_MIMO_MLP(Module):
    def __init__(
        self,
        input_obs_group_shapes: Union[
            Dict[str, Tuple[int, ...]], List[Tuple[int, ...]]
        ],
        output_shapes,
        mlp_layer_dims: List[int],
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        rnn_type: str = "LSTM",
        rnn_kwargs: Optional[Dict] = None,
        mlp_activation: Optional[Callable] = nn.ReLU,
        mlp_layer: Optional[Type[nn.Module]] = nn.Linear,
        per_step=True,
        encoder_kwargs=None,
    ):
        pass
