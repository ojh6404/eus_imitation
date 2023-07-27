#!/usr/bin/env python3


from abc import abstractmethod
from collections import OrderedDict
import numpy as np
from typing import Optional, Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as vision_models

import eus_imitation.utils.tensor_utils as TensorUtils


def calculate_conv_output_size(
    input_size: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    paddings: List[int],
) -> List[int]:
    assert len(kernel_sizes) == len(strides) == len(paddings)
    output_size = list(input_size)
    for i in range(len(kernel_sizes)):
        output_size[0] = (
            output_size[0] + 2 * paddings[i] - kernel_sizes[i]
        ) // strides[i] + 1
        output_size[1] = (
            output_size[1] + 2 * paddings[i] - kernel_sizes[i]
        ) // strides[i] + 1
    return output_size


def calculate_deconv_output_size(
    input_size: List[int],
    kernel_sizes: List[int],
    strides: List[int],
    paddings: List[int],
    output_paddings: List[int],
) -> List[int]:
    assert len(kernel_sizes) == len(strides) == len(paddings) == len(output_paddings)
    output_size = list(input_size)
    for i in range(len(kernel_sizes)):
        output_size[0] = (
            (output_size[0] - 1) * strides[i]
            - 2 * paddings[i]
            + kernel_sizes[i]
            + output_paddings[i]
        )
        output_size[1] = (
            (output_size[1] - 1) * strides[i]
            - 2 * paddings[i]
            + kernel_sizes[i]
            + output_paddings[i]
        )
    return output_size


class Reshape(nn.Module):
    """
    Module that reshapes a tensor.
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]]) -> None:
        super(Reshape, self).__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self._shape)


class Permute(nn.Module):
    """
    Module that permutes a tensor.
    """

    def __init__(self, dims: Union[List[int], Tuple[int, ...]]) -> None:
        super(Permute, self).__init__()
        self._dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(self._dims)


class Unsqueeze(nn.Module):
    """
    Module that unsqueezes a tensor.
    """

    def __init__(self, dim: int) -> None:
        super(Unsqueeze, self).__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(dim=self._dim)


class Squeeze(nn.Module):
    """
    Module that squeezes a tensor.
    """

    def __init__(self, dim: int) -> None:
        super(Squeeze, self).__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self._dim)


class SoftPositionEmbed(nn.Module):
    """
    Module that adds soft positional embeddings to a tensor.
    """

    def __init__(
        self, hidden_dim: int, resolution: Union[Tuple[int, int], List[int]]
    ) -> None:
        super(SoftPositionEmbed, self).__init__()
        self._hidden_dim = hidden_dim
        self._resolution = resolution
        self._embedding = nn.Linear(4, hidden_dim)
        self._grid = self.build_grid(resolution)  # device?

    def build_grid(self, resolution: Union[Tuple[int, int], List[int]]) -> torch.Tensor:
        ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
        grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = np.expand_dims(grid, axis=0)
        grid = grid.astype(np.float32)
        return TensorUtils.to_tensor(np.concatenate([grid, 1.0 - grid], axis=-1)).to(
            "cuda:0"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self._embedding(self._grid)
        return x + grid


class SlotAttention(nn.Module):
    """
    Module that performs slot attention.
    ref : https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        num_slots: int = 7,
        dim: int = 64,
        num_iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 128,
    ) -> None:
        super(SlotAttention, self).__init__()
        self._num_slots = num_slots
        self._num_iters = num_iters
        self._eps = eps
        self._scale = dim**-0.5
        hidden_dim = max(dim, hidden_dim)

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.xavier_uniform_(self.slots_log_sigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = MLP(
            input_dim=dim,
            output_dim=dim,
            layer_dims=[hidden_dim],
            activation=nn.ReLU,
        )

        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

    # def step(self, slots, k, v):
    #     q = self.to_q(self.norm_slots(slots))
    #     k = k * self._scale
    #     attn = F.softmax(torch.einsum("bkd,bqd->bkq", k, q), dim=-1)
    #     attn = attn / torch.sum(attn + self._eps, dim=-2, keepdim=True)
    #     updates = torch.einsum("bvq,bvd->bqd", attn, v)
    #     slots = self.gru(updates, slots)
    #     slots = slots + self.mlp(self.norm_mlp(slots))
    #     return slots

    # def iterate(self, f, x):
    #     for _ in range(self._num_iters):
    #         x = f(x)
    #     return x

    # def forward(self, inputs, slots):
    #     inputs = self.norm_input(inputs)
    #     k, v = self.to_k(inputs), self.to_v(inputs)
    #     slots = self.iterate(lambda x: self.step(x, k, v), slots)
    #     slots = self.step(slots.detach(), k, v)
    #     return slots

    def forward(self, inputs, num_slots=None):
        b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
        n_s = num_slots if num_slots is not None else self._num_slots
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.exp().expand(b, n_s, -1)
        slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        for _ in range(self._num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            dots = torch.einsum("bid,bjd->bij", q, k) * self._scale
            attn = dots.softmax(dim=1) + self._eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bjd,bij->bid", v, attn)
            slots = self.gru(updates.reshape(-1, d), slots_prev.reshape(-1, d))
            # slots = slots.reshape(b, -1, d)
            slots = slots.view(b, -1, d)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_dims: List[int],
        layer: nn.Module = nn.Linear,
        layer_kwargs: Optional[dict] = None,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super(MLP, self).__init__()
        if dropouts is not None:
            assert len(dropouts) == len(layer_dims)
        layers = []
        dim = input_dim
        layer_kwargs = layer_kwargs if layer_kwargs is not None else dict()
        for i, l in enumerate(layer_dims):
            layers.append(layer(dim, l, **layer_kwargs))
            if normalization is not None:
                layers.append(normalization(l))
            layers.append(activation())
            if dropouts is not None and dropouts[i] > 0.0:
                layers.append(nn.Dropout(dropouts[i]))
            dim = l
        layers.append(layer(dim, output_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self._layer = layer
        self._nets = layers
        self._model = nn.Sequential(*layers)

        self._layer_dims = layer_dims
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._dropouts = dropouts
        self._act = activation
        self._output_act = output_activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._model(inputs)


class RNN(nn.Module):
    def __init__(
        self,
        rnn_input_dim: int,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        rnn_type: str,
        rnn_kwargs: Optional[Dict] = None,
        per_step_net: Optional[nn.Module] = None,
    ) -> None:
        super(RNN, self).__init__()

        assert rnn_type in ["LSTM", "GRU"]
        assert per_step_net is None or isinstance(per_step_net, nn.Module)

        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        rnn_kwargs = rnn_kwargs if rnn_kwargs is not None else dict()

        self._rnn_input_dim = rnn_input_dim
        self._rnn_hidden_dim = rnn_hidden_dim
        self._rnn_num_layers = rnn_num_layers
        self._rnn_type = rnn_type
        self._per_step_net = per_step_net
        self._is_bidirectional = rnn_kwargs.get("bidirectional", False)
        self._num_directions = 2 if self._is_bidirectional else 1

        self.nets = rnn_cls(
            input_size=self._rnn_input_dim,
            hidden_size=self._rnn_hidden_dim,
            num_layers=self._rnn_num_layers,
            batch_first=True,
            **rnn_kwargs,
        )

    @property
    def rnn_type(self) -> str:
        return self._rnn_type

    def get_rnn_init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        h_0 = torch.zeros(
            self._rnn_num_layers * self._num_directions,
            batch_size,
            self._rnn_hidden_dim,
            device=device,
        )
        if self._rnn_type == "LSTM":
            c_0 = torch.zeros(
                self._rnn_num_layers * self._num_directions,
                batch_size,
                self._rnn_hidden_dim,
                device=device,
            )
            return (h_0, c_0)
        else:
            return h_0

    def forward(
        self,
        inputs: torch.Tensor,
        rnn_state: Optional[torch.Tensor] = None,
        return_rnn_state: bool = False,
    ):
        # def time_distributed(inputs, net):
        #     """
        #     function that applies a network to a time distributed input
        #     inputs : (batch_size, seq_len, ...)
        #     outputs : (batch_size, seq_len, ...)
        #     """
        #     batch_size, seq_len, = inputs.shape[:2]
        #     # inputs = inputs.reshape(-1, inputs.shape[-1])
        #     outputs = net(inputs)
        #     # outputs = outputs.reshape(batch_size, seq_len, -1)
        #     return outputs

        assert inputs.ndim == 3  # (batch_size, seq_len, input_dim)
        batch_size, _, _ = inputs.shape
        if rnn_state is None:
            rnn_state = self.get_rnn_init_state(batch_size, inputs.device)

        outputs, rnn_state = self.nets(inputs, rnn_state)
        if self._per_step_net is not None:
            outputs = self._per_step_net(outputs)
            # outputs = time_distributed(outputs, self._per_step_net)
        if return_rnn_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_step(self, inputs: torch.Tensor, rnn_state: torch.Tensor):
        """
        return rnn outputs and rnn state for the next step
        inputs : (batch_size, input_dim)
        """
        assert inputs.ndim == 2
        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(inputs, rnn_state, return_rnn_state=True)
        return outputs[:, 0, :], rnn_state  # (batch_size, rnn_hidden_dim)


class Conv(nn.Module):
    """
    Base 2D Convolutional neural network.
    inputs like (batch_size, channels, height, width)
    """

    def __init__(
        self,
        input_channel: int,
        channels: List[int],
        kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        layer: nn.Module = nn.Conv2d,
        layer_kwargs: Optional[dict] = None,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super(Conv, self).__init__()

        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        layers = []
        layer_kwargs = layer_kwargs if layer_kwargs is not None else dict()

        for i in range(len(channels)):
            if i == 0:
                in_channels = input_channel
            else:
                in_channels = channels[i - 1]
            out_channels = channels[i]
            layers.append(
                layer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    padding=paddings[i],
                    **layer_kwargs,
                )
            )
            if (
                normalization is not None and i != len(channels) - 1
            ):  # not the last layer
                layers.append(normalization(out_channels))
            if i != len(channels) - 1:  # not the last layer
                layers.append(activation())
            if dropouts is not None:
                layers.append(nn.Dropout(dropouts[i]))

        if output_activation is not None:
            layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class VisionModule(nn.Module):
    """
    inputs like uint8 (B, C, H, W) or (B, C, H, W) or (C, H, W) torch.Tensor
    """

    # @abstractmethod
    # def preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
    #     """
    #     preprocess inputs to fit the pretrained model
    #     """
    #     raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ConvEncoder(VisionModule):
    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        mean_var: bool = False,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super(ConvEncoder, self).__init__()
        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.mean_var = mean_var
        self.output_activation = (
            output_activation if output_activation is not None else lambda x: x
        )

        self.nets = nn.ModuleDict()
        self.nets["conv"] = Conv(
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            layer=nn.Conv2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=None,
        )
        self.nets["reshape"] = Reshape(
            (-1, channels[-1] * output_conv_size[0] * output_conv_size[1])
        )

        if mean_var:
            self.nets["mlp_mu"] = MLP(
                input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
                output_dim=latent_dim,
                layer_dims=[latent_dim * 4, latent_dim * 2],
                activation=activation,
                dropouts=None,
                normalization=nn.BatchNorm1d
                if normalization is not None
                else normalization,
            )
            self.nets["mlp_logvar"] = MLP(
                input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
                output_dim=latent_dim,
                layer_dims=[latent_dim * 4, latent_dim * 2],
                activation=activation,
                dropouts=None,
                normalization=nn.BatchNorm1d
                if normalization is not None
                else normalization,
            )
        else:
            self.nets["mlp"] = MLP(
                input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
                output_dim=latent_dim,
                layer_dims=[latent_dim * 4, latent_dim * 2],
                activation=activation,
                dropouts=None,
                normalization=nn.BatchNorm1d
                if normalization is not None
                else normalization,
            )

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.nets["conv"](x)
        x = self.nets["reshape"](x)
        if self.mean_var:
            mu = self.output_activation(self.nets["mlp_mu"](x))
            logvar = self.output_activation(self.nets["mlp_logvar"](x))
            z = self.reparametrize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.output_activation(self.nets["mlp"](x))
            return z


class ConvDecoder(VisionModule):
    def __init__(
        self,
        input_conv_size: List[int] = [4, 4],
        output_channel: int = 3,
        channels: List[int] = [256, 128, 64, 32, 16, 8],
        kernel_sizes: List[int] = [3, 4, 4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
    ) -> None:
        super(ConvDecoder, self).__init__()

        self.output_activation = (
            output_activation if output_activation is not None else lambda x: x
        )

        self.nets = nn.ModuleDict()
        self.nets["mlp"] = MLP(
            input_dim=latent_dim,
            output_dim=channels[0] * input_conv_size[0] * input_conv_size[1],
            layer_dims=[latent_dim * 2, latent_dim * 4],
            activation=activation,
            dropouts=None,
            normalization=nn.BatchNorm1d
            if normalization is not None
            else normalization,
        )
        self.nets["deconv"] = Conv(
            input_channel=channels[0],
            channels=channels[1:] + [output_channel],
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            layer=nn.ConvTranspose2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )
        self.nets["reshape"] = Reshape(
            (-1, channels[0], input_conv_size[0], input_conv_size[1])
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.nets["mlp"](z)
        x = self.nets["reshape"](x)
        x = self.nets["deconv"](x)
        return x


class AutoEncoder(VisionModule):
    """
    AutoEncoder for image compression using class Conv for Encoder and Decoder
    """

    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        encoder_kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        decoder_kernel_sizes: List[int] = [3, 4, 4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=nn.BatchNorm2d,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
    ) -> None:
        super(AutoEncoder, self).__init__()

        assert (
            len(channels)
            == len(encoder_kernel_sizes)
            == len(strides)
            == len(paddings)
            == len(decoder_kernel_sizes)
        )

        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ConvEncoder(
            input_size=input_size,
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
            latent_dim=latent_dim,
            mean_var=False,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=None,
        )

        self.nets["decoder"] = ConvDecoder(
            input_conv_size=output_conv_size,
            output_channel=input_channel,
            channels=list(reversed(channels)),
            kernel_sizes=decoder_kernel_sizes,
            strides=list(reversed(strides)),
            paddings=list(reversed(paddings)),
            latent_dim=latent_dim,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.nets["encoder"](x)
        x = self.nets["decoder"](z)
        return x, z

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        loss_dict = {}
        x_hat, z = self.forward(x)
        reconstruction_loss = nn.MSELoss()(x_hat, x)
        loss_dict["reconstruction_loss"] = reconstruction_loss
        return loss_dict


class VariationalAutoEncoder(VisionModule):
    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        encoder_kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        decoder_kernel_sizes: List[int] = [3, 4, 4, 4, 4, 4],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=nn.BatchNorm2d,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        assert (
            len(channels)
            == len(encoder_kernel_sizes)
            == len(strides)
            == len(paddings)
            == len(decoder_kernel_sizes)
        )

        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = ConvEncoder(
            input_size=input_size,
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
            latent_dim=latent_dim,
            mean_var=True,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=None,
        )

        self.nets["decoder"] = ConvDecoder(
            input_conv_size=output_conv_size,
            output_channel=input_channel,
            channels=list(reversed(channels)),
            kernel_sizes=decoder_kernel_sizes,
            strides=list(reversed(strides)),
            paddings=list(reversed(paddings)),
            latent_dim=latent_dim,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.nets["encoder"](x)
        x = self.nets["decoder"](z)
        return x, z, mu, logvar

    def kld_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # kld_weight = 1e-1 / torch.prod(torch.Tensor(mu.shape)) # TODO
        batch_size = mu.size(0)
        kld_weight = 1e-1 * mu.size(1) / (224 * 224 * 3 * batch_size)  # TODO
        kl_loss = (
            torch.mean(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),
                dim=0,
            )
            * kld_weight
        )
        return kl_loss

    def loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_dict = dict()
        x_hat, z, mu, logvar = self.forward(x)
        reconstruction_loss = nn.MSELoss()(x_hat, x)
        kld_loss = self.kld_loss(mu, logvar)
        loss_dict["reconstruction_loss"] = reconstruction_loss
        loss_dict["kld_loss"] = kld_loss
        return loss_dict


class SlotAttentionEncoder(VisionModule):
    """
    Slot Attention Encoder
    """

    def __init__(
        self,
        input_size: List[int] = [224, 224],
        input_channel: int = 3,
        channels: List[int] = [64, 64, 64, 64, 64, 64],
        kernel_sizes: List[int] = [5, 5, 5, 5, 5, 5],
        strides: List[int] = [1, 1, 1, 1, 1, 1],
        paddings: List[int] = [2, 2, 2, 2, 2, 2],
        num_iters: int = 3,
        eps: float = 1e-8,
        hidden_dim: int = 64,
        mlp_hidden_dim: int = 128,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = nn.ReLU,
        num_slots: int = 7,
    ) -> None:
        super(SlotAttentionEncoder, self).__init__()

        self.nets = nn.ModuleDict()

        self.nets["conv"] = Conv(
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

        self.nets["slot_attention"] = SlotAttention(
            num_slots=num_slots,
            dim=hidden_dim,
            num_iters=num_iters,
            eps=eps,
            hidden_dim=mlp_hidden_dim,
        )

        self.nets["pos_encoder"] = SoftPositionEmbed(
            hidden_dim=hidden_dim, resolution=input_size
        )

        self.nets["layer_norm"] = nn.LayerNorm(hidden_dim)
        self.nets["permute"] = Permute([0, 2, 3, 1])
        # self.nets["reshape"] = Reshape([-1, inputs_size[0] * inputs_size[1], hid_dim])

        self.nets["mlp_encoder"] = MLP(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            layer_dims=[hidden_dim],
            activation=activation,
            dropouts=dropouts,
            normalization=None,
            output_activation=None,
        )

    def forward(
        self, x: torch.Tensor, vectorized_slot_feature: bool = False
    ) -> torch.Tensor:
        x = self.nets["conv"](x)  # [B, 64, 128, 128]
        x = self.nets["permute"](x)  # [B, 128, 128, 64]
        x = self.nets["pos_encoder"](x)  # [B, 128, 128, 64]
        x = x.view(x.shape[0], -1, x.shape[-1])  # [B, 128*128, 64]
        x = self.nets["layer_norm"](x)
        x = self.nets["mlp_encoder"](x)
        if vectorized_slot_feature:
            x = self.nets["slot_attention"](x).view(x.shape[0], -1)
        else:
            x = self.nets["slot_attention"](x)
        return x


class SlotDecoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)

        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 4, stride=(2, 2), padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, 4, 4, stride=(2, 2), padding=1)

        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)  # [-1, 8, 8, slot_dim]
        x = x.permute(0, 3, 1, 2)  # [-1, slot_dim, 8, 8]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = x.permute(0, 2, 3, 1)  # [B, 128, 128, 4]
        return x


class SlotAttentionDecoder(VisionModule):
    """
    Slot Attention Encoder
    """

    def __init__(
        self,
        input_size: List[int] = [128, 128],
        input_channel: int = 3,
        channels: List[int] = [8, 16, 32, 64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3, 3, 3, 3],
        strides: List[int] = [2, 2, 2, 2, 2, 2],
        paddings: List[int] = [1, 1, 1, 1, 1, 1],
        latent_dim: int = 16,
        mean_var: bool = False,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
        num_slots: int = 7,
    ) -> None:
        super(SlotAttentionDecoder, self).__init__()
        hid_dim = 64

        self.nets = nn.ModuleDict()

        self.decoder_cnn = SlotDecoder(hid_dim, input_size)

        self.nets["layer_norm"] = nn.LayerNorm(hid_dim)

        self.nets["mlp_encoder"] = MLP(
            input_dim=hid_dim,
            output_dim=hid_dim,
            layer_dims=[hid_dim],
            activation=activation,
            dropouts=None,
            normalization=None,
            output_activation=None,
        )

    def forward(self, slots: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        batch_size = slots.shape[0]
        # slots [B, num_slots, slot_dim]
        # slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.view((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))

        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.view(
            batch_size, -1, x.shape[1], x.shape[2], x.shape[3]
        ).split([3, 1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots


class SlotAttentionAutoEncoder(VisionModule):
    """
    Slot Attention AutoEncoder
    """

    def __init__(
        self,
        input_size: List[int] = [128, 128],  # [224, 224],
        input_channel: int = 3,
        channels: List[int] = [64, 64, 64, 64],
        encoder_kernel_sizes: List[int] = [5, 5, 5, 5],
        decoder_kernel_sizes: List[int] = [3, 4, 4, 4],
        strides: List[int] = [1, 1, 1, 1],
        paddings: List[int] = [2, 2, 2, 2],
        latent_dim: int = 16,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=nn.BatchNorm2d,
        output_activation: Optional[nn.Module] = nn.Sigmoid,
    ) -> None:
        super(SlotAttentionAutoEncoder, self).__init__()

        self.nets = nn.ModuleDict()
        self.nets["encoder"] = SlotAttentionEncoder()
        self.nets["decoder"] = SlotAttentionDecoder()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        x = self.nets["encoder"](x)
        x = self.nets["decoder"](x)
        return x


class R3M(VisionModule):
    def __init__(
        self,
        input_channel=3,
        r3m_model_class="resnet18",
        freeze=True,
    ):
        super(R3M, self).__init__()

        try:
            from r3m import load_r3m
        except ImportError:
            print(
                "WARNING: could not load r3m library! Please follow https://github.com/facebookresearch/r3m to install R3M"
            )

        self.net = load_r3m(r3m_model_class)
        if freeze:
            self.net.eval()

        assert input_channel == 3  # R3M only support input image with channel size 3
        assert r3m_model_class in [
            "resnet18",
            "resnet34",
            "resnet50",
        ]  # make sure the selected r3m model do exist

        self._input_channel = input_channel
        self._r3m_model_class = r3m_model_class
        self._freeze = freeze

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    def preprocess(self, inputs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        preprocess inputs to fit the pretrained model
        """
        assert inputs.ndim == 4
        assert inputs.shape[1] == self._input_channel
        assert inputs.dtype in [np.uint8, np.float32, torch.uint8, torch.float32]
        inputs = TensorUtils.to_tensor(inputs)
        inputs = self.transform(inputs) * 255.0
        return inputs


class MVP(VisionModule):
    def __init__(
        self,
        input_channel=3,
        mvp_model_class="vitb-mae-egosoup",
        freeze=True,
    ):
        super(MVP, self).__init__()

        try:
            import mvp
        except ImportError:
            print(
                "WARNING: could not load mvp library! Please follow https://github.com/ir413/mvp to install MVP."
            )

        self.nets = mvp.load(mvp_model_class)
        if freeze:
            self.nets.freeze()

        assert input_channel == 3  # MVP only support input image with channel size 3
        assert mvp_model_class in [
            "vits-mae-hoi",
            "vits-mae-in",
            "vits-sup-in",
            "vitb-mae-egosoup",
            "vitl-256-mae-egosoup",
        ]  # make sure the selected r3m model do exist

        self._input_channel = input_channel
        self._freeze = freeze
        self._mvp_model_class = mvp_model_class

        if "256" in mvp_model_class:
            input_img_size = 256
        else:
            input_img_size = 224

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_img_size, input_img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess(self, inputs: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        preprocess inputs to fit the pretrained model
        """
        assert inputs.ndim == 4
        assert inputs.shape[1] == self._input_channel
        assert inputs.dtype in [np.uint8, np.float32, torch.uint8, torch.float32]
        inputs = TensorUtils.to_tensor(inputs)
        inputs = self.transform(inputs)
        return inputs


if __name__ == "__main__":
    # slot_attention_autoencoder = SlotAttentionAutoEncoder(dict())

    test_inputs = torch.randn(16, 3, 128, 128)

    slot_attention_encoder = SlotAttentionEncoder()
    slot_attention_decoder = SlotAttentionDecoder()

    test_outputs = slot_attention_encoder(test_inputs)
    test_outputs = slot_attention_decoder(test_outputs)

    for tensor in test_outputs:
        print(tensor.shape)

    from torchviz import make_dot

    INPUT_SIZE = 28 * 28

    model = NeuralNet()
    data = torch.randn(1, INPUT_SIZE)

    y = model(data)

    image = make_dot(y, params=dict(model.named_parameters()))
    image.format = "png"
    image.render("NeuralNet")
