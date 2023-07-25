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

import eus_imitation.util.tensor_utils as TensorUtils



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
            output_size[0] - 1
        ) * strides[i] - 2 * paddings[i] + kernel_sizes[i] + output_paddings[i]
        output_size[1] = (
            output_size[1] - 1
        ) * strides[i] - 2 * paddings[i] + kernel_sizes[i] + output_paddings[i]
    return output_size


class Reshape(nn.Module):
    """
    Module that reshapes a tensor.
    """

    def __init__(self, shape: Union[int, Tuple[int, ...]]):
        super(Reshape, self).__init__()
        self._shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(self._shape)


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

        assert inputs.ndim == 3 # (batch_size, seq_len, input_dim)
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
            if normalization is not None and i != len(channels) - 1: # not the last layer
                layers.append(normalization(out_channels))
            if i != len(channels) - 1: # not the last layer
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
        self.output_activation = output_activation if output_activation is not None else lambda x: x

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
        self.nets["reshape"] = Reshape((-1, channels[-1] * output_conv_size[0] * output_conv_size[1]))

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

        self.output_activation = output_activation if output_activation is not None else lambda x: x

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
        normalization= nn.BatchNorm2d,
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
        kld_weight = 1e-1 * mu.size(1) / (224 * 224 * 3 * batch_size) # TODO
        kl_loss = (
            torch.mean(
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1),
                dim=0,
            )
            * kld_weight
        )
        return kl_loss

    def loss(self, x : torch.Tensor) -> Dict[str, torch.Tensor]:
        loss_dict = dict()
        x_hat, z, mu, logvar = self.forward(x)
        reconstruction_loss = nn.MSELoss()(x_hat, x)
        kld_loss = self.kld_loss(mu, logvar)
        loss_dict["reconstruction_loss"] = reconstruction_loss
        loss_dict["kld_loss"] = kld_loss
        return loss_dict


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
    test_image_input = torch.randn(5, 3, 224, 224)
    test_autoencoder = AutoEncoder(
        input_size=test_image_input.shape[2:],
        input_channel=3,
        channels=[8, 16, 32, 64, 128, 256],
        encoder_kernel_sizes=[3, 3, 3, 3, 3, 3],
        decoder_kernel_sizes=[3, 4, 4, 4, 4, 4],
        strides=[2, 2, 2, 2, 2, 2],
        paddings=[1, 1, 1, 1, 1, 1],
        latent_dim=16,
        activation=nn.ReLU,
        dropouts=None,
        normalization=nn.BatchNorm2d,
        output_activation=None,
    )

    x, z = test_autoencoder(test_image_input)
    print(x.shape, z.shape)
    test_vae = VariationalAutoEncoder(
        input_size=test_image_input.shape[2:],
        input_channel=3,
        channels=[8, 16, 32, 64, 128, 256],
        encoder_kernel_sizes=[3, 3, 3, 3, 3, 3],
        decoder_kernel_sizes=[3, 4, 4, 4, 4, 4],
        strides=[2, 2, 2, 2, 2, 2],
        paddings=[1, 1, 1, 1, 1, 1],
        latent_dim=16,
        activation=nn.ReLU,
        dropouts=None,
        normalization=nn.BatchNorm2d,
        output_activation=None,
    )

    test_image_input = torch.randn(5, 3, 224, 224)
    x, z, mu, logvar = test_vae(test_image_input)
    print(x.shape, z.shape, mu.shape, logvar.shape)
