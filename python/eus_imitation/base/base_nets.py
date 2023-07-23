#!/usr/bin/env python3


from abc import abstractmethod
import numpy as np
from typing import Optional, Union, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import models as vision_models

import eus_imitation.util.tensor_utils as TensorUtils


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
        def time_distributed(inputs, net):
            batch_size, seq_len, _ = inputs.shape
            inputs = inputs.reshape(-1, inputs.shape[-1])
            outputs = net(inputs)
            outputs = outputs.reshape(batch_size, seq_len, -1)
            return outputs

        assert inputs.ndim == 3
        batch_size, seq_len, _ = inputs.shape
        if rnn_state is None:
            rnn_state = self.get_rnn_init_state(batch_size, inputs.device)
        outputs, rnn_state = self.nets(inputs, rnn_state)
        if self._per_step_net is not None:
            outputs = time_distributed(outputs, self._per_step_net)
            # outputs = TensorUtils.time_distributed(outputs, self._per_step_net)
        return outputs, rnn_state if return_rnn_state else outputs

    def forward_step(self, inputs: torch.Tensor, rnn_state: torch.Tensor):
        """
        return rnn outputs and rnn state for the next step
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
            if normalization is not None:
                layers.append(normalization(out_channels))
            layers.append(activation())
            if dropouts is not None:
                layers.append(nn.Dropout(dropouts[i]))

        if output_activation is not None:
            layers.append(output_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class VisionModule(nn.Module):
    """
    inputs like uint8 (B, C, H, W) or (B, C, H, W) of numpy ndarray or torch Tensor
    """

    @abstractmethod
    def preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        preprocess inputs to fit the pretrained model
        """
        raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.preprocess(inputs)
        return self.net(inputs)


class AutoEncoder(VisionModule):
    """
    AutoEncoder for image compression using class Conv for Encoder and Decoder
    """

    def __init__(
        self,
        input_size: List[int],
        input_channel: int,
        channels: List[int],
        encoder_kernel_sizes: List[int],
        decoder_kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        latent_dim: int,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super(AutoEncoder, self).__init__()

        assert (
            len(channels)
            == len(encoder_kernel_sizes)
            == len(strides)
            == len(paddings)
            == len(decoder_kernel_sizes)
        )

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

        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.encoder_conv = Conv(
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
            layer=nn.Conv2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

        self.decoder_conv = Conv(
            input_channel=channels[-1],
            channels=list(reversed(channels[:-1])) + [input_channel],
            kernel_sizes=decoder_kernel_sizes,
            strides=list(reversed(strides)),
            paddings=list(reversed(paddings)),
            layer=nn.ConvTranspose2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )
        self.encoder_mlp = MLP(
            input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
            output_dim=latent_dim,
            layer_dims=[latent_dim * 4, latent_dim * 2],
            activation=activation,
            dropouts=None,
            normalization=nn.BatchNorm1d
            if normalization is not None
            else normalization,
        )
        self.decoder_mlp = MLP(
            input_dim=latent_dim,
            output_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
            layer_dims=[latent_dim * 2, latent_dim * 4],
            activation=activation,
            dropouts=None,
            normalization=nn.BatchNorm1d
            if normalization is not None
            else normalization,
        )

        self.encoder_reshape = Reshape(
            (-1, channels[-1] * output_conv_size[0] * output_conv_size[1])
        )
        self.decoder_reshape = Reshape(
            (-1, channels[-1], output_conv_size[0], output_conv_size[1])
        )

        # self.encoder = nn.Sequential(
        #     self.encoder_conv, self.encoder_reshape, self.encoder_mlp
        # )
        # self.decoder = nn.Sequential(
        #     self.decoder_mlp, self.decoder_reshape, self.decoder_conv, nn.Sigmoid()
        # )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        x = self.encoder_conv(x)
        x = self.encoder_reshape(x)
        z = self.encoder_mlp(x)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_mlp(z)
        x = self.decoder_reshape(x)
        x = self.decoder_conv(x)
        x = self.unprocess(x)
        return x

    def preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.float() / 255.0

    def unprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        x = self.decode(z)
        return x, z


class VariationalAutoEncoder(VisionModule):
    def __init__(
        self,
        input_size: List[int],
        input_channel: int,
        channels: List[int],
        encoder_kernel_sizes: List[int],
        decoder_kernel_sizes: List[int],
        strides: List[int],
        paddings: List[int],
        latent_dim: int,
        activation: nn.Module = nn.ReLU,
        dropouts: Optional[List[float]] = None,
        normalization=None,
        output_activation: Optional[nn.Module] = None,
    ) -> None:
        super(VariationalAutoEncoder, self).__init__()
        assert (
            len(channels)
            == len(encoder_kernel_sizes)
            == len(strides)
            == len(paddings)
            == len(decoder_kernel_sizes)
        )

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

        output_conv_size = calculate_conv_output_size(
            input_size=input_size,  # TODO: input size
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
        )

        self.encoder_conv = Conv(
            input_channel=input_channel,
            channels=channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=strides,
            paddings=paddings,
            layer=nn.Conv2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )

        self.decoder_conv = Conv(
            input_channel=channels[-1],
            channels=list(reversed(channels[:-1])) + [input_channel],
            kernel_sizes=decoder_kernel_sizes,
            strides=list(reversed(strides)),
            paddings=list(reversed(paddings)),
            layer=nn.ConvTranspose2d,
            activation=activation,
            dropouts=dropouts,
            normalization=normalization,
            output_activation=output_activation,
        )
        self.encoder_mlp_mu = MLP(
            input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
            output_dim=latent_dim,
            layer_dims=[latent_dim * 4, latent_dim * 2],
            activation=activation,
            dropouts=None,
            normalization=nn.BatchNorm1d
            if normalization is not None
            else normalization,
        )
        self.encoder_mlp_logvar = MLP(
            input_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
            output_dim=latent_dim,
            layer_dims=[latent_dim * 4, latent_dim * 2],
            activation=activation,
            dropouts=None,
            normalization=nn.BatchNorm1d
            if normalization is not None
            else normalization,
        )
        self.decoder_mlp = MLP(
            input_dim=latent_dim,
            output_dim=channels[-1] * output_conv_size[0] * output_conv_size[1],
            layer_dims=[latent_dim * 2, latent_dim * 4],
            activation=activation,
            dropouts=None,
            normalization=nn.BatchNorm1d
            if normalization is not None
            else normalization,
        )

        self.encoder_reshape = Reshape(
            (-1, channels[-1] * output_conv_size[0] * output_conv_size[1])
        )
        self.decoder_reshape = Reshape(
            (-1, channels[-1], output_conv_size[0], output_conv_size[1])
        )

    def preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.float() / 255.0

    def unprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * 255.0

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.preprocess(x)
        x = self.encoder_conv(x)
        x = self.encoder_reshape(x)
        mu = self.encoder_mlp_mu(x)
        logvar = self.encoder_mlp_logvar(x)
        z = self.reparametrize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_mlp(z)
        x = self.decoder_reshape(x)
        x = self.decoder_conv(x)
        x = self.unprocess(x)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z, mu, logvar = self.encode(x)
        x = self.decode(z)
        return x, z, mu, logvar


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
    x, mu, logvar = test_vae(test_image_input)
    print(x.shape, mu.shape, logvar.shape)
