"""Utility for neural network layers"""

from typing import Dict, List

import torch
from torchvision.models.feature_extraction import create_feature_extractor


def get_activation(activation_name: str) -> torch.nn.Module:
    """Get relevant activation module
    Args:
        activation_name: Activation layer name, can be relu, leaky, sigmoid or identity
    """
    activation_dict = torch.nn.ModuleDict(
        {
            "relu": torch.nn.ReLU(),
            "leaky": torch.nn.LeakyReLU(0.02),
            "sigmoid": torch.nn.Sigmoid(),
            "identity": torch.nn.Identity(),
        }
    )
    return activation_dict[activation_name]


def get_normalization(normalization_name: str, out_filter: int) -> torch.nn.Module:
    """Get relevant normalization module
    Args:
        normalization_name: Normalization layer, can be batch or identity
        out_filter: Number of output filters in the preceding conv layer
    """
    normalization_dict = torch.nn.ModuleDict(
        {
            "batch": torch.nn.BatchNorm2d(out_filter),
            "identity": torch.nn.Identity(),
        }
    )
    return normalization_dict[normalization_name]


class ConvBnActLayer(torch.nn.Module):
    """Custom convolution, batch norm, activation layer
    Args:
        normalization_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name, can be relu, leaky, sigmoid or identity
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
    """

    def __init__(self, normalization_name: str, activation_name: str, **kwargs: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            kwargs["in_filter"],
            kwargs["out_filter"],
            kernel_size=kwargs["kernel"],
            stride=kwargs["stride"],
            padding=kwargs["pad"],
        )
        self.batch_norm = get_normalization(normalization_name, kwargs["out_filter"])
        self.activation = get_activation(activation_name)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        out = self.conv(x_inp)
        out = self.batch_norm(out)
        return self.activation(out)


class TransConvBnActLayer(torch.nn.Module):
    """Custom transposed convolution, batch norm, activation layer
    Args:
        normalization_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name, can be relu, leaky, sigmoid or identity
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Transposed convolution stride length
        pad: Transposed convolution input padding
        out_pad: Transposed convolution output padding
    """

    def __init__(self, normalization_name: str, activation_name: str, **kwargs: int) -> None:
        super().__init__()
        self.trans_conv = torch.nn.ConvTranspose2d(
            kwargs["in_filter"],
            kwargs["out_filter"],
            kernel_size=kwargs["kernel"],
            stride=kwargs["stride"],
            padding=kwargs["pad"],
            output_padding=kwargs["out_pad"],
        )
        self.batch_norm = get_normalization(normalization_name, kwargs["out_filter"])
        self.activation = get_activation(activation_name)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        out = self.trans_conv(x_inp)
        out = self.batch_norm(out)
        return self.activation(out)


class UpConvBnActLayer(torch.nn.Module):
    """Custom upsampling, convolution, batch norm, activation layer
    Args:
        scale: Upsampling scale factor
        norm_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name, can be relu, leaky, sigmoid or identity
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
    """

    def __init__(self, scale: float, norm_name: str, activation_name: str, **kwargs: int) -> None:
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=scale)
        self.conv = torch.nn.Conv2d(
            kwargs["in_filter"],
            kwargs["out_filter"],
            kernel_size=kwargs["kernel"],
            stride=kwargs["stride"],
            padding=kwargs["pad"],
        )
        self.batch_norm = get_normalization(norm_name, kwargs["out_filter"])
        self.activation = get_activation(activation_name)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        out = self.upsample(x_inp)
        out = self.conv(out)
        out = self.batch_norm(out)
        return self.activation(out)


class ReconstructionModel(torch.nn.Module):
    """Prediction model of back depth map
    Args:
        base_model: Base model to extend
        nodes: List of nodes in the output
    """

    def __init__(self, base_model: torch.nn.Module, nodes: List[str]) -> None:
        super().__init__()
        self.base_model = base_model
        self.model = create_feature_extractor(base_model, return_nodes=nodes)

    def forward(self, x_inp: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward propagation"""
        return self.model(x_inp)
