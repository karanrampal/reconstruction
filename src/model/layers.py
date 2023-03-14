"""Utility for neural network layers"""

from typing import List, Tuple

import numpy as np
import torch

torch.fx.wrap("min")
torch.fx.wrap("max")


def center_crop(img: torch.Tensor, new_height: int, new_width: int) -> torch.Tensor:
    """Crop center of tensor's last two dimension
    Args:
        img: 4D tensor to be center cropped
        new_eight: New height
        new_width: New width
    Returns:
        Center cropeed 4D tensor
    """
    _, _, height, width = img.shape
    left = max(0, (width - new_width) // 2)
    right = min((width + new_width) // 2, width)
    top = max(0, (height - new_height) // 2)
    bottom = min((height + new_height) // 2, height)

    return img[:, :, left:right, top:bottom]


def get_activation(activation_name: str) -> torch.nn.Module:
    """Get relevant activation module
    Args:
        activation_name: Activation layer name
    """
    activation_dict = torch.nn.ModuleDict(
        {
            "relu": torch.nn.ReLU(),
            "leaky": torch.nn.LeakyReLU(0.02),
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
            "identity": torch.nn.Identity(),
        }
    )
    return activation_dict[activation_name]


def get_normalization(
    normalization_name: str, out_filter: int, num_groups: int = 2
) -> torch.nn.Module:
    """Get relevant normalization module
    Args:
        normalization_name: Normalization layer, can be batch or identity
        out_filter: Number of output filters in the preceding conv layer
    """
    normalization_dict = torch.nn.ModuleDict(
        {
            "batch": torch.nn.BatchNorm2d(out_filter),
            "group": torch.nn.GroupNorm(num_groups, out_filter),
            "identity": torch.nn.Identity(),
        }
    )
    return normalization_dict[normalization_name]


class ConvBnActLayer(torch.nn.Module):
    """Custom convolution, batch norm, activation layer
    Args:
        normalization_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
        num_groups: Number of groups for group normalization
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
        self.batch_norm = get_normalization(
            normalization_name, kwargs["out_filter"], kwargs["num_groups"]
        )
        self.activation = get_activation(activation_name)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        out = self.conv(x_inp)
        out = self.batch_norm(out)
        return self.activation(out)


class PConvBnActLayer(torch.nn.Module):
    """Custom partial convolution, batch norm, activation layer
    Args:
        normalization_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
        num_groups: Number of groups for group normalization
    """

    def __init__(self, normalization_name: str, activation_name: str, **kwargs: int) -> None:
        super().__init__()
        self.conv = PartialConv2d(**kwargs)
        self.batch_norm = get_normalization(
            normalization_name, kwargs["out_filter"], kwargs["num_groups"]
        )
        self.activation = get_activation(activation_name)

    def forward(self, x_inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        out, mask = self.conv(x_inp, mask)
        out = self.batch_norm(out)
        return self.activation(out), mask


class P3ConvBnActLayer(torch.nn.Module):
    """Custom partial convolution, batch norm, activation layer
    Args:
        normalization_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
        num_groups: Number of groups for group normalization
    """

    def __init__(self, normalization_name: str, activation_name: str, **kwargs: int) -> None:
        super().__init__()
        self.conv1 = PConvBnActLayer(
            "identity",
            activation_name,
            in_filter=kwargs["in_filter"],
            out_filter=kwargs["out_filter"] // 2,
            kernel=1,
            stride=1,
            pad=0,
            num_groups=kwargs["out_filter"] // 2,
        )
        self.conv2 = PConvBnActLayer(
            normalization_name,
            activation_name,
            in_filter=kwargs["out_filter"] // 2,
            out_filter=kwargs["out_filter"] // 2,
            kernel=kwargs["kernel"],
            stride=kwargs["stride"],
            pad=kwargs["pad"],
            num_groups=kwargs["out_filter"] // 2,
        )
        self.conv3 = PConvBnActLayer(
            "identity",
            activation_name,
            in_filter=kwargs["out_filter"] // 2,
            out_filter=kwargs["out_filter"],
            kernel=1,
            stride=1,
            pad=0,
            num_groups=kwargs["out_filter"],
        )

    def forward(self, x_inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        out, mask = self.conv1(x_inp, mask)
        out, mask = self.conv2(out, mask)
        return self.conv3(out, mask)


class TransConvBnActLayer(torch.nn.Module):
    """Custom transposed convolution, batch norm, activation layer
    Args:
        normalization_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Transposed convolution stride length
        pad: Transposed convolution input padding
        out_pad: Transposed convolution output padding
        num_groups: Number of groups for group normalization
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
        self.batch_norm = get_normalization(
            normalization_name, kwargs["out_filter"], kwargs["num_groups"]
        )
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
        activation_name: Activation layer name
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
        num_groups: Number of groups for group normalization
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
        self.batch_norm = get_normalization(norm_name, kwargs["out_filter"], kwargs["num_groups"])
        self.activation = get_activation(activation_name)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        out = self.upsample(x_inp)
        out = self.conv(out)
        out = self.batch_norm(out)
        return self.activation(out)


class UpPConvBnActLayer(torch.nn.Module):
    """Custom upsampling, partial convolution, batch norm, activation layer
    Args:
        scale: Upsampling scale factor
        norm_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
        num_groups: Number of groups for group normalization
    """

    def __init__(self, scale: float, norm_name: str, activation_name: str, **kwargs: int) -> None:
        super().__init__()
        self.upsample = torch.nn.Upsample(scale_factor=scale)
        self.conv = PartialConv2d(**kwargs)
        self.batch_norm = get_normalization(norm_name, kwargs["out_filter"], kwargs["num_groups"])
        self.activation = get_activation(activation_name)

    def forward(self, x_inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        out = self.upsample(x_inp)
        _, _, height, width = mask.shape
        out, mask = self.conv(center_crop(out, height, width), mask)
        out = self.batch_norm(out)
        return self.activation(out), mask


class HPF(torch.nn.Module):
    """Concatenate High pass filtered input
    Args:
        kernel_sizes: Size of Gaussian kernels
    """

    def __init__(self, kernel_sizes: List[int]) -> None:
        super().__init__()
        self.weights = self._gaussian2d(kernel_sizes)

    def _gaussian2d(self, kernel_sizes: List[int]) -> torch.Tensor:
        """Calculate Gaussian kernels
        Args:
            kernel_sizes: Kernel sizes
        Returns:
            gaussian kernels of shape (num_kernel_sizes x 1 x max_kernel_size x max_kernel_size)
        """
        half_max = max(kernel_sizes) // 2
        ks_t = torch.tensor(kernel_sizes).unsqueeze(1).unsqueeze(1)
        range_ = torch.arange(-half_max, half_max + 1)
        i, j = torch.meshgrid(range_, range_, indexing="ij")
        sigmas = 0.3 * ((ks_t - 1) * 0.5 - 1) + 0.8
        weights = torch.exp(-(i.unsqueeze(0) ** 2 + j.unsqueeze(0) ** 2) / (2 * sigmas**2))
        weights /= np.pi * 2 * sigmas**2
        return weights.unsqueeze(1)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        out = torch.nn.functional.conv2d(x_inp, self.weights, padding="same")
        return torch.cat([x_inp, x_inp - out], dim=1)


class PositionalEncoding(torch.nn.Module):
    """Positional encoder
    Args:
        num_encodings: Number of encodings to use
        keep_input: Keep input along with encodings
        log_sampling: Use log sampling
    """

    def __init__(
        self, num_encodings: int = 6, keep_input: bool = True, log_sampling: bool = True
    ) -> None:
        super().__init__()
        self.keep_input = keep_input
        if log_sampling:
            self.frequency_bands = 2.0 ** torch.linspace(0.0, num_encodings - 1, num_encodings)
        else:
            self.frequency_bands = torch.linspace(1.0, 2.0 ** (num_encodings - 1), num_encodings)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        encoding = [x_inp] if self.keep_input else []

        for freq in self.frequency_bands:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(x_inp * freq))

        return torch.cat(encoding, dim=1)


class PartialConv2d(torch.nn.Module):
    """Partial convolution implementation
    Args:
        in_channels: Number of input filters
        out_channels: Number of output filters
        kernel_size: Kernel size
        stride: Convolution stride length
        paddding: Convolution input padding
        output_padding: Pad output
        num_groups: Number of groups for group normalization
    """

    def __init__(self, **kwargs: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            kwargs["in_filter"],
            kwargs["out_filter"],
            kwargs["kernel"],
            kwargs["stride"],
            kwargs["pad"],
        )
        self.wts = torch.ones(1, 1, kwargs["kernel"], kwargs["kernel"])

    def forward(
        self, input_: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        output = self.conv(input_ * mask)
        bias = self.conv.bias.view(1, -1, 1, 1)  # type: ignore[union-attr]

        with torch.no_grad():
            sums_ = torch.nn.functional.conv2d(
                mask, self.wts.to(mask.device), stride=self.conv.stride, padding=self.conv.padding
            )
            mask_zero_loc = sums_ == 0.0
            updates = sums_.masked_fill(mask_zero_loc, 1.0)
            updated_mask = torch.ones_like(sums_)
            updated_mask.masked_fill_(mask_zero_loc, 0.0)

        updated_output = (output - bias) / updates + bias
        updated_output.masked_fill_(mask_zero_loc, 0.0)

        return updated_output, updated_mask


class ResLayer(torch.nn.Module):
    """Custom residual partial convolution, batch norm, activation layer
    Args:
        normalization_name: Normalization layer, can be batch or identity
        activation_name: Activation layer name
    Kwargs:
        in_filter: Number of input filters
        out_filter: Number of output filters
        kernel: Kernel size
        stride: Convolution stride length
        pad: Convolution input padding
        num_groups: Number of groups for group normalization
    """

    def __init__(self, normalization_name: str, activation_name: str, **kwargs: int) -> None:
        super().__init__()
        self.conv1 = PConvBnActLayer(
            normalization_name,
            activation_name,
            in_filter=kwargs["in_filter"],
            out_filter=kwargs["in_filter"],
            kernel=3,
            stride=1,
            pad=1,
            num_groups=kwargs["in_filter"],
        )
        self.conv2 = PConvBnActLayer(
            normalization_name,
            activation_name,
            in_filter=kwargs["in_filter"],
            out_filter=kwargs["in_filter"],
            kernel=3,
            stride=1,
            pad=1,
            num_groups=kwargs["in_filter"],
        )
        self.conv3 = PConvBnActLayer(
            normalization_name,
            activation_name,
            in_filter=kwargs["in_filter"],
            out_filter=kwargs["out_filter"],
            kernel=kwargs["kernel"],
            stride=kwargs["stride"],
            pad=kwargs["pad"],
            num_groups=kwargs["out_filter"],
        )

    def forward(self, x_inp: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        out, mask = self.conv1(x_inp, mask)
        out, mask = self.conv2(out, mask)
        return self.conv3(out + x_inp, mask)
