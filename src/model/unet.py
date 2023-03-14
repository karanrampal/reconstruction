"""Unet implementation"""

from typing import List, Tuple

import torch

import model.layers as layer


class UnetUp(torch.nn.Module):
    """UNet type upscaling and convolutions
    Args:
        in_filter: Number of input channels
        out_filter: Number of output channels
        kernel: Kernel size
    """

    def __init__(self, in_filter: int, out_filter: int, kernel: int) -> None:
        super().__init__()
        self.up_conv = layer.UpPConvBnActLayer(
            2.0,
            "identity",
            "identity",
            in_filter=in_filter,
            out_filter=out_filter,
            kernel=kernel,
            stride=1,
            pad=kernel // 2,
            num_groups=2,
        )
        self.conv = layer.PConvBnActLayer(  # layer.ResLayer(
            "group",
            "leaky",
            in_filter=out_filter * 2,
            out_filter=out_filter,
            kernel=3,
            stride=1,
            pad=1,
            num_groups=out_filter,
        )

    def forward(
        self, x_small: torch.Tensor, x_big: torch.Tensor, m_big: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation"""
        x_up, mask = self.up_conv(x_small, m_big)
        xcat = torch.cat([x_up, x_big], dim=1)
        out, mask = self.conv(xcat, mask)
        return out, mask


class UNet(torch.nn.Module):
    """UNet archtecture
    Args:
        input_channel: Input channels
        output_channel: Output channels
        filters: List of output channels
        kernels: List of kernel size
    """

    def __init__(
        self, input_channel: int, output_channel: int, filters: List[int], kernels: List[int]
    ) -> None:
        super().__init__()
        enc = [
            layer.PConvBnActLayer(
                "identity",
                "leaky",
                in_filter=input_channel,
                out_filter=filters[0],
                kernel=kernels[0],
                stride=1,
                pad=kernels[0] // 2,
                num_groups=2,
            )
        ]
        enc += [
            layer.PConvBnActLayer(
                # layer.ResLayer(
                "group",
                "leaky",
                in_filter=filters[i - 1],
                out_filter=filters[i],
                kernel=kernels[i],
                stride=2,
                pad=kernels[i] // 2,
                num_groups=filters[i],
            )
            for i in range(1, len(kernels))
        ]
        self.encoder = torch.nn.ModuleList(enc)

        self.decoder = torch.nn.ModuleList(
            [UnetUp(filters[i], filters[i - 1], 3) for i in reversed(range(1, len(filters)))]
        )

        self.out_conv = layer.ConvBnActLayer(
            "identity",
            "sigmoid",
            in_filter=filters[0],
            out_filter=output_channel,
            kernel=3,
            stride=1,
            pad=1,
            num_groups=1,
        )

    def forward(self, x_t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        enc_trace, mask_trace = [], []
        for enc_layer in self.encoder:
            x_t, mask = enc_layer(x_t, mask)
            enc_trace.append(x_t)
            mask_trace.append(mask)

        for i, dec_layer in enumerate(self.decoder):
            x_t, mask = dec_layer(x_t, enc_trace[-i - 2], mask_trace[-i - 2])

        return self.out_conv(x_t)
