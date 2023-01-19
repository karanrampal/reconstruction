"""Unet implementation"""

from typing import List

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
        self.up_conv = layer.UpConvBnActLayer(
            2.0,
            "batch",
            "relu",
            in_filter=in_filter,
            out_filter=in_filter // 2,
            kernel=kernel,
            stride=1,
            pad=kernel // 2,
        )
        self.conv = layer.ConvBnActLayer(
            "batch", "relu", in_filter=in_filter, out_filter=out_filter, kernel=3, stride=1, pad=1
        )

    def forward(self, x_small: torch.Tensor, x_big: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        x_up = self.up_conv(x_small)
        xcat = torch.cat([x_up, x_big], dim=1)
        return self.conv(xcat)


class UNet(torch.nn.Module):
    """UNet archtecture"""

    def __init__(self, filters: List[int], kernels: List[int]) -> None:
        super().__init__()
        filters = [filters[0] // 2] + filters

        enc = [
            layer.ConvBnActLayer(
                "batch",
                "relu",
                in_filter=1,
                out_filter=filters[0],
                kernel=kernels[0],
                stride=1,
                pad=kernels[0] // 2,
            )
        ]
        enc += [
            layer.ConvBnActLayer(
                "batch",
                "relu",
                in_filter=filters[i],
                out_filter=filters[i + 1],
                kernel=kernels[i],
                stride=2,
                pad=kernels[i] // 2,
            )
            for i in range(len(kernels))
        ]
        self.encoder = torch.nn.ModuleList(enc)

        self.decoder = torch.nn.ModuleList(
            [
                UnetUp(filters[i + 1], filters[i], kernels[i])
                for i in range(len(kernels) - 1, -1, -1)
            ]
        )

        self.out_conv = layer.ConvBnActLayer(
            "identity",
            "relu",
            in_filter=filters[0],
            out_filter=1,
            kernel=1,
            stride=1,
            pad=0,
        )

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        enc_trace = []
        for enc_layer in self.encoder:
            x_t = enc_layer(x_t)
            enc_trace.append(x_t)

        out = enc_trace[-1]
        for i, dec_layer in enumerate(self.decoder):
            out = dec_layer(out, enc_trace[-i - 2])

        return self.out_conv(out)
