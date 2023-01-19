"""Neural Network model"""

from typing import List

import torch

import model.layers as layer


class BPModel(torch.nn.Module):
    """Prediction model of back depth map
    Args:
        filters: List of filters in each encoder layer
        kernels: List of kernels in each encoder layer
        embeddings: Embedding dimension
    """

    def __init__(self, filters: List[int], kernels: List[int]) -> None:
        super().__init__()
        filters = [1] + filters
        self.encoder = self._encoder(filters, kernels)
        self.decoder = self._decoder(filters, kernels)

    def _encoder(self, filters: List[int], kernels: List[int]) -> torch.nn.Sequential:
        """Encoder helper function"""
        layers = [
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
        return torch.nn.Sequential(*layers)

    def _decoder(self, filters: List[int], kernels: List[int]) -> torch.nn.Sequential:
        """Decoder helper function"""
        layers = [
            layer.TransConvBnActLayer(
                "batch",
                "relu",
                in_filter=filters[i + 1],
                out_filter=filters[i],
                kernel=kernels[i],
                stride=2,
                pad=kernels[i] // 2,
                out_pad=1,
            )
            for i in range(len(kernels) - 1, 1, -1)
        ]
        layers += [
            layer.TransConvBnActLayer(
                "identity",
                "relu",
                in_filter=filters[2],
                out_filter=filters[1],
                kernel=kernels[1],
                stride=2,
                pad=kernels[1] // 2,
                out_pad=1,
            ),
            layer.TransConvBnActLayer(
                "identity",
                "relu",
                in_filter=filters[1],
                out_filter=filters[0],
                kernel=kernels[0],
                stride=2,
                pad=kernels[0] // 2,
                out_pad=1,
            ),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        x_inp = self.encoder(x_inp)
        out = self.decoder(x_inp)

        return out
