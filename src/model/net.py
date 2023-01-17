"""Neural Network model"""

from typing import Dict, List

import torch
from torchvision.models.feature_extraction import create_feature_extractor

import model.layers as layer


class BPModel(torch.nn.Module):
    """Prediction model of back depth map
    Args:
        filters: List of filters in each encoder layer
        kernels: List of kernels in each encoder layer
        embeddings: Embedding dimension
    """

    def __init__(self, filters: List[int], kernels: List[int], embeddings: int) -> None:
        super().__init__()
        filters = [1] + filters

        # Encoder
        self.encoder = self._encoder(filters, kernels)

        # Linear
        self.fc1 = layer.linear_bn_relu(4 * 4 * filters[-1], embeddings)
        self.fc2 = layer.linear_bn_relu(embeddings, 4 * 4 * filters[-1])

        # Decoder
        self.decoder = self._decoder(filters, kernels)

    def _encoder(self, filters: List[int], kernels: List[int]) -> torch.nn.Sequential:
        """Encoder helper function"""
        layers = [
            layer.conv_bn_relu(filters[i], filters[i + 1], kernels[i], 2, kernels[i] // 2)
            for i in range(len(kernels))
        ]
        return torch.nn.Sequential(*layers)

    def _decoder(self, filters: List[int], kernels: List[int]) -> torch.nn.Sequential:
        """Decoder helper function"""
        layers = [
            layer.trans_conv_bn_relu(filters[i + 1], filters[i], kernels[i], 2, kernels[i] // 2)
            for i in range(len(kernels) - 1, 1, -1)
        ]
        layers += [
            layer.trans_conv_relu(filters[2], filters[1], kernels[1], 2, kernels[1] // 2),
            layer.trans_conv_sigmoid(filters[1], filters[0], kernels[0], 2, kernels[0] // 2),
        ]
        return torch.nn.Sequential(*layers)

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        x_inp = self.encoder(x_inp)
        x_inp = x_inp.reshape(-1, 4 * 4 * 256)

        x_inp = self.fc1(x_inp)
        x_inp = self.fc2(x_inp)

        x_inp = x_inp.reshape(-1, 256, 4, 4)
        out = self.decoder(x_inp)

        return out


class ReconstructionModel(torch.nn.Module):
    """Prediction model of back depth map
    Args:
        filters: List of filters in each encoder layer
        kernels: List of kernels in each encoder layer
        embeddings: Embedding dimension
        nodes: List of nodes in the output
    """

    def __init__(
        self, filters: List[int], kernels: List[int], embeddings: int, nodes: List[str]
    ) -> None:
        super().__init__()
        model = BPModel(filters, kernels, embeddings)
        self.model = create_feature_extractor(model, return_nodes=nodes)

    def forward(self, x_inp: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward propagation"""
        return self.model(x_inp)
