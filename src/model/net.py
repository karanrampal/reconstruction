"""Neural Network model"""

from typing import Dict, List

import torch
from torchvision.models.feature_extraction import create_feature_extractor

from config_manager.manager import Params

MSE_LOSS = torch.nn.MSELoss()


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
        self.fc1 = self._linear(4 * 4 * filters[-1], embeddings)
        self.fc2 = self._linear(embeddings, 4 * 4 * filters[-1])

        # Decoder
        self.decoder = self._decoder(filters, kernels)
        self.dconv1 = self._up_conv_relu(filters[2], filters[1], kernels[1], kernels[1] // 2)
        self.dconv2 = self._up_conv_sigmoid(filters[1], filters[0], kernels[0], kernels[0] // 2)

    def _down_conv_bn_relu(
        self, in_filter: int, out_filter: int, kernel: int, pad: int
    ) -> torch.nn.Sequential:
        """Helper function for creating down convolution layer"""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_filter, out_filter, kernel_size=kernel, stride=2, padding=pad),
            torch.nn.BatchNorm2d(out_filter),
            torch.nn.ReLU(),
        )

    def _encoder(self, filters: List[int], kernels: List[int]) -> torch.nn.Sequential:
        """Encoder helper function"""
        layers = [
            self._down_conv_bn_relu(
                filters[i],
                filters[i + 1],
                kernels[i],
                kernels[i] // 2,
            )
            for i in range(len(kernels))
        ]
        return torch.nn.Sequential(*layers)

    def _up_conv_bn_relu(
        self, in_filter: int, out_filter: int, kernel: int, pad: int
    ) -> torch.nn.Sequential:
        """Helper function for creating up convolution layer"""
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_filter,
                out_filter,
                kernel_size=kernel,
                stride=2,
                padding=pad,
                output_padding=1,
            ),
            torch.nn.BatchNorm2d(out_filter),
            torch.nn.ReLU(),
        )

    def _up_conv_relu(
        self, in_filter: int, out_filter: int, kernel: int, pad: int
    ) -> torch.nn.Sequential:
        """Helper function for creating up convolution layer"""
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_filter,
                out_filter,
                kernel_size=kernel,
                stride=2,
                padding=pad,
                output_padding=1,
            ),
            torch.nn.ReLU(),
        )

    def _up_conv_sigmoid(
        self, in_filter: int, out_filter: int, kernel: int, pad: int
    ) -> torch.nn.Sequential:
        """Helper function for creating up convolution layer"""
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_filter,
                out_filter,
                kernel_size=kernel,
                stride=2,
                padding=pad,
                output_padding=1,
            ),
            torch.nn.Sigmoid(),
        )

    def _decoder(self, filters: List[int], kernels: List[int]) -> torch.nn.Sequential:
        """Decoder helper function"""
        layers = [
            self._up_conv_bn_relu(
                filters[i + 1],
                filters[i],
                kernels[i],
                kernels[i] // 2,
            )
            for i in range(len(kernels) - 1, 1, -1)
        ]
        return torch.nn.Sequential(*layers)

    def _linear(self, in_: int, out_: int) -> torch.nn.Sequential:
        """Linear layers helper function"""
        return torch.nn.Sequential(
            torch.nn.Linear(in_, out_),
            torch.nn.BatchNorm1d(out_),
            torch.nn.ReLU(),
        )

    def forward(self, x_inp: torch.Tensor) -> torch.Tensor:
        """Forward propagation"""
        x_inp = self.encoder(x_inp)
        x_inp = x_inp.reshape(-1, 4 * 4 * 256)

        x_inp = self.fc1(x_inp)
        x_inp = self.fc2(x_inp)
        x_inp = x_inp.reshape(-1, 256, 4, 4)

        x_inp = self.decoder(x_inp)
        x_inp = self.dconv1(x_inp)
        out = self.dconv2(x_inp)

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


def total_variation_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Total variation loss
    Args:
        outputs: Model predictions
    Returns:
        Total variation loss
    """
    x_var = outputs[..., :, 1:] - outputs[..., :, :-1]
    y_var = outputs[..., 1:, :] - outputs[..., :-1, :]
    return x_var.abs().mean() + y_var.abs().mean()


def style_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate style loss
    Args:
        outputs: Model predictions
        labels: Ground truth labels
    Returns:
        Style loss
    """
    out_loss = torch.einsum("bcij,bdij->bcd", outputs, outputs)
    label_loss = torch.einsum("bcij,bdij->bcd", labels, labels)
    return (out_loss - label_loss).abs().mean()


def loss_fn(
    outputs: Dict[str, torch.Tensor],
    out_label: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    params: Params,
) -> torch.Tensor:
    """Loss function
    Args:
        outputs: Model predictions
        out_label: Model prediction on the ground truth
        labels: Ground truth labels
        params: Hyperparameters
    Returns:
        Total loss
    """
    predictions = outputs[params.output_layer]

    loss = MSE_LOSS(predictions, labels) * params.mse
    loss += total_variation_loss(predictions) * params.tvl
    for node in params.style_nodes:
        loss += style_loss(outputs[node], out_label[node]) * params.style
        loss += MSE_LOSS(outputs[node], out_label[node]) * params.content

    loss /= params.mse + params.tvl + params.style + params.content
    return loss
