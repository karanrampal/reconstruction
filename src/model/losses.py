"""Loss functions"""

from typing import Dict

import torch

from config_manager.manager import Params

MSE_LOSS = torch.nn.MSELoss()
L1_LOSS = torch.nn.L1Loss()


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
        loss += L1_LOSS(outputs[node], out_label[node]) * params.content

    loss /= params.mse + params.tvl + params.style + params.content
    return loss
