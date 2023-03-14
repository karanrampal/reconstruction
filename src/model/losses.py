"""Loss functions"""

from typing import Dict, Tuple

import torch

from config_manager.manager import Params

L1_LOSS = torch.nn.L1Loss()
CE_LOSS = torch.nn.CrossEntropyLoss()


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
    return (out_loss - label_loss).abs().mean() / (outputs.shape[-1] * outputs.shape[-2])


def log_l1_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate the log L1 loss
    Args:
        outputs: Model predictions
        labels: Ground truth labels
    Returns:
        Log L1 loss
    """
    return torch.log(1.0 + (outputs - labels).abs()).mean()


def grad_loss(
    outputs: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, device: str
) -> torch.Tensor:
    """Gradient loss
    Args:
        outputs: Model predictions
        labels: Ground truth labels
        device: CPU or GPU
    Returns:
        Gradient loss
    """
    sobel = torch.tensor(
        [
            [[[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]],
            [[[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]],
        ],
        device=torch.device(device),
    )

    out = torch.nn.functional.conv2d(outputs, sobel, padding="same")
    g_t = torch.nn.functional.conv2d(labels, sobel, padding="same")

    return torch.log(
        (out[mask.expand(-1, 2, -1, -1)] - g_t[mask.expand(-1, 2, -1, -1)]).abs() + 1.0
    ).mean()


def discriminator_loss(
    disc_emb: torch.Tensor, net_emb: torch.Tensor, params: Params
) -> Tuple[torch.Tensor, float, float]:
    """Discriminator loss
    Args:
        disc_emb: Discrimindator embeddings
        net_emb: Net embeddings
        params: Hyper-parameters
    Returns:
        Discriminator loss
    """
    emb = net_emb.flatten(start_dim=1)
    emb = emb / emb.norm(dim=1, keepdim=True)
    inp = disc_emb.flatten(start_dim=1)
    inp = inp / inp.norm(dim=1, keepdim=True)

    out = (inp @ emb.t()) * torch.exp(torch.tensor(params.temperature))

    batch_size = out.shape[0]
    labels = torch.arange(batch_size, device=torch.device(params.device))
    tmp = CE_LOSS(out, labels) + CE_LOSS(out.t(), labels)

    acc = avg_acc_gpu(out.detach(), labels.detach())
    f_1 = avg_f1_score_gpu(out.detach(), labels.detach(), batch_size)

    return tmp / 2.0, acc, f_1


def loss_fn(
    outputs: Dict[str, torch.Tensor],
    vgg_outputs: Dict[str, torch.Tensor],
    label_out: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    mask: torch.Tensor,
    params: Params,
) -> Tuple[torch.Tensor, float, float]:
    """Loss function
    Args:
        outputs: Model predictions
        vgg_outputs: Ground truth model outputs
        labels: Ground truth labels
        params: Hyperparameters
    Returns:
        Total loss
    """
    predictions = outputs[params.output_layer]
    num = len(params.style_layers)
    num_cascades = len(params.cascade_layers) + 1

    loss = L1_LOSS(predictions[mask], labels[mask]) * params.person
    for cascade_names in params.cascade_layers + [params.output_layer]:
        out = outputs[cascade_names]
        loss += L1_LOSS(out[~mask], labels[~mask]) * params.bg
        loss += log_l1_loss(out[mask], labels[mask]) * params.exp
        loss += grad_loss(out, labels, mask, params.device) * params.grad

    loss += total_variation_loss(predictions) * params.tvl

    for node, node_v in zip(params.style_layers, params.vgg_style):
        loss += style_loss(outputs[node], vgg_outputs[node_v]) * params.style
        loss += L1_LOSS(outputs[node], vgg_outputs[node_v]) * params.perceptual

    tmp, acc, f_1 = discriminator_loss(
        label_out[params.embedding_layer], outputs[params.embedding_layer], params
    )
    loss += tmp * params.disc

    loss /= (
        (params.person + params.bg + params.grad + params.exp) * num_cascades
        + (params.style + params.perceptual) * num
        + params.tvl
        + params.disc
    )
    return loss, acc, f_1


def avg_acc_gpu(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
    Returns:
        average accuracy in [0,1]
    """
    preds = outputs.argmax(dim=1).to(torch.int64)
    avg_acc = (preds == labels).to(torch.float32).mean()
    return avg_acc.item()


def avg_f1_score_gpu(
    outputs: torch.Tensor, labels: torch.Tensor, num_classes: int, eps: float = 1e-7
) -> float:
    """Compute the F1 score, given the outputs and labels for all images.
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        num_classes: Number of classes
        eps: Epsilon
    Returns:
        average f1 score
    """
    preds = (outputs).argmax(dim=1).to(torch.int64)
    pred_ohe = torch.nn.functional.one_hot(preds, num_classes)
    label_ohe = torch.nn.functional.one_hot(labels, num_classes)

    true_pos = (label_ohe * pred_ohe).sum(0)
    false_pos = ((1 - label_ohe) * pred_ohe).sum(0)
    false_neg = (label_ohe * (1 - pred_ohe)).sum(0)

    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    avg_f1 = 2 * (precision * recall) / (precision + recall + eps)
    wts = label_ohe.sum(0)
    wtd_macro_f1 = (avg_f1 * wts).sum() / wts.sum()

    return wtd_macro_f1.item()


def confusion_matrix(outputs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Create confusion matrix
    Args:
        outputs: Logits of the network
        labels: Ground truth labels
        num_classes: Number of classes
    Returns:
        Confusion matrix as a tensor
    """
    num = labels.shape[0]
    conf_mat = torch.zeros((1, num, num_classes, num_classes), dtype=torch.int64)
    preds = (outputs).argmax(dim=1).to(torch.int64)
    conf_mat[0, range(num), labels, preds] = 1
    return conf_mat.sum(1, keepdim=True)
