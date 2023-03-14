"""Training and evaluation functions"""

import os
from typing import Dict, Tuple

import torch
import torchvision.transforms as tvt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import VGG19_BN_Weights, vgg19_bn
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm.notebook import tqdm

from config_manager.manager import Params


def train_per_epoch(
    net: torch.nn.Module,
    vgg: torch.nn.Module,
    preprocess: tvt._presets.ImageClassification,
    train_dataloader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    params: Params,
) -> Tuple[float, float, float, float]:
    """Train for an epoch
    Args:
        net: Neural network
        vgg: VGG network
        preprocess: VGG pre-processor
        train_dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        params: Hyperparameters
    Returns:
        Epoch loss
    """
    net.train()

    epoch_loss, accuracy, f1_score, err = 0.0, 0.0, 0.0, 0.0
    for inputs, labels, mask in tqdm(train_dataloader):
        inputs = inputs.to(torch.device(params.device))
        labels = labels.to(torch.device(params.device))
        mask = mask.to(torch.device(params.device))

        optimizer.zero_grad()
        outputs = net(inputs, mask.to(torch.float32))
        label_out = net(labels, mask.to(torch.float32))
        with torch.no_grad():
            tmp = preprocess(labels.expand(-1, 3, -1, -1))
            vgg_outputs = vgg(tmp)
        loss, acc, f_1 = criterion(outputs, vgg_outputs, label_out, labels, mask, params)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        accuracy += acc
        f1_score += f_1
        err += (outputs[params.output_layer].detach() - labels.detach()).abs().mean()

    num = len(train_dataloader)
    return epoch_loss / num, accuracy / num, f1_score / num, err / num


def evaluate(
    net: torch.nn.Module,
    vgg: torch.nn.Module,
    preprocess: tvt._presets.ImageClassification,
    test_dataloader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    params: Params,
) -> Tuple[torch.Tensor, float, float, float, float]:
    """Evaluate modelh
    Args:
        net: Neural network
        vgg: VGG network
        preprocess: VGG pre-processor
        test_dataloader: Test dataloader
        criterion: Loss function
        params: Hyperparameters
    Returns:
        Evaluated batch and Epoch loss
    """
    net.eval()

    epoch_loss, accuracy, f1_score, err = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, labels, mask in test_dataloader:
            images = images.to(torch.device(params.device))
            labels = labels.to(torch.device(params.device))
            mask = mask.to(torch.device(params.device))

            outputs = net(images, mask.to(torch.float32))
            label_out = net(labels, mask.to(torch.float32))
            tmp = preprocess(labels.expand(-1, 3, -1, -1))
            vgg_outputs = vgg(tmp)
            loss, acc, f_1 = criterion(outputs, vgg_outputs, label_out, labels, mask, params)

            epoch_loss += loss.item()
            accuracy += acc
            f1_score += f_1
            err += (outputs[params.output_layer] - labels).abs().mean()

    num = len(test_dataloader)
    return outputs[params.output_layer], epoch_loss / num, accuracy / num, f1_score / num, err / num


def get_style_net(params: Params) -> Tuple[torch.nn.Module, tvt._presets.ImageClassification]:
    """Get network for comparing styles
    Args:
        params: Hyper-parameters
    Returns:
        Style neural network and it's pre-processor
    """
    weights = VGG19_BN_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    vmodel = vgg19_bn(weights=weights)
    for param in vmodel.parameters():
        param.requires_grad = False
    vmodel.eval()

    vgg = create_feature_extractor(vmodel, return_nodes=params.vgg_style)
    vgg.to(torch.device(params.device))

    return vgg, preprocess


def train_evaluate(
    base_model: torch.nn.Module,
    net: torch.nn.Module,
    dataloader: Dict[str, DataLoader],
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    params: Params,
) -> None:
    """Train and evaluate model
    Args:
        base_model: Base model to plot the graph
        net: Neural network
        discriminator: Discriminator network
        dataloader: Train/test dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Scheduler
        params: Hyperparameters
    Returns:
        train and test losses
    """
    print("Writing graph ...")
    writer = SummaryWriter(params.save_path)
    images, _, _ = next(iter(dataloader["train"]))
    writer.add_graph(base_model, images)

    net.to(torch.device(params.device))
    vgg, preprocess = get_style_net(params)

    best_loss = 100.0
    print("Training started ...")
    for epoch in range(params.epochs):
        print(f"\nEpoch: {epoch + 1}/{params.epochs}")
        t_loss, acc, f1_score, err = train_per_epoch(
            net, vgg, preprocess, dataloader["train"], criterion, optimizer, params
        )
        print(
            f"Train loss: {t_loss:.3f}, error: {err:.3f}, accuracy: {acc:.3f}, F1: {f1_score:.3f}"
        )
        writer.add_scalar("Loss/train", t_loss, epoch)
        writer.add_scalar("accuracy/train", acc, epoch)
        writer.add_scalar("F1/train", f1_score, epoch)
        writer.add_scalar("Error/train", err, epoch)

        output, e_loss, acc, f1_score, err = evaluate(
            net, vgg, preprocess, dataloader["test"], criterion, params
        )
        print(f"Test loss: {e_loss:.3f}, error: {err:.3f}, accuracy: {acc:.3f}, F1: {f1_score:.3f}")
        writer.add_scalar("Loss/test", e_loss, epoch)
        writer.add_scalar("accuracy/test", acc, epoch)
        writer.add_scalar("F1/test", f1_score, epoch)
        writer.add_scalar("Error/test", err, epoch)
        writer.add_images("Test batch", output, epoch)

        scheduler.step()
        writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch)

        if e_loss < best_loss:
            best_loss = e_loss
            print("Saving new best model ...")
            torch.save(net.state_dict(), os.path.join(params.save_path, "best_net.pt"))
    writer.close()

    print("Saving last model ...")
    torch.save(net.state_dict(), os.path.join(params.save_path, f"last_{params.epochs}_net.pt"))
    params.save(os.path.join(params.save_path, "params.yml"))
