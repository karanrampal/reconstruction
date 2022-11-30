"""Training and evaluation functions"""

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config_manager.manager import Params
from model.net import BPModel


def train_per_epoch(
    net: torch.nn.Module,
    train_dataloader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    params: Params,
) -> float:
    """Train for an epoch
    Args:
        net: Neural network
        train_dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        params: Hyperparameters
    Returns:
        Epoch loss
    """
    net.train()

    epoch_loss = []
    for inputs, labels in tqdm(train_dataloader):
        inputs = inputs.to(params.device)
        labels = labels.to(params.device)

        optimizer.zero_grad()
        outputs = net(inputs)
        label_outputs = net(labels)
        loss = criterion(outputs, label_outputs, labels, params)
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)


def evaluate(
    net: torch.nn.Module,
    test_dataloader: DataLoader,
    criterion: torch.nn.modules.loss._Loss,
    params: Params,
) -> Tuple[torch.Tensor, float]:
    """Evaluate modelh
    Args:
        net: Neural network
        test_dataloader: Test dataloader
        criterion: Loss function
        params: Hyperparameters
    Returns:
        Evaluated batch and Epoch loss
    """
    net.eval()

    epoch_loss = []
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(params.device)
            labels = labels.to(params.device)

            outputs = net(images)
            label_outputs = net(labels)
            loss = criterion(outputs, label_outputs, labels, params)

            epoch_loss.append(loss.item())
    return outputs[params.output_layer], np.mean(epoch_loss)


# pylint: disable=too-many-arguments
def train_evaluate(
    net: torch.nn.Module,
    dataloader: Dict[str, DataLoader],
    criterion: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    params: Params,
) -> Tuple[List[float], List[float]]:
    """Train and evaluate model
    Args:
        net: Neural network
        dataloader: Train/test dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Scheduler
        params: Hyperparameters
    Returns:
        train and test losses
    """
    writer = SummaryWriter(params.tb_path)
    images, _ = next(iter(dataloader["train"]))
    writer.add_graph(BPModel(params.filters, params.kernels, params.embeddings), images)

    net = net.to(params.device)

    best_loss = 100.0
    train_loss, test_loss = [], []
    for epoch in range(params.epochs):
        t_loss = train_per_epoch(net, dataloader["train"], criterion, optimizer, params)
        print(f"Epoch: {epoch + 1}, Train loss: {t_loss:.3f}")
        writer.add_scalar("Loss/train", t_loss, epoch)
        train_loss.append(t_loss)

        output, e_loss = evaluate(net, dataloader["test"], criterion, params)
        print(f"Test loss: {e_loss:.3f}")
        writer.add_scalar("Loss/test", e_loss, epoch)
        writer.add_images("Test batch", output, epoch)
        test_loss.append(e_loss)

        scheduler.step()
        writer.add_scalar("Learning rate", scheduler.get_last_lr()[0], epoch)

        if e_loss < best_loss:
            best_loss = e_loss
            print("Saving new best model ...")
            torch.save(net.state_dict(), os.path.join(params.save_path, "best_net.pt"))
    writer.close()

    print("Saving last model ...")
    torch.save(net.state_dict(), os.path.join(params.save_path, f"last_{params.epochs}_net.pt"))

    return train_loss, test_loss
