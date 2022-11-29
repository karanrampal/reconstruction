"""Define custom dataset class extending the Pytorch Dataset class"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config_manager.manager import Params


class BackPredictionDataset(Dataset):
    """Custom class for Back depth image prediction dataset
    Args:
        root: Directory containing the dataset
        file_name: Train/val/test csv file relative to the root
        transforms: Data augmentation to be done
    """

    def __init__(
        self,
        root: str,
        file_name: str,
        transforms: Optional[tvt.Compose] = None,
    ) -> None:
        self.root = root
        self.file_list = pd.read_csv(os.path.join(root, file_name), header=None)
        self.transforms = transforms
        self.front_img = np.load(os.path.join(root, "avg_front_image.npy"))

    def _load_img(self, path_: str, avg_img: np.ndarray) -> torch.Tensor:
        """Load image and normalzie"""
        img = np.asarray(Image.open(path_))
        img_tensor = torch.tensor(img, dtype=torch.float32)

        result = ((img_tensor - avg_img.min()) / (avg_img.max() - avg_img.min())).unsqueeze(0)

        return result

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an item from the dataset given the index idx"""
        im_name = self.file_list.iloc[idx][0]
        front_img = self._load_img(os.path.join(self.root, "front", im_name), self.front_img)
        back_img = self._load_img(os.path.join(self.root, "back", im_name), self.front_img)

        front_img = tvt.functional.affine(
            front_img, shear=0.0, scale=2.0, translate=(0, 0), angle=0.0
        )
        back_img = tvt.functional.affine(
            back_img, shear=0.0, scale=2.0, translate=(0, 0), angle=0.0
        )

        if self.transforms:
            front_img = self.transforms(front_img)
            back_img = self.transforms(back_img)

        mask = torch.zeros_like(back_img)
        mask[back_img > 0.0] = 1.0
        return front_img, back_img, mask

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.file_list)


def get_transforms(params: Params) -> tvt.Compose:
    """Data augmentation
    Args:
        params: Hyper parameters
    Returns:
        Pytorch augmentations
    """
    trans = [tvt.Resize((params.resize, params.resize))]
    return tvt.Compose(trans)


def get_dataloaders(modes: List[str], params: Params) -> Dict[str, DataLoader]:
    """Get DataLoader objects.
    Args:
        modes: Mode of operation i.e. 'train', 'test'
        params: Hyperparameters
    Returns:
        DataLoader object for each mode
    """
    dataloaders = {}
    trans = get_transforms(params)

    for mode in modes:
        if mode == "train":
            dataset = BackPredictionDataset(params.data_path, "train.csv", trans)
            shuffle = True
        else:
            dataset = BackPredictionDataset(params.data_path, "test.csv", trans)
            shuffle = False

        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            shuffle=shuffle,
        )
    return dataloaders
