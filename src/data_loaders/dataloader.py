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
        params: Hyper-parameters
        file_name: Train/val/test csv file relative to the root
        transforms: Data augmentation to be done
    """

    def __init__(
        self,
        params: Params,
        file_name: str,
        transforms: Optional[tvt.Compose] = None,
    ) -> None:
        self.root = params.data_path
        self.file_list = pd.read_csv(os.path.join(self.root, file_name), header=None)
        self.transforms = transforms

    def _load_img(self, path_: str) -> torch.Tensor:
        """Load image and normalzie"""
        img = np.asarray(Image.open(path_))
        img_tensor = torch.tensor(img, dtype=torch.float32)

        return img_tensor.unsqueeze(0)

    def _normalization(self, img: torch.Tensor, invert: bool = False) -> torch.Tensor:
        """Normalize data to 0-1"""
        mask = img > 0.0
        min_ = img[mask].min()
        max_ = img.max()

        img[mask] -= min_
        img[mask] /= max_ - min_

        if invert:
            img[mask] = 1.0 - img[mask]
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an item from the dataset given the index idx"""
        im_name = self.file_list.iloc[idx][0]
        front_img = self._load_img(os.path.join(self.root, "front", im_name))
        back_img = self._load_img(os.path.join(self.root, "back", im_name))

        front_img = self._normalization(front_img)
        back_img = self._normalization(back_img)

        if self.transforms:
            front_img = self.transforms(front_img)
            back_img = self.transforms(back_img)

        back_mask = (back_img > 0.0).to(torch.bool)
        return front_img, back_img, back_mask

    def __len__(self) -> int:
        """Length of the dataset"""
        return len(self.file_list)


def get_transforms(params: Params, is_train: bool) -> tvt.Compose:
    """Data augmentation
    Args:
        params: Hyper parameters
        is_train: Train or eval mode
    Returns:
        Pytorch augmentations
    """
    trans = []
    if is_train:
        trans += [
            tvt.RandomAffine(degrees=0, scale=(2.0, 2.0)),
        ]
    else:
        trans += [
            tvt.RandomAffine(degrees=0, scale=(2.0, 2.0)),
        ]
    trans += [
        tvt.Resize((params.resize, params.resize), interpolation=tvt.InterpolationMode.NEAREST),
    ]
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

    for mode in modes:
        if mode == "train":
            trans = get_transforms(params, True)
            dataset = BackPredictionDataset(params, "train.csv", trans)
            shuffle = True
        else:
            trans = get_transforms(params, False)
            dataset = BackPredictionDataset(params, "test.csv", trans)
            shuffle = False

        dataloaders[mode] = DataLoader(
            dataset,
            batch_size=params.batch_size,
            num_workers=params.num_workers,
            pin_memory=params.pin_memory,
            shuffle=shuffle,
        )
    return dataloaders
