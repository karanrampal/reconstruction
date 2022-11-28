"""Define custom dataset class extending the Pytorch Dataset class"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as tvt
from PIL import Image
from torch.utils.data import Dataset


class BackPredictionDataset(Dataset):
    """Custom class for Back depth image prediction dataset
    Args:
        root: Directory containing the dataset
        file_list: List of the train/val/test file relative to the root
        front_img: Average front image
        transforms: Data augmentation to be done
    Raises:
        Value error if mode is incorrect
    """

    def __init__(
        self,
        root: str,
        file_list: List[str],
        front_img: np.ndarray,
        transforms: Optional[tvt.Compose] = None,
    ) -> None:
        self.root = root
        self.file_list = file_list
        self.transforms = transforms
        self.front_img = front_img

    def _load_img(self, path_: str, avg_img: np.ndarray) -> torch.Tensor:
        """Load image and normalzie"""
        img = np.asarray(Image.open(path_))
        img_tensor = torch.tensor(img, dtype=torch.float32)

        result = ((img_tensor - avg_img.min()) / (avg_img.max() - avg_img.min())).unsqueeze(0)

        return result

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get an item from the dataset given the index idx"""
        im_name = self.file_list[idx]
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
