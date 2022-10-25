"""RGBD data loading, saving for experimentation"""

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d

DataType = Union[float, List[List[float]], Dict[str, float]]


class DataHandler:
    """Data handler for loading, saving realsense camera data
    Args:
        root: Root directory
        img_path: Relative path of images i.e. RGB and depth
        calib_path: Relative path of calibration data
    """
    def __init__(self, root: str, img_path: str, calib_path: str, cams_to_keep: Optional[List[str]] = None) -> None:
        self.root = root
        self.img_path = os.path.join(root, img_path)
        self.calibration_path = os.path.join(root, calib_path)
        self.cams_to_keep = cams_to_keep

    def _get_serial(self, name: str) -> str:
        """Get serial id from file name"""
        return name.split("_")[-1].split(".")[0]

    def _filter_serial(self, data: Dict[str, DataType]):
        """Filter the camera serial id to keep"""
        if self.cams_to_keep:
            return {serial: val for serial, val in data.items() if serial in self.cams_to_keep}
        return data

    def _read_json(self, path: str) -> Dict[str, DataType]:
        """Read json file"""
        with open(path, "r", encoding="utf-8") as fout:
            data = json.load(fout)

        return self._filter_serial(data)
    
    def _load_imgs(self, name: str, type: str) -> Dict[str, o3d.geometry.Image]:
        """Load images helper function
        Args:
            name: Name of person
            type: RGB or depth
        Returns:
            Mapping of camera serial id and image
        """
        path = os.path.join(self.img_path, name, type)
        paths = [os.path.join(path, fin) for fin in os.listdir(path)]
        imgs = {self._get_serial(img_path): o3d.io.read_image(img_path) for img_path in paths}

        return self._filter_serial(imgs)
    
    def load_images(self, name: str) -> Tuple[Dict[str, o3d.geometry.Image], Dict[str, o3d.geometry.Image]]:
        """Load images
        Args:
            name: Name of person
        Returns:
            Mapping of camera serial id and image
        """
        color_images = self._load_imgs(name, "RGB")
        depth_images = self._load_imgs(name, "depth")

        return color_images, depth_images

    def load_depth_scales(self) -> Dict[str, float]:
        """Load depth scales for each camera"""
        data = self._read_json(os.path.join(self.calibration_path, "device_depth_scales.json"))

        return data

    def load_transformations(self) -> Dict[str, np.ndarray]:
        """Load transformations for each camera exceptht the main camera"""
        data = self._read_json(os.path.join(self.calibration_path, "transformations.json"))
        out = {serial: np.asarray(mat) for serial, mat in data.items()}

        return out

    def load_intrinsics(self) -> Dict[str, o3d.camera.PinholeCameraIntrinsic]:
        """Load intrinsics for each camera"""
        intrin_files = {
            f.split("_")[0]: os.path.join(self.calibration_path, f)
            for f in os.listdir(self.calibration_path)
            if "intrinsics.json" in f
        }
        intrin_files = self._filter_serial(intrin_files)
        out = {}
        for serial, path in intrin_files.items():
            data = self._read_json(path)
            out[serial] = o3d.camera.PinholeCameraIntrinsic(**data[serial])

        return out
