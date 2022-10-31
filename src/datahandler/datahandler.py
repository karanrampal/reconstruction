"""RGBD data loading, saving for experimentation"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d


class DataHandler:
    """Data handler for loading, saving realsense camera data
    Args:
        root: Root directory
        img_path: Relative path of images i.e. RGB and depth
        calib_path: Relative path of calibration data
    """

    def __init__(
        self,
        root: str,
        img_path: str,
        calib_path: str,
        cams_to_keep: Optional[List[str]] = None,
    ) -> None:
        self.root = root
        self.img_path = os.path.join(root, img_path)
        self.calibration_path = os.path.join(root, calib_path)
        self.cams_to_keep = cams_to_keep

    def _get_serial(self, name: str) -> str:
        """Get serial id from file name"""
        return name.split("_")[-1].split(".")[0]

    def _filter_serial(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter the camera serial id to keep"""
        if self.cams_to_keep:
            return {
                serial: val
                for serial, val in data.items()
                if serial in self.cams_to_keep
            }
        return data

    def _read_json(self, path: str) -> Dict[str, Any]:
        """Read json file"""
        with open(path, "r", encoding="utf-8") as fout:
            data = json.load(fout)

        return data

    def _load_imgs(self, name: str, mode: str) -> Dict[str, np.ndarray]:
        """Load images helper function
        Args:
            name: Name of person
            mode: RGB or depth
        Returns:
            Mapping of camera serial id and image
        """
        path = os.path.join(self.img_path, name, mode)
        paths = [os.path.join(path, fin) for fin in os.listdir(path) if ".png" in fin]
        if not paths:
            print(f"No .png files in {path}!")
            return {}
        imgs = {
            self._get_serial(img_path): np.asarray(o3d.io.read_image(img_path))
            for img_path in paths
        }

        return self._filter_serial(imgs)

    def load_images(
        self, name: str
    ) -> Tuple[Dict[str, o3d.geometry.Image], Dict[str, np.ndarray]]:
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
        path_ = os.path.join(self.calibration_path, "device_depth_scales.json")
        if not os.path.isfile(path_):
            print(f"No {path_} such file!")
            return {}
        data = self._read_json(path_)

        return self._filter_serial(data)

    def load_transformations(self) -> Dict[str, np.ndarray]:
        """Load transformations for each camera exceptht the main camera"""
        path_ = os.path.join(self.calibration_path, "transformations.json")
        if not os.path.isfile(path_):
            print(f"No {path_} such file!")
            return {}
        data = self._read_json(path_)
        data_filt = self._filter_serial(data)
        out = {
            serial: np.asarray(mat["transformation_matrix"])
            for serial, mat in data_filt.items()
        }

        return out

    def load_intrinsics(self) -> Dict[str, o3d.camera.PinholeCameraIntrinsic]:
        """Load intrinsics for each camera"""
        tmp = {
            f.split("_")[0]: os.path.join(self.calibration_path, f)
            for f in os.listdir(self.calibration_path)
            if "intrinsics.json" in f
        }
        if not tmp:
            print(f"No intrinsic files in {self.calibration_path}!")
            return {}
        intrin_files = self._filter_serial(tmp)
        out = {}
        for serial, path in intrin_files.items():
            data = self._read_json(path)
            data["cx"] = data.pop("ppx")
            data["cy"] = data.pop("ppy")
            out[serial] = o3d.camera.PinholeCameraIntrinsic(**data)

        return out

    def _save_imgs(
        self,
        img_list: List[np.ndarray],
        save_path: str,
        names: Optional[List[str]] = None,
    ) -> None:
        """Save image helper method
        Args:
            img_list: List of images
            save_path: Path to save output
            names: Names of each file
        """
        os.makedirs(save_path, exist_ok=True)

        for i, img in enumerate(img_list):
            name = names[i] if names else f"{i:06}.png"
            o3d.io.write_image(os.path.join(save_path, name), o3d.geometry.Image(img))

    def save_images(
        self,
        save_path: str,
        rgb: List[np.ndarray],
        depth: Optional[List[np.ndarray]] = None,
        names: Optional[List[str]] = None,
    ) -> None:
        """Save images
        Args:
            save_path: Path to save output
            rgb: Color images
            depth: Depth images
            names: List of names for each image
        """
        path_ = os.path.join(save_path, "color")
        self._save_imgs(rgb, path_, names)
        if depth:
            path_ = os.path.join(save_path, "depth")
            self._save_imgs(depth, path_, names)
