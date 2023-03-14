#!/usr/bin/env python
"""Create synthetic dataset based on the avatars"""

import argparse
import glob
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d
from numpy.random import MT19937, RandomState, SeedSequence
from scipy.signal import convolve2d
from scipy.ndimage import grey_dilation
from tqdm import tqdm

from pointcloud.pointcloud import PointCloudManip
from utils import utils

RNG = RandomState(MT19937(SeedSequence(123456789)))
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def args_parser() -> argparse.Namespace:
    """Argument parser for the CLI"""
    parser = argparse.ArgumentParser(description="CLI for creating synthetic dataset")
    parser.add_argument(
        "-d",
        "--root_dir",
        type=str,
        default="../datasets/caesar_pointclouds",
        help="Root directory for avatars",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for created datasets",
    )
    parser.add_argument(
        "-p", "--num_points", type=int, default=20000, help="Number of points to sample from mesh"
    )
    parser.add_argument("-n", "--num_samples", type=int, default=1, help="Number of augmentations")
    parser.add_argument("-s", "--scale", type=float, default=12.0, help="Scale depth values")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument(
        "-t", "--std_translate", type=float, default=0.0, help="Standard deviaiton for translation"
    )
    parser.add_argument(
        "-a", "--std_angle", type=float, default=0.0, help="Standard deviaiton for rotation"
    )
    parser.add_argument(
        "-r",
        "--radius_constant",
        type=int,
        default=1000,
        help="Constant for radius of spherical projection",
    )
    parser.add_argument("--kernel_size", type=int, default=5, help="Post processing filter size")
    return parser.parse_args()


def create_synthetic_data_folders(params: Dict[str, Any]) -> None:
    """Create folder structure for synthetic dataset
    Args:
        params: Hyper parameters
    """
    params["output_dir"] = os.path.join(
        params["output_dir"], time.strftime("dataset_%Y%m%d-%H%M%S")
    )

    os.makedirs(os.path.join(params["output_dir"], "back"), exist_ok=True)

    os.makedirs(os.path.join(params["output_dir"], "front"), exist_ok=True)


def save_camera_params(
    output_dir: str, camera_params: o3d.camera.PinholeCameraParameters, scale: float
) -> None:
    """Save camera parameters
    Args:
        output_dir: Output directory
        camera_params: Camera intrinsic and extrinsic matrices
        scale: Depth scale
    Raises:
        ValueError if output directory does not exist
    """
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory: {output_dir} does not exist!")

    data = {"depth_scale": scale}
    file_path = os.path.join(output_dir, "depth_scale.json")
    utils.write_json(file_path, data)

    data = {"extrinsics": camera_params.extrinsic.tolist()}
    file_path = os.path.join(output_dir, "extrinsics.json")
    utils.write_json(file_path, data)

    data = {"height": camera_params.intrinsic.height, "width": camera_params.intrinsic.width}
    data.update(dict(zip(["fx", "fy"], camera_params.intrinsic.get_focal_length())))
    data.update(dict(zip(["cx", "cy"], camera_params.intrinsic.get_principal_point())))
    file_path = os.path.join(output_dir, "intrinsics.json")
    utils.write_json(file_path, data)


def augment_data(
    pcd: o3d.geometry.PointCloud,
    std_angle: float,
    std_translate: float,
    radius_constant: int,
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """Agment avatar point cloud
    Args:
        pcd: Point cloud
        std_angle: Standard deviation for rotation
        std_translate: Standard deviation for translation
        radius_constant: Constant for radius of spherical projection
    Returns:
        Front and back depth images
    """
    # Remove hidden points from pcd
    locs = RNG.normal(pcd.get_center()[:2], (std_translate, std_translate), 2)
    front, back = PointCloudManip.remove_hidden_points(pcd, locs, radius_constant)

    # Random rotate front and back pcds
    angles = RNG.normal(0.0, std_angle, 3)
    rot_front = PointCloudManip.rotate_pcd(front, angles)
    rot_back = PointCloudManip.rotate_pcd(back, angles)

    return rot_front, rot_back


def create_synthetic_dataset(avatar_list: List[str], params: Dict[str, Any]) -> None:
    """Create dataset by writing depth images for front, back of a mesh
    Args:
        avatar_list: Paths of each avatar
        params: Hyper-parameters, following
            output_dir: Output directory
            height: Image height
            width: Image width
            num_samples: Number of samples to augment
            num_points: Number of points in pcd from mesh
            sigma_angle: Standard deviation for rotation
            sigma_cut: Standard deviaiton for cut along x axis
            scale: Scale depth values for visualization
    Raises:
        Value error if Avatar list is empty
    """
    if not avatar_list:
        raise ValueError("Avatar list is empty!")

    create_synthetic_data_folders(params)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=params["width"], height=params["height"], visible=False)

    # Save camera parameters
    control = vis.get_view_control().convert_to_pinhole_camera_parameters()
    save_camera_params(params["output_dir"], control, params["scale"])

    for avatar in avatar_list:
        # Create output dirs
        name = os.path.basename(avatar)[:-4]
        print(name)

        pcd = o3d.io.read_point_cloud(avatar)
        pcd = PointCloudManip.rotate_pcd(pcd, (0, np.pi / 2, -np.pi / 4))

        for j in tqdm(range(params["num_samples"])):
            # Augment avatar
            front_pcd, back_pcd = augment_data(
                pcd, params["std_angle"], params["std_translate"], params["radius_constant"]
            )

            # Save as image
            utils.save_depth_as_image(
                front_pcd,
                vis,
                os.path.join(params["output_dir"], "front", f"{name}_{j}.png"),
                params["scale"],
            )
            utils.save_depth_as_image(
                back_pcd,
                vis,
                os.path.join(params["output_dir"], "back", f"{name}_{j}.png"),
                params["scale"],
            )

    vis.destroy_window()


def post_processing(params: Dict[str, Any]) -> None:
    """Post process the images e.g. median filter to remove holes
    Args:
        params: Hyper-parameters
    """
    print("\nPost processing images ...")
    out_file_list = glob.glob(params["output_dir"] + "/**/*.png", recursive=True)
    for i in tqdm(out_file_list):
        img = o3d.io.read_image(i)
        im_filt = grey_dilation(img, size=(params["kernel_size"], params["kernel_size"]))
        o3d.io.write_image(i, o3d.geometry.Image(im_filt))


def calculate_normal_vectors(params: Dict[str, Any]) -> None:
    """Calculate normal vectors for the depth image
    Args:
        params: Hyper-parameters
    """
    print("\nNormal vector calculation ...")
    out_file_list = glob.glob(params["output_dir"] + "/**/*.png", recursive=True)

    grady = np.array([-1, 0, 1]).reshape(-1, 1)
    gradx = grady.T
    normal = np.zeros((params["height"], params["width"], 3), dtype=np.float32)
    normal[:, :, -1] = -1.0

    for i in tqdm(out_file_list):
        img = o3d.io.read_image(i)

        normal[:, :, 0] = -convolve2d(img, gradx, mode="same")
        normal[:, :, 1] = -convolve2d(img, grady, mode="same")
        denom = np.expand_dims(np.linalg.norm(normal, axis=-1), -1)
        denom[denom == 0.0] = 1.0
        normal /= denom

        path_ = os.path.join(os.path.dirname(i), os.path.basename(i).replace(".png", ".npy"))
        np.save(path_, normal)


def main() -> None:
    """Main function"""
    args = args_parser()

    avatar_list = glob.glob(args.root_dir + "/**/*.ply", recursive=True)

    create_synthetic_dataset(avatar_list, vars(args))
    post_processing(vars(args))
    # calculate_normal_vectors(vars(args))


if __name__ == "__main__":
    main()
