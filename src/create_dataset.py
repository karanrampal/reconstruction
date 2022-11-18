#!/usr/bin/env python
"""Create synthetic dataset based on the avatars"""

import argparse
import glob
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d
from numpy.random import MT19937, RandomState, SeedSequence
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
        default="../datasets/AVATARS",
        help="Root directory for avatars",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="output",
        help="Output directory for created datasets",
    )
    parser.add_argument(
        "-p", "--num_points", type=int, default=20000, help="Number of points to sample from mesh"
    )
    parser.add_argument("-s", "--num_samples", type=int, default=20, help="Number of augmentations")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument(
        "-c", "--std_translate", type=float, default=0.2, help="Standard deviaiton for translation"
    )
    parser.add_argument(
        "-a", "--std_angle", type=float, default=0.1, help="Standard deviaiton for rotation"
    )
    parser.add_argument(
        "-r",
        "--radius_constant",
        type=int,
        default=1000,
        help="Constant for radius of spherical projection",
    )
    return parser.parse_args()


def create_synthetic_data_folders(output_dir: str, class_id: str) -> Tuple[str, str]:
    """Create folder structure for synthetic dataset
    Args:
        output_dir: Output directory
        class_id: Class id of the avatar
    Returns:
        Names of the front and back output directories
    """
    out_back_dir = os.path.join(output_dir, class_id, "back")
    os.makedirs(out_back_dir, exist_ok=True)

    out_front_dir = os.path.join(output_dir, class_id, "front")
    os.makedirs(out_front_dir, exist_ok=True)

    return out_back_dir, out_front_dir


def save_camera_params(output_dir: str, camera_params: o3d.camera.PinholeCameraParameters) -> None:
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

    data = {"extrinsics": camera_params.extrinsic.tolist()}
    file_path = os.path.join(output_dir, "extrinsics.json")
    utils.write_json(file_path, data)

    data = {"height": camera_params.intrinsic.height, "width": camera_params.intrinsic.width}
    data.update(dict(zip(["fx", "fy"], camera_params.intrinsic.get_focal_length())))
    data.update(dict(zip(["cx", "cy"], camera_params.intrinsic.get_principal_point())))
    file_path = os.path.join(output_dir, "intrinsics.json")
    utils.write_json(file_path, data)


def augment_data(
    visualizer: o3d.visualization.Visualizer,
    pcd: o3d.geometry.PointCloud,
    std_angle: float,
    std_translate: float,
    radius_constant: int,
) -> Tuple[o3d.geometry.Image, o3d.geometry.Image]:
    """Agment avatar point cloud
    Args:
        visualizer: Open3d visualizer
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
    rot_back = PointCloudManip.rotate_pcd(back, angles)
    rot_front = PointCloudManip.rotate_pcd(front, angles)

    # Extract image
    back_depth_img = utils.capture_depth_from_camera(rot_back, visualizer)
    front_depth_img = utils.capture_depth_from_camera(rot_front, visualizer)

    return back_depth_img, front_depth_img


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
    Raises:
        Value error if Avatar list is empty
    """
    if not avatar_list:
        raise ValueError("Avatar list is empty!")

    output_dir = os.path.join(params["out_dir"], time.strftime("dataset_%Y%m%d-%H%M%S"))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=params["width"], height=params["height"], visible=False)

    for i, avatar in enumerate(avatar_list):
        # Create output dirs
        name = Path(avatar).parts[-3] + "_" + os.path.basename(avatar)[:-4]
        print(name)
        out_back_dir, out_front_dir = create_synthetic_data_folders(output_dir, name)

        # Read mesh
        mesh = o3d.io.read_triangle_mesh(avatar)
        mesh.compute_vertex_normals()

        # Convert to pcd
        pcd = mesh.sample_points_uniformly(number_of_points=params["num_points"])
        pcd = PointCloudManip.rotate_pcd(pcd, (0, 0, -np.pi / 2))

        for j in tqdm(range(params["num_samples"])):
            # Augment avatar
            back_depth_img, front_depth_img = augment_data(
                vis, pcd, params["std_angle"], params["std_translate"], params["radius_constant"]
            )

            # Save as image
            cv2.imwrite(
                os.path.join(out_back_dir, f"{i}_{j}.png"),
                np.asarray(back_depth_img).round().astype(np.uint16),
            )
            cv2.imwrite(
                os.path.join(out_front_dir, f"{i}_{j}.png"),
                np.asarray(front_depth_img).round().astype(np.uint16),
            )

    # Save camera parameters
    control = vis.get_view_control().convert_to_pinhole_camera_parameters()
    save_camera_params(output_dir, control)

    vis.destroy_window()


def main() -> None:
    """Main function"""
    args = args_parser()

    avatar_list = glob.glob(args.root_dir + "/**/*.obj", recursive=True)
    avatar_list = list(filter(lambda x: "MAMA" not in x, avatar_list))

    create_synthetic_dataset(avatar_list, vars(args))


if __name__ == "__main__":
    main()
