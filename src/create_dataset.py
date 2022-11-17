#!/usr/bin/env python
"""Create synthetic dataset based on the avatars"""

import argparse
import glob
import json
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
        "-r",
        "--root",
        type=str,
        default="../datasets/AVATARS",
        help="Root directory for avatars",
    )
    parser.add_argument(
        "-d",
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
        "-c", "--std_cut", type=float, default=0.2, help="Standard deviaiton for cut along z axis"
    )
    parser.add_argument(
        "-a", "--std_angle", type=float, default=0.1, help="Standard deviaiton for rotation"
    )
    return parser.parse_args()


def create_synthetic_data_folders(output_dir: str) -> Tuple[str, str, str]:
    """Create folder structure for synthetic dataset
    Args:
        output_dir: Output directory
    """
    data_time_str = time.strftime("dataset_%Y%m%d-%H%M%S")

    output_dir = os.path.join(output_dir, data_time_str)

    out_back_dir = os.path.join(output_dir, "back")
    os.makedirs(out_back_dir, exist_ok=True)

    out_front_dir = os.path.join(output_dir, "front")
    os.makedirs(out_front_dir, exist_ok=True)

    return output_dir, out_back_dir, out_front_dir


def write_json(file_path: str, data: Dict[str, Any]) -> None:
    """Write data to a json file
    Args:
        file_path: File path
        data: Data to write
    """
    with open(file_path, "w", encoding="utf-8") as fout:
        json.dump(data, fout)


def save_camera_params(output_dir: str, camera_params: o3d.camera.PinholeCameraParameters) -> None:
    """Save camera parameters
    Args:
        output_dir: Output directory
        camera_params: Camera intrinsic and extrinsic matrices
        scale: Depth scale
    """
    data = {"extrinsics": camera_params.extrinsic.tolist()}
    file_path = os.path.join(output_dir, "extrinsics.json")
    write_json(file_path, data)

    data = {"height": camera_params.intrinsic.height, "width": camera_params.intrinsic.width}
    data.update(dict(zip(["fx", "fy"], camera_params.intrinsic.get_focal_length())))
    data.update(dict(zip(["cx", "cy"], camera_params.intrinsic.get_principal_point())))
    file_path = os.path.join(output_dir, "intrinsics.json")
    write_json(file_path, data)


def augment_data(
    visualizer: o3d.visualization.Visualizer,
    pcd: o3d.geometry.PointCloud,
    std_angle: float,
    std_cut: float,
) -> Tuple[o3d.geometry.Image, o3d.geometry.Image]:
    """Agment avatar point cloud
    Args:
    """
    # Random rotate pcd
    angles = RNG.normal(0.0, std_angle, 3)
    rot = PointCloudManip.rotate_pcd(pcd, angles)

    # Random cut geometry along z axis
    zcut = RNG.normal(rot.get_center()[2], std_cut)
    front, back = utils.cut_geometry(rot, (None, None, zcut))

    # Extract image
    back_depth_img = utils.capture_depth_from_camera(back, visualizer)
    front_depth_img = utils.capture_depth_from_camera(front, visualizer)

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
    """
    # Create output dirs
    out_dir, out_back_dir, out_front_dir = create_synthetic_data_folders(params["out_dir"])

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=params["width"], height=params["height"], visible=False)

    control = vis.get_view_control().convert_to_pinhole_camera_parameters()
    save_camera_params(out_dir, control)

    for avatar in avatar_list:
        name = Path(avatar).parts[-3] + "_" + os.path.basename(avatar)[:-4]
        print(name)
        # Read mesh
        mesh = o3d.io.read_triangle_mesh(avatar)
        mesh.compute_vertex_normals()

        # Convert to pcd
        pcd = mesh.sample_points_uniformly(number_of_points=params["num_points"])
        rot_pcd = PointCloudManip.rotate_pcd(pcd, (0, 0, -np.pi / 2))

        for i in tqdm(range(params["num_samples"])):
            # Augment avatar
            back_depth_img, front_depth_img = augment_data(
                vis, rot_pcd, params["std_angle"], params["std_cut"]
            )

            # Save as image
            cv2.imwrite(
                os.path.join(out_back_dir, name + f"_{i}.png"),
                np.asarray(back_depth_img).round().astype(np.uint16),
            )
            cv2.imwrite(
                os.path.join(out_front_dir, name + f"_{i}.png"),
                np.asarray(front_depth_img).round().astype(np.uint16),
            )

    vis.destroy_window()


def main() -> None:
    """Main function"""
    args = args_parser()

    avatar_list = glob.glob(args.root + "/**/*.obj", recursive=True)
    avatar_list = list(filter(lambda x: "MAMA" not in x, avatar_list))

    create_synthetic_dataset(avatar_list, vars(args))


if __name__ == "__main__":
    main()
