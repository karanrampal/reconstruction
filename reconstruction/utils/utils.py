"""Utility functions"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def plot_rgbd(rgb: List[np.ndarray], depth: List[np.ndarray]) -> None:
    """Plot rgb and d images side by side
    Args:
        rgb: List of rgb images to plot
        depth: List of depth images to plot
    """
    num = len(rgb)
    fig, axi = plt.subplots(num, 2, figsize=(12, num * 3))
    for i in range(num):
        axi[i, 0].imshow(rgb[i], cmap="RdBu")
        cb_ = axi[i, 1].imshow(depth[i], cmap="RdBu")
        fig.colorbar(cb_, ax=axi[i, 1])
    fig.tight_layout()


def create_point_cloud(
    rgbs: Dict[str, np.ndarray],
    depths: Dict[str, np.ndarray],
    scales: Dict[str, float],
    transformations: Dict[str, np.ndarray],
    intrinsics: Dict[str, o3d.camera.PinholeCameraIntrinsic],
) -> o3d.geometry.PointCloud:
    """Create point cloud from rgbd images
    Args:
        rgbs: RGB images
        depths: Depth images
        scales: Depth scales for each camera
        transformations: Transformations for each camera
        intrinsics: Camera intrinsics
    Returns:
        Point cloud
    """
    pcd = o3d.geometry.PointCloud()
    for serial in depths.keys():
        color_img = o3d.geometry.Image(rgbs[serial])
        depth_img = o3d.geometry.Image(depths[serial])
        intrinsic = intrinsics[serial]
        transform = transformations.get(serial)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_img,
            depth_img,
            depth_scale=1 / scales[serial],
            depth_trunc=4,
            convert_rgb_to_intensity=False,
        )
        pc_ = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
        if transform is not None:
            pc_.transform(transform)
        pcd += pc_
    return pcd


def create_mesh_tsdf(
    rgbs: Dict[str, np.ndarray],
    depths: Dict[str, np.ndarray],
    scales: Dict[str, float],
    transformations: Dict[str, np.ndarray],
    intrinsics: Dict[str, o3d.camera.PinholeCameraIntrinsic],
) -> o3d.geometry.TriangleMesh:
    """Create mesh from rgbd images
    Args:
        rgbs: RGB images
        depths: Depth images
        scales: Depth scales for each camera
        transformations: Transformations for each camera
        intrinsics: Camera intrinsics
    Returns:
        Triangle mesh
    """
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for serial in rgbs.keys():
        color = o3d.geometry.Image(rgbs[serial])
        depth = o3d.geometry.Image(depths[serial])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1 / scales[serial],
            depth_trunc=4.0,
            convert_rgb_to_intensity=False,
        )
        mat = transformations.get(serial, np.eye(4))
        volume.integrate(rgbd, intrinsics[serial], np.linalg.inv(mat))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh
