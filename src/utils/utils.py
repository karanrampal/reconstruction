"""Utility functions"""

from typing import Dict, List, Optional, Tuple

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
    if num == 1:
        axi = axi.reshape(1, -1)
    for i in range(num):
        axi[i, 0].imshow(rgb[i], cmap="RdBu")
        cb_ = axi[i, 1].imshow(depth[i], cmap="RdBu")
        fig.colorbar(cb_, ax=axi[i, 1])
    fig.tight_layout()


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


def cut_geometry(
    geometry: o3d.geometry.Geometry, cuts: Tuple[Optional[float], Optional[float], Optional[float]]
) -> Tuple[o3d.geometry.Geometry, o3d.geometry.Geometry]:
    """Cut geometry along an axis
    Args:
        geometry: Open3d mesh or point cloud
        cuts: Cuts along the x, y, z axis
    Returns:
        Two sub geometries
    """
    min_bounds = geometry.get_min_bound()
    max_bounds = geometry.get_max_bound()

    b_max_bound = [y if y is not None else x for x, y in zip(max_bounds, cuts)]
    f_min_bound = [y if y is not None else x for x, y in zip(min_bounds, cuts)]

    b_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bounds, max_bound=b_max_bound)
    f_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=f_min_bound, max_bound=max_bounds)

    back = geometry.crop(b_bbox)
    front = geometry.crop(f_bbox)

    return front, back


def capture_depth_from_camera(
    geometry: o3d.geometry.Geometry, visualizer: o3d.visualization.Visualizer
) -> o3d.geometry.Image:
    """Capture depth image from camera buffer
    Args:
        geometry: Open3d geometry such as mesh or point cloud
        visualizer: Open3d visualizer
    """
    visualizer.add_geometry(geometry)
    visualizer.poll_events()
    visualizer.update_renderer()
    depth_img = visualizer.capture_depth_float_buffer()
    visualizer.clear_geometries()
    return depth_img
