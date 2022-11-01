"""Point cloud manipulation"""

from typing import Dict, Tuple

import numpy as np
import open3d as o3d


class PointCloudManip:
    """Create, manipulate point clouds"""

    @classmethod
    def create_point_cloud(
        cls,
        images: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]],
        scales: Dict[str, float],
        transformations: Dict[str, np.ndarray],
        intrinsics: Dict[str, o3d.camera.PinholeCameraIntrinsic],
    ) -> o3d.geometry.PointCloud:
        """Create point cloud from rgbd images
        Args:
            images: RGB and depth images
            scales: Depth scales for each camera
            transformations: Transformations for each camera
            intrinsics: Camera intrinsics
        Returns:
            Point cloud
        """
        rgbs, depths = images
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

    @classmethod
    def distance_filter(
        cls,
        pcd: o3d.geometry.PointCloud,
        pcd_bg: o3d.geometry.PointCloud,
        thr: float = 0.05,
    ) -> o3d.geometry.PointCloud:
        """Filter point cloud to remove the backgound using distance of each point
        Args:
            pcd: Point cloud of scene plus object of interest
            pcd_bg: Point cloud of background
            thr: Threshold for distance filtering
        Returns:
            Filtered point cloud of the object of interest
        """
        dists = pcd.compute_point_cloud_distance(pcd_bg)
        dists = np.asarray(dists)
        ind = np.where(dists > thr)[0]
        pcd_without_bg = pcd.select_by_index(ind)
        return pcd_without_bg

    @classmethod
    def segment_out_plane(
        cls,
        pcd: o3d.geometry.PointCloud,
        thr: float = 0.05,
        ransac_n: int = 3,
        num_iterations: int = 1000,
    ) -> o3d.geometry.PointCloud:
        """Segmentation of a plane from point cloud
        Args:
            pcd: Point cloud of scene plus object of interest
            thr: Threshold for distance filtering
            ransac_n: Number of points to sample in RANSAC
            num_iterations: Number of iterations of RANSAC
        Returns:
            Segmented point cloud
        """
        _, inliers = pcd.segment_plane(
            distance_threshold=thr, ransac_n=ransac_n, num_iterations=num_iterations
        )
        outliers = pcd.select_by_index(inliers, invert=True)
        return outliers

    @classmethod
    def crop_pcd(
        cls,
        pcd: o3d.geometry.PointCloud,
        min_bound: Tuple[float, float, float],
        max_bound: Tuple[float, float, float],
    ) -> o3d.geometry.PointCloud:
        """Segmentation of a plane from point cloud
        Args:
            pcd: Point cloud of scene plus object of interest
            min_bound: Min bounds for x, y, z
            max_bound: Max bounds for x, y, z
        Returns:
            Cropped point cloud
        """
        assert len(min_bound) == 3, "Min bounds for x,y,z required"
        assert len(max_bound) == 3, "Max bounds for x,y,z required"

        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )
        return pcd.crop(bbox)
