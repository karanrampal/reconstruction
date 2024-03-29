"""Point cloud manipulation"""

import copy
from typing import Dict, Tuple, Union

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
        if pcd.is_empty():
            raise ValueError("Point cloud is empty!")
        if pcd_bg.is_empty():
            raise ValueError("Background point cloud is empty!")

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
        if pcd.is_empty():
            raise ValueError("No points in point cloud!")

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
        if pcd.is_empty():
            raise ValueError("Point cloud doesn't contain points!")
        assert len(min_bound) == 3, "Min bounds for x,y,z required"
        assert len(max_bound) == 3, "Max bounds for x,y,z required"

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        return pcd.crop(bbox)

    @classmethod
    def rotate_pcd(
        cls, pcd: o3d.geometry.PointCloud, angles: Union[Tuple[float, float, float], np.ndarray]
    ) -> o3d.geometry.PointCloud:
        """Rotate point cloud
        Args:
            pcd: Point cloud
            angles: Angles to rotate across x, y, z
        Returns:
            Rotated point cloud
        """
        rot = pcd.get_rotation_matrix_from_xyz(angles)
        rot_pcd = copy.deepcopy(pcd)
        return rot_pcd.rotate(rot)

    @classmethod
    def pcd_to_rgbd(
        cls,
        pcd: o3d.geometry.PointCloud,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        depth_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert point cloud to rgb and depth image
        Args:
            pcd: Point cloud
            intrinsic: Camera intrinsic
            depth_scale: Depth scale
        Returns:
            RGBD imagen from point cloud
        """
        points = np.asarray(pcd.points)

        f_x, f_y = intrinsic.get_focal_length()
        c_x, c_y = intrinsic.get_principal_point()

        rgb = np.zeros((intrinsic.height, intrinsic.width, 3), dtype=np.float64)
        depth = np.zeros((intrinsic.height, intrinsic.width), dtype=np.float64)

        d_vals = points[:, 2] * depth_scale
        u_vals = ((points[:, 0] * f_x) / points[:, 2] + c_x).astype(np.int16)
        v_vals = ((points[:, 1] * f_y) / points[:, 2] + c_y).astype(np.int16)

        depth[v_vals, u_vals] = d_vals
        rgb[v_vals, u_vals, :] = np.asarray(pcd.colors) if pcd.has_colors() else 1.0

        return rgb, depth

    @classmethod
    def remove_hidden_points(
        cls, pcd: o3d.geometry.PointCloud, locs: np.ndarray, radius_constant: int = 100
    ) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Remove hidden points from a pcd given a camera x, y location
        Args:
            pcd: Point cloud
            locs: X, Y location of camera
            radius_constant: Constant for radius of spherical projection
        Returns:
            Front and back point clouds
        """
        x_loc, y_loc = locs
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        _, ind = pcd.hidden_point_removal((x_loc, y_loc, diameter), diameter * radius_constant)

        front = pcd.select_by_index(ind)
        back = pcd.select_by_index(ind, invert=True)

        return front, back
