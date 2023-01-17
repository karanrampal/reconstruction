"""Unit tests for point cloud manipulation"""

import open3d as o3d
import pytest

from pointcloud.pointcloud import PointCloudManip


def test_distance_filter() -> None:
    """Test if empty point cloud raises error"""
    pcd = o3d.geometry.PointCloud()
    with pytest.raises(ValueError) as err:
        PointCloudManip.distance_filter(pcd, pcd)
    assert str(err.value) == "Point cloud is empty!"


def test_segment_out_plane() -> None:
    """Test if empty point cloud raises error"""
    pcd = o3d.geometry.PointCloud()
    with pytest.raises(ValueError) as err:
        PointCloudManip.segment_out_plane(pcd)
    assert str(err.value) == "No points in point cloud!"


def test_crop_pcd() -> None:
    """Test if empty point cloud raises error"""
    pcd = o3d.geometry.PointCloud()
    with pytest.raises(ValueError) as err:
        PointCloudManip.crop_pcd(pcd, (1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
    assert str(err.value) == "Point cloud doesn't contain points!"
