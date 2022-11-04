"""Unit tests for utility functions"""

from typing import Dict

import numpy as np
import open3d as o3d
import pytest

from utils.utils import create_mesh_tsdf


@pytest.mark.parametrize(
    "img, intrin_d",
    [
        (
            np.eye(4, dtype=np.uint8),
            {"width": 4, "height": 4, "fx": 2.0, "fy": 2.0, "cx": 1.0, "cy": 1.0},
        )
    ],
)
def test_create_mesh_tsdf(img: np.ndarray, intrin_d: Dict[str, int]) -> None:
    """Test if mesh has normals"""
    intrin = o3d.camera.PinholeCameraIntrinsic(**intrin_d)

    rgbs = {"a": np.dstack([img, img, img])}
    depths = {"a": img}
    scales = {"a": 0.1}
    transformations: Dict[str, np.ndarray] = {}
    intrinsics = {"a": intrin}

    mesh = create_mesh_tsdf(rgbs, depths, scales, transformations, intrinsics)

    assert not mesh.has_vertex_normals()
