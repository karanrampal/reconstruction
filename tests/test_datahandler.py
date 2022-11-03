"""Unit tests for datahandler"""

import json
import os
from typing import Dict, List

import numpy as np
import open3d as o3d
import pytest
from pytest_mock import MockerFixture

from datahandler.datahandler import DataHandler


@pytest.mark.parametrize(
    "root, im_path, calib_path, cams_to_keep",
    [("a", "b", "c", ["d"]), ("a", "b", "c", None)],
)
def test_datahandler_init(
    root: str, im_path: str, calib_path: str, cams_to_keep: List[str]
) -> None:
    """Unit test for datahandler initilization"""
    dh_ = DataHandler(root, im_path, calib_path, cams_to_keep)

    assert dh_.root == root
    assert dh_.img_path == os.path.join(root, im_path)
    assert dh_.calibration_path == os.path.join(root, calib_path)
    assert dh_.cams_to_keep == cams_to_keep


@pytest.mark.parametrize(
    "cams_to_keep, data, expected",
    [
        (["a"], {"a": 1, "b": 2}, {"a": 1}),
        (None, {"a": 1, "b": 2}, {"a": 1, "b": 2}),
    ],
)
def test_load_depth_scales(
    mocker: MockerFixture,
    cams_to_keep: List[str],
    data: Dict[str, int],
    expected: Dict[str, int],
) -> None:
    """Unit test for loading depth scales via datahandler"""
    mocker.patch("os.path.isfile", return_value=True)

    test_data = json.dumps(data)
    mocked_open = mocker.mock_open(read_data=test_data)
    mocker.patch("builtins.open", mocked_open)

    dh_ = DataHandler("root", "im_path", "calib_path", cams_to_keep)
    output = dh_.load_depth_scales()

    assert output == expected


def test_load_depth_scales_no_file(capsys: pytest.CaptureFixture) -> None:
    """Unit test for loading depth scales when no file exists"""
    dh_ = DataHandler("root", "im_path", "calib_path", ["cams_to_keep"])
    output = dh_.load_depth_scales()

    path_ = os.path.join(dh_.calibration_path, "device_depth_scales.json")

    captured = capsys.readouterr()

    assert captured.out == f"No {path_} such file!\n"
    assert not output


def test_load_transformations_no_file(capsys: pytest.CaptureFixture) -> None:
    """Unit test for loading transformations when no file exists"""
    dh_ = DataHandler("root", "im_path", "calib_path", ["cams_to_keep"])
    output = dh_.load_transformations()

    path_ = os.path.join(dh_.calibration_path, "transformations.json")

    captured = capsys.readouterr()

    assert captured.out == f"No {path_} such file!\n"
    assert not output


@pytest.mark.parametrize(
    "cams_to_keep, data, expected",
    [
        (["a"], {"a": {"transformation_matrix": [1, 2]}, "b": {"c": [2]}}, [[1, 2]]),
        (
            None,
            {
                "a": {"transformation_matrix": [1, 2]},
                "b": {"transformation_matrix": [3, 4]},
            },
            [[1, 2], [3, 4]],
        ),
    ],
)
def test_load_transformations(
    mocker: MockerFixture,
    cams_to_keep: List[str],
    data: Dict[str, Dict[str, List[int]]],
    expected: List[int],
) -> None:
    """Unit test for loading transformations"""
    mocker.patch("os.path.isfile", return_value=True)

    test_data = json.dumps(data)
    mocked_open = mocker.mock_open(read_data=test_data)
    mocker.patch("builtins.open", mocked_open)

    dh_ = DataHandler("root", "im_path", "calib_path", cams_to_keep)
    output = dh_.load_transformations()
    out_val = list(output.values())

    assert np.array_equal(out_val, expected)


def test_load_intrinsics_no_file(mocker: MockerFixture, capsys: pytest.CaptureFixture) -> None:
    """Unit test for loading intrinsics when no file exist in the directory"""
    mocker.patch("os.listdir", return_value=[])

    dh_ = DataHandler("root", "im_path", "calib_path", ["cams_to_keep"])
    output = dh_.load_intrinsics()

    captured = capsys.readouterr()

    assert captured.out == f"No intrinsic files in {dh_.calibration_path}!\n"
    assert not output


@pytest.mark.parametrize(
    "files_, cams_to_keep, data, expect",
    [
        (
            "123_intrinsics.json",
            "123",
            {"width": 1, "height": 2, "fx": 3, "fy": 4, "ppx": 5, "ppy": 6},
            {"width": 1, "height": 2, "fx": 3, "fy": 4, "cx": 5, "cy": 6},
        )
    ],
)
def test_load_intrinsics(
    mocker: MockerFixture,
    files_: str,
    cams_to_keep: str,
    data: Dict[str, int],
    expect: Dict[str, int],
) -> None:
    """Unit test for loading intrinsics"""
    mocker.patch("os.listdir", return_value=[files_])

    test_data = json.dumps(data)
    mocked_open = mocker.mock_open(read_data=test_data)
    mocker.patch("builtins.open", mocked_open)

    dh_ = DataHandler("root", "im_path", "calib_path", [cams_to_keep])
    output = dh_.load_intrinsics()

    expected = o3d.camera.PinholeCameraIntrinsic(**expect)

    assert np.array_equal(output[cams_to_keep].intrinsic_matrix, expected.intrinsic_matrix)
