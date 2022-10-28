"""Unit tests for datahandler"""

import json
import os
from typing import Dict, List

import pytest
from pytest_mock import MockerFixture

from reconstruction.datahandler.datahandler import DataHandler


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
def test_datahandler_load_depth_scales(
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


def test_datahandler_load_depth_scales_no_file(mocker: MockerFixture) -> None:
    """Unit test for loading depth scales via datahandler when no file exists"""
    dh_ = DataHandler("root", "im_path", "calib_path", ["cams_to_keep"])
    output = dh_.load_depth_scales()

    assert output == {}
