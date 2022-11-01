"""Unit tests for segmentation module"""

import numpy as np
import pytest

from segmentation.segmentation import Segmentation

def test_segmentation_init() -> None:
    """Test class initilization"""
    seg = Segmentation()
    params = next(seg.model.parameters())
    
    assert not seg.model.training
    assert seg.device == params.device.type

def test_apply_mask_exceptions() -> None:
    """Test if improper mask raises error"""
    height, width = 5, 4
    img = [np.random.randn(height, width, 3)] * 2
    mask = np.random.randn(height, width)
    with pytest.raises(AssertionError) as err:
        Segmentation.apply_masks(img, mask)
    assert str(err.value) == "Masks should be 3 dimensional"

def test_apply_mask_exception_depth() -> None:
    """Test if importper depth raises error"""
    height, width = 5, 4
    img = [np.random.randn(height, width, 3)] * 2
    mask = np.random.randn(height, width, 1)
    depths = [np.random.randn(height, width)]
    with pytest.raises(AssertionError) as err:
        Segmentation.apply_masks(img, mask, depths)
    assert str(err.value) == "Number of rgb and depth images should be same!"
