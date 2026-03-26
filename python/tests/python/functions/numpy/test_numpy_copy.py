from docc.python import native
import pytest
import numpy as np


def test_numpy_copy():
    """Test array.copy() method"""

    # Test 1D copy with shape/stride/value checks
    @native
    def copy_1d(a):
        return a.copy()

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = copy_1d(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, a)

    # Test 2D C-order copy
    @native
    def copy_2d_c(a):
        return a.copy()

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = copy_2d_c(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # C-order strides
    assert result.dtype == np.float64
    assert np.array_equal(result, a)

    # Test 2D F-order copy - output is C-order
    @native
    def copy_2d_f(a):
        return a.copy()

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = copy_2d_f(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, a)

    # Test integer array copy
    @native
    def copy_int(a):
        return a.copy()

    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = copy_int(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, a)

    # Test strided array copy (non-contiguous input)
    @native
    def copy_strided(a):
        return a.copy()

    base = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    a = base[::2, :]  # rows 0 and 2, strides are (32, 8)
    assert a.strides == (32, 8)  # Verify non-contiguous input
    result = copy_strided(a)
    assert result.shape == (2, 2)
    assert result.strides == (16, 8)  # Output should be contiguous
    assert result.dtype == np.float64
    assert np.array_equal(result, a)

    # Test copy independence - modifying original doesn't affect copy
    @native
    def copy_modify_original(a):
        b = a.copy()
        a[0] = 999.0
        return b

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = copy_modify_original(a)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert np.array_equal(result, expected)

    # Test copy independence - modifying copy doesn't affect original
    @native
    def copy_modify_copy(a):
        b = a.copy()
        b[0] = 999.0
        return a

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = copy_modify_copy(a)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert np.array_equal(result, expected)
