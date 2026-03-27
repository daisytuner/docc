from docc.python import native
import pytest
import numpy as np


def test_numpy_astype():
    """Test array.astype() dtype conversion"""

    @native
    def astype_float64_to_int64(A):
        return A.astype(np.int64)

    A = np.array([1.1, 2.9, 3.5, 4.2, 5.8], dtype=np.float64)
    result = astype_float64_to_int64(A)
    expected = A.astype(np.int64)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    @native
    def astype_int64_to_float64(A):
        return A.astype(np.float64)

    A = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = astype_int64_to_float64(A)
    expected = A.astype(np.float64)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def astype_float64_to_float32(A):
        return A.astype(np.float32)

    A = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64)
    result = astype_float64_to_float32(A)
    expected = A.astype(np.float32)
    assert result.shape == (5,)
    assert result.strides == (4,)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)

    @native
    def astype_2d(A):
        return A.astype(np.int32)

    A = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float64)
    result = astype_2d(A)
    expected = A.astype(np.int32)
    assert result.shape == (3, 2)
    assert result.strides == (8, 4)
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)

    @native
    def astype_int32_to_int64(A):
        return A.astype(np.int64)

    A = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = astype_int32_to_int64(A)
    expected = A.astype(np.int64)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test F-order array - output is C-order
    @native
    def astype_f_order(A):
        return A.astype(np.int32)

    A = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], dtype=np.float64, order="F")
    result = astype_f_order(A)
    expected = A.astype(np.int32)
    assert result.shape == (2, 3)
    assert result.strides == (12, 4)  # Return arrays are always C-order
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)

    # Test strided array input (e.g., sliced array with non-contiguous strides)
    @native
    def astype_strided(A):
        return A.astype(np.int32)

    # Create a strided view: every other row of a larger array
    base = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    A = base[::2, :]  # rows 0 and 2, strides are (32, 8) instead of (16, 8)
    assert A.strides == (32, 8)  # Verify non-contiguous input
    result = astype_strided(A)
    expected = A.astype(np.int32)
    assert result.shape == (2, 2)
    assert result.strides == (8, 4)  # Output should be contiguous with scaled strides
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)

    # Test that copy=False raises an error
    @native
    def astype_copy_false(A):
        return A.astype(np.int64, copy=False)

    A = np.array([1, 2, 3], dtype=np.int32)
    with pytest.raises(NotImplementedError):
        astype_copy_false(A)
