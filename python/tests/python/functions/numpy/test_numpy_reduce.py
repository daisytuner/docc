import pytest
import numpy as np
from docc.python import native


def test_sum_1d():
    """Test np.sum on 1D arrays with shape/stride checks"""

    @native
    def sum_1d_all(a) -> float:
        return np.sum(a)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = sum_1d_all(a)
    expected = np.sum(a)
    assert abs(result - expected) < 1e-10

    @native
    def sum_1d_axis0(a):
        return np.sum(a, axis=0)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = sum_1d_axis0(a)
    expected = np.sum(a, axis=0)
    assert abs(result - expected) < 1e-10

    @native
    def sum_1d_keepdims(a):
        return np.sum(a, axis=0, keepdims=True)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = sum_1d_keepdims(a)
    expected = np.sum(a, axis=0, keepdims=True)
    assert result.shape == (1,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_sum_2d_c_order():
    """Test np.sum on 2D C-order arrays with various axes"""

    # Test axis=0 (sum along rows)
    @native
    def sum_2d_axis0(a):
        return np.sum(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = sum_2d_axis0(a)
    expected = np.sum(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test axis=1 (sum along columns)
    @native
    def sum_2d_axis1(a):
        return np.sum(a, axis=1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = sum_2d_axis1(a)
    expected = np.sum(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test axis=-1 (last axis)
    @native
    def sum_2d_axis_neg1(a):
        return np.sum(a, axis=-1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = sum_2d_axis_neg1(a)
    expected = np.sum(a, axis=-1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test full reduction (axis=None)
    @native
    def sum_2d_all(a) -> float:
        return np.sum(a)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = sum_2d_all(a)
    expected = np.sum(a)
    assert abs(result - expected) < 1e-10


def test_sum_2d_f_order():
    """Test np.sum on 2D F-order arrays with various axes"""

    # Test axis=0 (sum along rows)
    @native
    def sum_2d_f_axis0(a):
        return np.sum(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = sum_2d_f_axis0(a)
    expected = np.sum(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test axis=1 (sum along columns)
    @native
    def sum_2d_f_axis1(a):
        return np.sum(a, axis=1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = sum_2d_f_axis1(a)
    expected = np.sum(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_sum_2d_keepdims():
    """Test np.sum with keepdims=True"""

    @native
    def sum_keepdims_axis0(a):
        return np.sum(a, axis=0, keepdims=True)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = sum_keepdims_axis0(a)
    expected = np.sum(a, axis=0, keepdims=True)
    assert result.shape == (1, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def sum_keepdims_axis1(a):
        return np.sum(a, axis=1, keepdims=True)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = sum_keepdims_axis1(a)
    expected = np.sum(a, axis=1, keepdims=True)
    assert result.shape == (2, 1)
    assert result.strides == (8, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_sum_3d():
    """Test np.sum on 3D arrays with various axes"""

    @native
    def sum_3d_axis0(a):
        return np.sum(a, axis=0)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = sum_3d_axis0(a)
    expected = np.sum(a, axis=0)
    assert result.shape == (3, 4)
    assert result.strides == (32, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def sum_3d_axis1(a):
        return np.sum(a, axis=1)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = sum_3d_axis1(a)
    expected = np.sum(a, axis=1)
    assert result.shape == (2, 4)
    assert result.strides == (32, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def sum_3d_axis2(a):
        return np.sum(a, axis=2)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = sum_3d_axis2(a)
    expected = np.sum(a, axis=2)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def sum_3d_all(a) -> float:
        return np.sum(a)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = sum_3d_all(a)
    expected = np.sum(a)
    assert abs(result - expected) < 1e-10


def test_sum_dtypes():
    """Test np.sum with different dtypes"""

    @native
    def sum_int64(a):
        return np.sum(a, axis=0)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    result = sum_int64(a)
    expected = np.sum(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    @native
    def sum_float32(a):
        return np.sum(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = sum_float32(a)
    expected = np.sum(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (4,)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)

    @native
    def sum_int32(a):
        return np.sum(a, axis=0)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    result = sum_int32(a)
    expected = np.sum(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (4,)
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)


def test_mean_1d():
    """Test np.mean on 1D arrays"""

    @native
    def mean_1d_all(a) -> float:
        return np.mean(a)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = mean_1d_all(a)
    expected = np.mean(a)
    assert abs(result - expected) < 1e-10

    @native
    def mean_1d_axis0(a):
        return np.mean(a, axis=0)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = mean_1d_axis0(a)
    expected = np.mean(a, axis=0)
    assert abs(result - expected) < 1e-10


def test_mean_2d():
    """Test np.mean on 2D arrays with various axes"""

    @native
    def mean_2d_axis0(a):
        return np.mean(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = mean_2d_axis0(a)
    expected = np.mean(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def mean_2d_axis1(a):
        return np.mean(a, axis=1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = mean_2d_axis1(a)
    expected = np.mean(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def mean_2d_all(a) -> float:
        return np.mean(a)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = mean_2d_all(a)
    expected = np.mean(a)
    assert abs(result - expected) < 1e-10


def test_mean_2d_f_order():
    """Test np.mean on 2D F-order arrays"""

    @native
    def mean_2d_f_axis0(a):
        return np.mean(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = mean_2d_f_axis0(a)
    expected = np.mean(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def mean_2d_f_axis1(a):
        return np.mean(a, axis=1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = mean_2d_f_axis1(a)
    expected = np.mean(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_mean_keepdims():
    """Test np.mean with keepdims=True"""

    @native
    def mean_keepdims_axis0(a):
        return np.mean(a, axis=0, keepdims=True)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = mean_keepdims_axis0(a)
    expected = np.mean(a, axis=0, keepdims=True)
    assert result.shape == (1, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def mean_keepdims_axis1(a):
        return np.mean(a, axis=1, keepdims=True)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = mean_keepdims_axis1(a)
    expected = np.mean(a, axis=1, keepdims=True)
    assert result.shape == (2, 1)
    assert result.strides == (8, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_mean_3d():
    """Test np.mean on 3D arrays with various axes"""

    @native
    def mean_3d_axis0(a):
        return np.mean(a, axis=0)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = mean_3d_axis0(a)
    expected = np.mean(a, axis=0)
    assert result.shape == (3, 4)
    assert result.strides == (32, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def mean_3d_axis2(a):
        return np.mean(a, axis=2)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = mean_3d_axis2(a)
    expected = np.mean(a, axis=2)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_std_1d():
    """Test np.std on 1D arrays"""

    @native
    def std_1d_all(a) -> float:
        return np.std(a)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = std_1d_all(a)
    expected = np.std(a)
    assert abs(result - expected) < 1e-10


def test_std_2d():
    """Test np.std on 2D arrays with various axes"""

    @native
    def std_2d_axis0(a):
        return np.std(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = std_2d_axis0(a)
    expected = np.std(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def std_2d_axis1(a):
        return np.std(a, axis=1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = std_2d_axis1(a)
    expected = np.std(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def std_2d_all(a) -> float:
        return np.std(a)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = std_2d_all(a)
    expected = np.std(a)
    assert abs(result - expected) < 1e-10


def test_std_2d_f_order():
    """Test np.std on 2D F-order arrays"""

    @native
    def std_2d_f_axis0(a):
        return np.std(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = std_2d_f_axis0(a)
    expected = np.std(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_std_keepdims():
    """Test np.std with keepdims=True"""

    @native
    def std_keepdims_axis0(a):
        return np.std(a, axis=0, keepdims=True)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = std_keepdims_axis0(a)
    expected = np.std(a, axis=0, keepdims=True)
    assert result.shape == (1, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def std_keepdims_axis1(a):
        return np.std(a, axis=1, keepdims=True)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = std_keepdims_axis1(a)
    expected = np.std(a, axis=1, keepdims=True)
    assert result.shape == (2, 1)
    assert result.strides == (8, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_std_3d():
    """Test np.std on 3D arrays"""

    @native
    def std_3d_axis0(a):
        return np.std(a, axis=0)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = std_3d_axis0(a)
    expected = np.std(a, axis=0)
    assert result.shape == (3, 4)
    assert result.strides == (32, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def std_3d_axis2(a):
        return np.std(a, axis=2)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = std_3d_axis2(a)
    expected = np.std(a, axis=2)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_max_1d():
    """Test np.max on 1D arrays"""

    @native
    def max_1d_all(a) -> float:
        return np.max(a)

    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
    result = max_1d_all(a)
    expected = np.max(a)
    assert abs(result - expected) < 1e-10

    @native
    def max_1d_axis0(a):
        return np.max(a, axis=0)

    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
    result = max_1d_axis0(a)
    expected = np.max(a, axis=0)
    assert abs(result - expected) < 1e-10


def test_max_2d():
    """Test np.max on 2D arrays with various axes"""

    @native
    def max_2d_axis0(a):
        return np.max(a, axis=0)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = max_2d_axis0(a)
    expected = np.max(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def max_2d_axis1(a):
        return np.max(a, axis=1)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = max_2d_axis1(a)
    expected = np.max(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def max_2d_all(a) -> float:
        return np.max(a)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = max_2d_all(a)
    expected = np.max(a)
    assert abs(result - expected) < 1e-10


def test_max_2d_f_order():
    """Test np.max on 2D F-order arrays"""

    @native
    def max_2d_f_axis0(a):
        return np.max(a, axis=0)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64, order="F")
    result = max_2d_f_axis0(a)
    expected = np.max(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def max_2d_f_axis1(a):
        return np.max(a, axis=1)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64, order="F")
    result = max_2d_f_axis1(a)
    expected = np.max(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_max_keepdims():
    """Test np.max with keepdims=True"""

    @native
    def max_keepdims_axis0(a):
        return np.max(a, axis=0, keepdims=True)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = max_keepdims_axis0(a)
    expected = np.max(a, axis=0, keepdims=True)
    assert result.shape == (1, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def max_keepdims_axis1(a):
        return np.max(a, axis=1, keepdims=True)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = max_keepdims_axis1(a)
    expected = np.max(a, axis=1, keepdims=True)
    assert result.shape == (2, 1)
    assert result.strides == (8, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_max_3d():
    """Test np.max on 3D arrays"""

    @native
    def max_3d_axis0(a):
        return np.max(a, axis=0)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = max_3d_axis0(a)
    expected = np.max(a, axis=0)
    assert result.shape == (3, 4)
    assert result.strides == (32, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def max_3d_axis2(a):
        return np.max(a, axis=2)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = max_3d_axis2(a)
    expected = np.max(a, axis=2)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_max_dtypes():
    """Test np.max with different dtypes"""

    @native
    def max_int64(a):
        return np.max(a, axis=0)

    a = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int64)
    result = max_int64(a)
    expected = np.max(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    @native
    def max_float32(a):
        return np.max(a, axis=0)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float32)
    result = max_float32(a)
    expected = np.max(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (4,)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)


def test_min_1d():
    """Test np.min on 1D arrays"""

    @native
    def min_1d_all(a) -> float:
        return np.min(a)

    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
    result = min_1d_all(a)
    expected = np.min(a)
    assert abs(result - expected) < 1e-10

    @native
    def min_1d_axis0(a):
        return np.min(a, axis=0)

    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0], dtype=np.float64)
    result = min_1d_axis0(a)
    expected = np.min(a, axis=0)
    assert abs(result - expected) < 1e-10


def test_min_2d():
    """Test np.min on 2D arrays with various axes"""

    @native
    def min_2d_axis0(a):
        return np.min(a, axis=0)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = min_2d_axis0(a)
    expected = np.min(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def min_2d_axis1(a):
        return np.min(a, axis=1)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = min_2d_axis1(a)
    expected = np.min(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def min_2d_all(a) -> float:
        return np.min(a)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = min_2d_all(a)
    expected = np.min(a)
    assert abs(result - expected) < 1e-10


def test_min_2d_f_order():
    """Test np.min on 2D F-order arrays"""

    @native
    def min_2d_f_axis0(a):
        return np.min(a, axis=0)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64, order="F")
    result = min_2d_f_axis0(a)
    expected = np.min(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def min_2d_f_axis1(a):
        return np.min(a, axis=1)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64, order="F")
    result = min_2d_f_axis1(a)
    expected = np.min(a, axis=1)
    assert result.shape == (2,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_min_keepdims():
    """Test np.min with keepdims=True"""

    @native
    def min_keepdims_axis0(a):
        return np.min(a, axis=0, keepdims=True)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = min_keepdims_axis0(a)
    expected = np.min(a, axis=0, keepdims=True)
    assert result.shape == (1, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def min_keepdims_axis1(a):
        return np.min(a, axis=1, keepdims=True)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    result = min_keepdims_axis1(a)
    expected = np.min(a, axis=1, keepdims=True)
    assert result.shape == (2, 1)
    assert result.strides == (8, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_min_3d():
    """Test np.min on 3D arrays"""

    @native
    def min_3d_axis0(a):
        return np.min(a, axis=0)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = min_3d_axis0(a)
    expected = np.min(a, axis=0)
    assert result.shape == (3, 4)
    assert result.strides == (32, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    @native
    def min_3d_axis2(a):
        return np.min(a, axis=2)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = min_3d_axis2(a)
    expected = np.min(a, axis=2)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_min_dtypes():
    """Test np.min with different dtypes"""

    @native
    def min_int64(a):
        return np.min(a, axis=0)

    a = np.array([[1, 5, 3], [4, 2, 6]], dtype=np.int64)
    result = min_int64(a)
    expected = np.min(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    @native
    def min_float32(a):
        return np.min(a, axis=0)

    a = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float32)
    result = min_float32(a)
    expected = np.min(a, axis=0)
    assert result.shape == (3,)
    assert result.strides == (4,)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)
