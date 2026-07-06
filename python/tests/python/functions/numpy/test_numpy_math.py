from docc.python import native
import pytest
import numpy as np
import math


def test_numpy_pow():
    """Test np.power with shape/stride preservation"""

    # Test 1D float power
    @native
    def pow_1d(a, b):
        return np.power(a, b)

    a = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    b = np.array([3.0, 2.0, 0.5, 1.0], dtype=np.float64)
    result = pow_1d(a, b)
    expected = np.power(a, b)
    assert result.shape == (4,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D C-order power
    @native
    def pow_2d_c(a, b):
        return np.power(a, b)

    a = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float64, order="C")
    b = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float64, order="C")
    result = pow_2d_c(a, b)
    expected = np.power(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # C-order strides
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D F-order power - output is C-order
    @native
    def pow_2d_f(a, b):
        return np.power(a, b)

    a = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float64, order="F")
    b = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], dtype=np.float64, order="F")
    result = pow_2d_f(a, b)
    expected = np.power(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test strided input produces contiguous output
    @native
    def pow_strided(a, b):
        return np.power(a, b)

    a_full = np.array(
        [[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]],
        dtype=np.float64,
        order="C",
    )
    b_full = np.array(
        [[2.0, 2.0, 2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    b = b_full[:, ::2]
    result = pow_strided(a, b)
    expected = np.power(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_numpy_abs():
    """Test np.abs with shape/stride preservation"""

    # Test 1D float abs
    @native
    def abs_1d(a):
        return np.abs(a)

    a = np.array([-5.0, 3.0, -2.0, 0.0, -1.5], dtype=np.float64)
    result = abs_1d(a)
    expected = np.abs(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order abs
    @native
    def abs_2d_c(a):
        return np.abs(a)

    a = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float64, order="C")
    result = abs_2d_c(a)
    expected = np.abs(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order abs - output is C-order
    @native
    def abs_2d_f(a):
        return np.abs(a)

    a = np.array([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0]], dtype=np.float64, order="F")
    result = abs_2d_f(a)
    expected = np.abs(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer abs
    @native
    def abs_int(a):
        return np.abs(a)

    a = np.array([-5, 3, -2, 0, -1], dtype=np.int64)
    result = abs_int(a)
    expected = np.abs(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def abs_strided(a):
        return np.abs(a)

    a_full = np.array(
        [[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], [-7.0, 8.0, -9.0, 10.0, -11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    result = abs_strided(a)
    expected = np.abs(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_numpy_sqrt():
    """Test np.sqrt with shape/stride preservation"""

    # Test 1D sqrt
    @native
    def sqrt_1d(a):
        return np.sqrt(a)

    a = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
    result = sqrt_1d(a)
    expected = np.sqrt(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D C-order sqrt
    @native
    def sqrt_2d_c(a):
        return np.sqrt(a)

    a = np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]], dtype=np.float64, order="C")
    result = sqrt_2d_c(a)
    expected = np.sqrt(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D F-order sqrt - output is C-order
    @native
    def sqrt_2d_f(a):
        return np.sqrt(a)

    a = np.array([[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]], dtype=np.float64, order="F")
    result = sqrt_2d_f(a)
    expected = np.sqrt(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test strided input produces contiguous output
    @native
    def sqrt_strided(a):
        return np.sqrt(a)

    a_full = np.array(
        [[1.0, 4.0, 9.0, 16.0, 25.0, 36.0], [49.0, 64.0, 81.0, 100.0, 121.0, 144.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    result = sqrt_strided(a)
    expected = np.sqrt(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_numpy_tanh():
    """Test np.tanh with shape/stride preservation"""

    # Test 1D tanh
    @native
    def tanh_1d(a):
        return np.tanh(a)

    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    result = tanh_1d(a)
    expected = np.tanh(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D C-order tanh
    @native
    def tanh_2d_c(a):
        return np.tanh(a)

    a = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]], dtype=np.float64, order="C")
    result = tanh_2d_c(a)
    expected = np.tanh(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D F-order tanh - output is C-order
    @native
    def tanh_2d_f(a):
        return np.tanh(a)

    a = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]], dtype=np.float64, order="F")
    result = tanh_2d_f(a)
    expected = np.tanh(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test strided input produces contiguous output
    @native
    def tanh_strided(a):
        return np.tanh(a)

    a_full = np.array(
        [[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    result = tanh_strided(a)
    expected = np.tanh(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_numpy_exp():
    """Test np.exp with shape/stride preservation"""

    # Test 1D exp
    @native
    def exp_1d(a):
        return np.exp(a)

    a = np.array([0.0, 1.0, 2.0, -1.0, -2.0], dtype=np.float64)
    result = exp_1d(a)
    expected = np.exp(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D C-order exp
    @native
    def exp_2d_c(a):
        return np.exp(a)

    a = np.array([[0.0, 1.0, 2.0], [-1.0, -2.0, 0.5]], dtype=np.float64, order="C")
    result = exp_2d_c(a)
    expected = np.exp(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D F-order exp - output is C-order
    @native
    def exp_2d_f(a):
        return np.exp(a)

    a = np.array([[0.0, 1.0, 2.0], [-1.0, -2.0, 0.5]], dtype=np.float64, order="F")
    result = exp_2d_f(a)
    expected = np.exp(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test strided input produces contiguous output
    @native
    def exp_strided(a):
        return np.exp(a)

    a_full = np.array(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    result = exp_strided(a)
    expected = np.exp(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_numpy_minimum():
    """Test np.minimum with shape/stride preservation"""

    # Test 1D minimum
    @native
    def minimum_1d(a, b):
        return np.minimum(a, b)

    a = np.array([1.0, 5.0, 3.0, 7.0, 2.0], dtype=np.float64)
    b = np.array([2.0, 3.0, 4.0, 1.0, 8.0], dtype=np.float64)
    result = minimum_1d(a, b)
    expected = np.minimum(a, b)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order minimum
    @native
    def minimum_2d_c(a, b):
        return np.minimum(a, b)

    a = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]], dtype=np.float64, order="C")
    b = np.array([[2.0, 3.0, 4.0], [1.0, 8.0, 6.0]], dtype=np.float64, order="C")
    result = minimum_2d_c(a, b)
    expected = np.minimum(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order minimum - output is C-order
    @native
    def minimum_2d_f(a, b):
        return np.minimum(a, b)

    a = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]], dtype=np.float64, order="F")
    b = np.array([[2.0, 3.0, 4.0], [1.0, 8.0, 6.0]], dtype=np.float64, order="F")
    result = minimum_2d_f(a, b)
    expected = np.minimum(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer minimum
    @native
    def minimum_int(a, b):
        return np.minimum(a, b)

    a = np.array([1, 5, 3, 7, 2], dtype=np.int64)
    b = np.array([2, 3, 4, 1, 8], dtype=np.int64)
    result = minimum_int(a, b)
    expected = np.minimum(a, b)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def minimum_strided(a, b):
        return np.minimum(a, b)

    a_full = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    b_full = np.array(
        [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [12.0, 11.0, 10.0, 9.0, 8.0, 7.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    b = b_full[:, ::2]
    result = minimum_strided(a, b)
    expected = np.minimum(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_numpy_maximum():
    """Test np.maximum with shape/stride preservation"""

    # Test 1D maximum
    @native
    def maximum_1d(a, b):
        return np.maximum(a, b)

    a = np.array([1.0, 5.0, 3.0, 7.0, 2.0], dtype=np.float64)
    b = np.array([2.0, 3.0, 4.0, 1.0, 8.0], dtype=np.float64)
    result = maximum_1d(a, b)
    expected = np.maximum(a, b)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order maximum
    @native
    def maximum_2d_c(a, b):
        return np.maximum(a, b)

    a = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]], dtype=np.float64, order="C")
    b = np.array([[2.0, 3.0, 4.0], [1.0, 8.0, 6.0]], dtype=np.float64, order="C")
    result = maximum_2d_c(a, b)
    expected = np.maximum(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order maximum - output is C-order
    @native
    def maximum_2d_f(a, b):
        return np.maximum(a, b)

    a = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]], dtype=np.float64, order="F")
    b = np.array([[2.0, 3.0, 4.0], [1.0, 8.0, 6.0]], dtype=np.float64, order="F")
    result = maximum_2d_f(a, b)
    expected = np.maximum(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer maximum
    @native
    def maximum_int(a, b):
        return np.maximum(a, b)

    a = np.array([1, 5, 3, 7, 2], dtype=np.int64)
    b = np.array([2, 3, 4, 1, 8], dtype=np.int64)
    result = maximum_int(a, b)
    expected = np.maximum(a, b)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def maximum_strided(a, b):
        return np.maximum(a, b)

    a_full = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    b_full = np.array(
        [[6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [12.0, 11.0, 10.0, 9.0, 8.0, 7.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    b = b_full[:, ::2]
    result = maximum_strided(a, b)
    expected = np.maximum(a, b)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_numpy_cbrt():
    """Test np.cbrt (cube root), including negatives which x**(1/3) can't do."""

    # 1D cbrt, written into an output array
    @native
    def cbrt_1d(a, out):
        out[:] = np.cbrt(a)

    a = np.array([1.0, 8.0, 27.0, -64.0, 0.0], dtype=np.float64)
    out = np.zeros_like(a)
    cbrt_1d(a.copy(), out)
    assert np.allclose(out, np.cbrt(a))

    # cbrt used within a larger expression
    @native
    def cbrt_expr(a, out):
        out[:] = np.cbrt(a) + 1.0

    a2 = np.array([1.0, 8.0, 27.0], dtype=np.float64)
    out2 = np.zeros_like(a2)
    cbrt_expr(a2.copy(), out2)
    assert np.allclose(out2, np.cbrt(a2) + 1.0)

    # 2D cbrt
    @native
    def cbrt_2d(a, out):
        out[:] = np.cbrt(a)

    a3 = np.arange(1.0, 13.0, dtype=np.float64).reshape(3, 4)
    out3 = np.zeros_like(a3)
    cbrt_2d(a3.copy(), out3)
    assert np.allclose(out3, np.cbrt(a3))

    # Scalar cbrt
    @native
    def cbrt_scalar(a):
        return np.cbrt(a[0])

    result = cbrt_scalar(np.array([27.0], dtype=np.float64))
    assert np.isclose(result, 3.0)
