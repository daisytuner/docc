"""Tests for NumPy broadcasting in elementwise operations."""

import numpy as np
import pytest
from docc.python import native


def test_broadcast_1d_to_2d_add():
    """Test adding 1D bias to 2D matrix with shape/stride checks."""

    @native
    def add_bias(x, b):
        return x + b

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    res = add_bias(x, b)
    expected = x + b
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)  # C-order strides
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_1d_to_2d_sub():
    """Test subtracting 1D array from 2D matrix with shape/stride checks."""

    @native
    def sub_bias(x, b):
        return x - b

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    res = sub_bias(x, b)
    expected = x - b
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_1d_to_2d_mul():
    """Test multiplying 2D matrix by 1D array with shape/stride checks."""

    @native
    def scale(x, s):
        return x * s

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    s = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    res = scale(x, s)
    expected = x * s
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_1d_to_2d_div():
    """Test dividing 2D matrix by 1D array with shape/stride checks."""

    @native
    def normalize(x, d):
        return x / d

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    res = normalize(x, d)
    expected = x / d
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_scalar_to_2d_add():
    """Test broadcasting scalar constant to 2D with shape/stride checks."""

    @native
    def add_scalar(x):
        return x + 1.0

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    res = add_scalar(x)
    expected = x + 1.0
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_scalar_to_2d_mul():
    """Test broadcasting scalar constant to 2D via multiplication."""

    @native
    def scale_scalar(x):
        return x * 2.0

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    res = scale_scalar(x)
    expected = x * 2.0
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_float32():
    """Test broadcasting preserves float32 dtype."""

    @native
    def add_bias_f32(x, b):
        return x + b

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    res = add_bias_f32(x, b)
    expected = x + b
    assert res.shape == (2, 3)
    assert res.strides == (12, 4)  # float32 strides
    assert res.dtype == np.float32
    assert np.allclose(res, expected, rtol=1e-5)


def test_broadcast_int64():
    """Test broadcasting with int64 dtype."""

    @native
    def add_bias_int(x, b):
        return x + b

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    b = np.array([10, 20, 30], dtype=np.int64)
    res = add_bias_int(x, b)
    expected = x + b
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.int64
    assert np.array_equal(res, expected)


def test_broadcast_int32():
    """Test broadcasting with int32 dtype."""

    @native
    def add_bias_int32(x, b):
        return x + b

    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    b = np.array([10, 20, 30], dtype=np.int32)
    res = add_bias_int32(x, b)
    expected = x + b
    assert res.shape == (2, 3)
    assert res.strides == (12, 4)  # int32 strides
    assert res.dtype == np.int32
    assert np.array_equal(res, expected)


def test_broadcast_1d_to_3d():
    """Test broadcasting 1D array to 3D with shape/stride checks."""

    @native
    def add_bias_3d(x, b):
        return x + b

    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    b = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    res = add_bias_3d(x, b)
    expected = x + b
    assert res.shape == (2, 3, 4)
    assert res.strides == (96, 32, 8)  # C-order 3D strides
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_2d_to_3d():
    """Test broadcasting 2D array to 3D with shape/stride checks."""

    @native
    def add_bias_2d_to_3d(x, b):
        return x + b

    x = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    b = np.random.rand(3, 4).astype(np.float64)
    res = add_bias_2d_to_3d(x, b)
    expected = x + b
    assert res.shape == (2, 3, 4)
    assert res.strides == (96, 32, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_f_order_input():
    """Test broadcasting with F-order input array."""

    @native
    def add_bias_f(x, b):
        return x + b

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    b = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    res = add_bias_f(x, b)
    expected = x + b
    assert res.shape == (2, 3)
    # Output is C-order since we create a new array
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_maximum():
    """Test np.maximum with broadcasting and shape/stride checks."""

    @native
    def max_threshold(x, t):
        return np.maximum(x, t)

    x = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    t = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    res = max_threshold(x, t)
    expected = np.maximum(x, t)
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_minimum():
    """Test np.minimum with broadcasting and shape/stride checks."""

    @native
    def min_threshold(x, t):
        return np.minimum(x, t)

    x = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]], dtype=np.float64)
    t = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    res = min_threshold(x, t)
    expected = np.minimum(x, t)
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_gemm_plus_bias():
    """Test GEMM + broadcast bias pattern with shape/stride checks."""

    @native
    def linear_layer(x, w, b):
        return x @ w + b

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    res = linear_layer(x, w, b)
    expected = x @ w + b
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_gemm_plus_bias_float32():
    """Test GEMM + broadcast bias preserves float32."""

    @native
    def linear_layer_f32(x, w, b):
        return x @ w + b

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    res = linear_layer_f32(x, w, b)
    expected = x @ w + b
    assert res.shape == (2, 3)
    assert res.strides == (12, 4)
    assert res.dtype == np.float32
    assert np.allclose(res, expected, rtol=1e-4)


def test_broadcast_mlp_layer():
    """Test MLP layer: matmul + bias + relu with shape/stride checks."""

    @native
    def mlp_layer(x, w, b):
        y = x @ w + b
        return np.maximum(y, 0)

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    w = np.array([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]], dtype=np.float64)
    b = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    res = mlp_layer(x, w, b)
    expected = np.maximum(x @ w + b, 0)
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_two_layer_mlp():
    """Test two-layer MLP with broadcast biases and shape/stride checks."""

    @native
    def two_layer_mlp(x, w1, b1, w2, b2):
        h = np.maximum(x @ w1 + b1, 0)
        return h @ w2 + b2

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    w1 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    b1 = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    w2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    b2 = np.array([0.1, 0.2], dtype=np.float64)

    res = two_layer_mlp(x, w1, b1, w2, b2)
    h = np.maximum(x @ w1 + b1, 0)
    expected = h @ w2 + b2
    assert res.shape == (2, 2)
    assert res.strides == (16, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_clip_pattern():
    """Test clipping pattern with broadcasting and shape/stride checks."""

    @native
    def clip_values(x, low, high):
        return np.minimum(np.maximum(x, low), high)

    x = np.array([[0.1, 0.5, 0.9], [0.2, 0.4, 0.8]], dtype=np.float64)
    low = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    high = np.array([0.7, 0.6, 0.8], dtype=np.float64)
    res = clip_values(x, low, high)
    expected = np.minimum(np.maximum(x, low), high)
    assert res.shape == (2, 3)
    assert res.strides == (24, 8)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)


def test_broadcast_column_vector():
    """Test adding column vector (N, 1) to 2D matrix."""

    @native
    def add_column(x, c):
        return x + c

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    c = np.array([[0.1], [0.2]], dtype=np.float64)
    res = add_column(x, c)
    expected = x + c
    assert res.shape == (2, 3)
    assert res.dtype == np.float64
    assert np.allclose(res, expected)
