import pytest
import numpy as np

from docc.python import native


def test_numpy_add():
    """Test elementwise addition with shape/stride preservation"""

    # Test 1D float addition using + operator
    @native
    def add_1d(a, b):
        return a + b

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    b = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    result = add_1d(a, b)
    expected = a + b
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order addition
    @native
    def add_2d_c(a, b):
        return a + b

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    b = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64, order="C")
    result = add_2d_c(a, b)
    expected = a + b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # C-order strides
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order addition - output is contiguous C-order
    @native
    def add_2d_f(a, b):
        return a + b

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    b = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64, order="F")
    result = add_2d_f(a, b)
    expected = a + b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer addition
    @native
    def add_int(a, b):
        return a + b

    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    result = add_int(a, b)
    expected = a + b
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test strided input (non-contiguous slices) produces contiguous output
    @native
    def add_strided(a, b):
        return a + b

    a_full = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    b_full = np.array(
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [70.0, 80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]  # shape (2, 3), strides (48, 16) - non-contiguous
    b = b_full[:, ::2]
    result = add_strided(a, b)
    expected = a + b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_numpy_sub():
    """Test elementwise subtraction with shape/stride preservation"""

    # Test 1D float subtraction using np.subtract
    @native
    def sub_1d(a, b):
        return np.subtract(a, b)

    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = sub_1d(a, b)
    expected = np.subtract(a, b)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order subtraction using - operator
    @native
    def sub_2d_c(a, b):
        return a - b

    a = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64, order="C")
    b = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = sub_2d_c(a, b)
    expected = a - b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer subtraction
    @native
    def sub_int(a, b):
        return a - b

    a = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    b = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = sub_int(a, b)
    expected = a - b
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def sub_strided(a, b):
        return a - b

    a_full = np.array(
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [70.0, 80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=np.float64,
        order="C",
    )
    b_full = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    b = b_full[:, ::2]
    result = sub_strided(a, b)
    expected = a - b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_numpy_mul():
    """Test elementwise multiplication with shape/stride preservation"""

    # Test 1D float multiplication using np.multiply
    @native
    def mul_1d(a, b):
        return np.multiply(a, b)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    result = mul_1d(a, b)
    expected = np.multiply(a, b)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order multiplication using * operator
    @native
    def mul_2d_c(a, b):
        return a * b

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    b = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float64, order="C")
    result = mul_2d_c(a, b)
    expected = a * b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order multiplication - output is contiguous C-order
    @native
    def mul_2d_f(a, b):
        return a * b

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    b = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float64, order="F")
    result = mul_2d_f(a, b)
    expected = a * b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer multiplication
    @native
    def mul_int(a, b):
        return a * b

    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    result = mul_int(a, b)
    expected = a * b
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def mul_strided(a, b):
        return a * b

    a_full = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    b_full = np.array(
        [[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    b = b_full[:, ::2]
    result = mul_strided(a, b)
    expected = a * b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_numpy_mul_scalar():
    """Test array-scalar multiplication"""

    # Test 1D float * scalar
    @native
    def mul_scalar_1d(a):
        return a * 3.0

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = mul_scalar_1d(a)
    expected = a * 3.0
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order * scalar
    @native
    def mul_scalar_2d_c(a):
        return a * 2.0

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = mul_scalar_2d_c(a)
    expected = a * 2.0
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def mul_scalar_strided(a):
        return a * 2.5

    a_full = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    result = mul_scalar_strided(a)
    expected = a * 2.5
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_numpy_div():
    """Test elementwise division with shape/stride preservation"""

    # Test 1D float division using np.divide
    @native
    def div_1d(a, b):
        return np.divide(a, b)

    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
    b = np.array([2.0, 4.0, 5.0, 8.0, 10.0], dtype=np.float64)
    result = div_1d(a, b)
    expected = np.divide(a, b)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test 2D C-order division using / operator
    @native
    def div_2d_c(a, b):
        return a / b

    a = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64, order="C")
    b = np.array([[2.0, 4.0, 5.0], [8.0, 10.0, 12.0]], dtype=np.float64, order="C")
    result = div_2d_c(a, b)
    expected = a / b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)

    # Test integer division
    @native
    def div_int(a, b):
        return a / b

    a = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    b = np.array([2, 4, 5, 8, 10], dtype=np.int64)
    result = div_int(a, b)
    expected = a // b  # Integer division
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def div_strided(a, b):
        return a / b

    a_full = np.array(
        [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [70.0, 80.0, 90.0, 100.0, 110.0, 120.0]],
        dtype=np.float64,
        order="C",
    )
    b_full = np.array(
        [[2.0, 4.0, 5.0, 8.0, 10.0, 12.0], [14.0, 16.0, 18.0, 20.0, 22.0, 24.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    b = b_full[:, ::2]
    result = div_strided(a, b)
    expected = a / b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_numpy_negate():
    """Test unary negation with shape/stride preservation"""

    # Test 1D float negation
    @native
    def negate_1d(a):
        return -a

    a = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float64)
    result = negate_1d(a)
    expected = -a
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D C-order negation
    @native
    def negate_2d_c(a):
        return -a

    a = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64, order="C")
    result = negate_2d_c(a)
    expected = -a
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # C-order strides
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order negation - output is contiguous C-order
    @native
    def negate_2d_f(a):
        return -a

    a = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float64, order="F")
    result = negate_2d_f(a)
    expected = -a
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer negation
    @native
    def negate_int(a):
        return -a

    a = np.array([1, -2, 3, -4, 5], dtype=np.int64)
    result = negate_int(a)
    expected = -a
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test combined negate and mul: -a * b
    @native
    def negate_mul(a, b):
        return -a * b

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    b = np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]], dtype=np.float64, order="F")
    result = negate_mul(a, b)
    expected = -a * b
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test strided input produces contiguous output
    @native
    def negate_strided(a):
        return -a

    a_full = np.array(
        [[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], [-7.0, 8.0, -9.0, 10.0, -11.0, 12.0]],
        dtype=np.float64,
        order="C",
    )
    a = a_full[:, ::2]
    result = negate_strided(a)
    expected = -a
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Output is contiguous C-order
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)
