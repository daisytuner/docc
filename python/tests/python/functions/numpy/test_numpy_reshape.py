from docc.python import native
import numpy as np
import pytest


def test_transpose_T():
    """Test .T attribute transpose with value checks.

    Note: Return arrays are always contiguous C-order copies,
    so strides reflect the output shape, not the view.
    """

    # Test 2D C-order .T
    @native
    def transpose_T_2d_c(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = transpose_T_2d_c(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (16, 8)  # Contiguous C-order for (3, 2) float64
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order .T
    @native
    def transpose_T_2d_f(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = transpose_T_2d_f(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (16, 8)  # Contiguous C-order for (3, 2) float64
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer array .T
    @native
    def transpose_T_int(a):
        return a.T

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    result = transpose_T_int(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (16, 8)  # Contiguous C-order for (3, 2) int64
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test square matrix .T
    @native
    def transpose_T_square(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
    result = transpose_T_square(a)
    expected = a.T
    assert result.shape == (3, 3)
    assert result.strides == (24, 8)  # Contiguous C-order for (3, 3) float64
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    @native
    def transpose_float32(a):
        return a.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    result = transpose_float32(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (8, 4)  # Contiguous C-order for (3, 2) float32
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)

    @native
    def transpose_int32(a):
        return a.T

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    result = transpose_int32(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.strides == (8, 4)  # Contiguous C-order for (3, 2) int32
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)


def test_transpose_func():
    """Test np.transpose() function with value checks.

    Note: Return arrays are always contiguous C-order copies.
    """

    # Test 2D C-order transpose
    @native
    def transpose_func_2d_c(a):
        return np.transpose(a)

    a = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64, order="C"
    )
    result = transpose_func_2d_c(a)
    expected = np.transpose(a)
    assert result.shape == (4, 2)
    assert result.strides == (16, 8)  # Contiguous C-order for (4, 2) float64
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D F-order transpose
    @native
    def transpose_func_2d_f(a):
        return np.transpose(a)

    a = np.array(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64, order="F"
    )
    result = transpose_func_2d_f(a)
    expected = np.transpose(a)
    assert result.shape == (4, 2)
    assert result.strides == (16, 8)  # Contiguous C-order for (4, 2) float64
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test integer transpose
    @native
    def transpose_func_int(a):
        return np.transpose(a)

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    result = transpose_func_int(a)
    expected = np.transpose(a)
    assert result.shape == (3, 3)
    assert result.strides == (24, 8)  # Contiguous C-order for (3, 3) int64
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)


def test_transpose_axes():
    """Test np.transpose() with explicit axes parameter.

    Note: Return arrays are always contiguous C-order copies.
    """

    # Test 2D transpose with axes=(1, 0) - equivalent to .T
    @native
    def transpose_axes_2d(a):
        return np.transpose(a, axes=(1, 0))

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = transpose_axes_2d(a)
    expected = np.transpose(a, axes=(1, 0))
    assert result.shape == (3, 2)
    assert result.strides == (16, 8)  # Contiguous C-order for (3, 2) float64
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D identity permutation axes=(0, 1)
    @native
    def transpose_axes_identity(a):
        return np.transpose(a, axes=(0, 1))

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = transpose_axes_identity(a)
    expected = np.transpose(a, axes=(0, 1))
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Contiguous C-order for (2, 3) float64
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D transpose with axes=(2, 0, 1)
    @native
    def transpose_axes_3d(a):
        return np.transpose(a, axes=(2, 0, 1))

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = transpose_axes_3d(a)
    expected = np.transpose(a, axes=(2, 0, 1))
    assert result.shape == (4, 2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D transpose with axes=(1, 2, 0)
    @native
    def transpose_axes_3d_alt(a):
        return np.transpose(a, axes=(1, 2, 0))

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = transpose_axes_3d_alt(a)
    expected = np.transpose(a, axes=(1, 2, 0))
    assert result.shape == (3, 4, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_transpose_3d():
    """Test 3D array transpose with shape/stride checks"""

    # Test 3D C-order .T (reverses all axes)
    @native
    def transpose_3d_T(a):
        return a.T

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = transpose_3d_T(a)
    expected = a.T
    assert result.shape == (4, 3, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D F-order .T
    @native
    def transpose_3d_T_f(a):
        return a.T

    a = np.asfortranarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4))
    result = transpose_3d_T_f(a)
    expected = a.T
    assert result.shape == (4, 3, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D transpose preserves values correctly
    @native
    def transpose_3d_check_values(a):
        return np.transpose(a)

    a = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    result = transpose_3d_check_values(a)
    expected = np.transpose(a)
    assert result.shape == (5, 4, 3)
    assert result.dtype == np.float64
    # Check specific values
    assert result[0, 0, 0] == a[0, 0, 0]
    assert result[1, 0, 0] == a[0, 0, 1]
    assert result[0, 1, 0] == a[0, 1, 0]
    assert result[0, 0, 1] == a[1, 0, 0]
    assert np.array_equal(result, expected)


def test_transpose_strided():
    """Test transpose with strided (non-contiguous) input arrays"""

    # Test strided 2D input transpose
    @native
    def transpose_strided_2d(a):
        return a.T

    a_full = np.arange(24, dtype=np.float64).reshape(4, 6)
    a = a_full[::2, ::2]  # Shape (2, 3), non-contiguous
    assert not a.flags["C_CONTIGUOUS"]
    result = transpose_strided_2d(a)
    expected = a.T
    assert result.shape == (3, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test row-sliced input transpose
    @native
    def transpose_row_sliced(a):
        return np.transpose(a)

    a_full = np.arange(24, dtype=np.float64).reshape(6, 4)
    a = a_full[::2, :]  # Every other row
    result = transpose_row_sliced(a)
    expected = np.transpose(a)
    assert result.shape == (4, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test column-sliced input transpose
    @native
    def transpose_col_sliced(a):
        return np.transpose(a)

    a_full = np.arange(24, dtype=np.float64).reshape(4, 6)
    a = a_full[:, ::3]  # Every 3rd column
    result = transpose_col_sliced(a)
    expected = np.transpose(a)
    assert result.shape == (2, 4)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


# =============================================================================
# FLIP TESTS
# =============================================================================


def test_flip_1d():
    """Test np.flip on 1D arrays"""

    @native
    def flip_1d(a):
        return np.flip(a)

    # Test basic 1D flip
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = flip_1d(a)
    expected = np.flip(a)
    assert result.shape == (5,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 1D flip with integers
    @native
    def flip_1d_int(a):
        return np.flip(a)

    a = np.array([10, 20, 30, 40], dtype=np.int64)
    result = flip_1d_int(a)
    expected = np.flip(a)
    assert result.shape == (4,)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test 1D flip single element
    @native
    def flip_1d_single(a):
        return np.flip(a)

    a = np.array([42.0], dtype=np.float64)
    result = flip_1d_single(a)
    expected = np.flip(a)
    assert result.shape == (1,)
    assert np.array_equal(result, expected)


def test_flip_2d():
    """Test np.flip on 2D arrays with different axes"""

    # Test 2D flip all axes (default)
    @native
    def flip_2d_all(a):
        return np.flip(a)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = flip_2d_all(a)
    expected = np.flip(a)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D flip axis=0 (flip rows)
    @native
    def flip_2d_axis0(a):
        return np.flip(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = flip_2d_axis0(a)
    expected = np.flip(a, axis=0)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D flip axis=1 (flip columns)
    @native
    def flip_2d_axis1(a):
        return np.flip(a, axis=1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = flip_2d_axis1(a)
    expected = np.flip(a, axis=1)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D flip with negative axis
    @native
    def flip_2d_neg_axis(a):
        return np.flip(a, axis=-1)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = flip_2d_neg_axis(a)
    expected = np.flip(a, axis=-1)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


def test_flip_2d_orders():
    """Test np.flip on 2D arrays with C and F order"""

    # Test C-order flip
    @native
    def flip_c_order(a):
        return np.flip(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = flip_c_order(a)
    expected = np.flip(a, axis=0)
    assert np.array_equal(result, expected)

    # Test F-order flip
    @native
    def flip_f_order(a):
        return np.flip(a, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="F")
    result = flip_f_order(a)
    expected = np.flip(a, axis=0)
    assert np.array_equal(result, expected)


def test_flip_3d():
    """Test np.flip on 3D arrays"""

    # Test 3D flip axis=0
    @native
    def flip_3d_axis0(a):
        return np.flip(a, axis=0)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = flip_3d_axis0(a)
    expected = np.flip(a, axis=0)
    assert result.shape == (2, 3, 4)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D flip axis=1
    @native
    def flip_3d_axis1(a):
        return np.flip(a, axis=1)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = flip_3d_axis1(a)
    expected = np.flip(a, axis=1)
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, expected)

    # Test 3D flip axis=2
    @native
    def flip_3d_axis2(a):
        return np.flip(a, axis=2)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = flip_3d_axis2(a)
    expected = np.flip(a, axis=2)
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, expected)

    # Test 3D flip axis=-1 (same as axis=2)
    @native
    def flip_3d_neg_axis(a):
        return np.flip(a, axis=-1)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = flip_3d_neg_axis(a)
    expected = np.flip(a, axis=-1)
    assert np.array_equal(result, expected)


def test_fliplr():
    """Test np.fliplr (flip left-right, axis=1)"""

    # Test 2D fliplr
    @native
    def fliplr_2d(a):
        return np.fliplr(a)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = fliplr_2d(a)
    expected = np.fliplr(a)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 3D fliplr
    @native
    def fliplr_3d(a):
        return np.fliplr(a)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = fliplr_3d(a)
    expected = np.fliplr(a)
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, expected)

    # Test fliplr with integer array
    @native
    def fliplr_int(a):
        return np.fliplr(a)

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    result = fliplr_int(a)
    expected = np.fliplr(a)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)


def test_flipud():
    """Test np.flipud (flip up-down, axis=0)"""

    # Test 2D flipud
    @native
    def flipud_2d(a):
        return np.flipud(a)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = flipud_2d(a)
    expected = np.flipud(a)
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 1D flipud (same as flip for 1D)
    @native
    def flipud_1d(a):
        return np.flipud(a)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = flipud_1d(a)
    expected = np.flipud(a)
    assert result.shape == (5,)
    assert np.array_equal(result, expected)

    # Test 3D flipud
    @native
    def flipud_3d(a):
        return np.flipud(a)

    a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    result = flipud_3d(a)
    expected = np.flipud(a)
    assert result.shape == (2, 3, 4)
    assert np.array_equal(result, expected)

    # Test flipud with integer array
    @native
    def flipud_int(a):
        return np.flipud(a)

    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    result = flipud_int(a)
    expected = np.flipud(a)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)


def test_flip_dtypes():
    """Test flip operations with various data types"""

    # Test float32
    @native
    def flip_float32(a):
        return np.flip(a)

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = flip_float32(a)
    expected = np.flip(a)
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)

    # Test int32
    @native
    def flip_int32(a):
        return np.flip(a)

    a = np.array([1, 2, 3, 4], dtype=np.int32)
    result = flip_int32(a)
    expected = np.flip(a)
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)


def test_chained_views():
    """Test chained view operations (flip, transpose, reshape combinations)."""

    # Test flip then add scalar (arithmetic on view)
    @native
    def flip_then_add(a):
        b = np.flip(a)
        return b + 1.0

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = flip_then_add(a)
    expected = np.flip(a) + 1.0
    assert result.shape == (5,)
    assert np.array_equal(result, expected)

    # Test flip then transpose
    @native
    def flip_then_transpose(a):
        b = np.flip(a, axis=1)
        return b.T

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = flip_then_transpose(a)
    expected = np.flip(a, axis=1).T
    assert result.shape == (3, 2)
    assert np.array_equal(result, expected)

    # Test transpose then flip
    @native
    def transpose_then_flip(a):
        b = a.T
        return np.flip(b, axis=0)

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = transpose_then_flip(a)
    expected = np.flip(a.T, axis=0)
    assert result.shape == (3, 2)
    assert np.array_equal(result, expected)

    # Test flip then multiply
    @native
    def flip_then_multiply(a, b):
        c = np.flip(a)
        return c * b

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    result = flip_then_multiply(a, b)
    expected = np.flip(a) * b
    assert np.array_equal(result, expected)

    # Test transpose then arithmetic
    @native
    def transpose_then_add(a, b):
        c = a.T
        return c + b

    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    b = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    result = transpose_then_add(a, b)
    expected = a.T + b
    assert result.shape == (2, 3)
    assert np.array_equal(result, expected)


# =============================================================================
# RESHAPE TESTS
# =============================================================================


def test_reshape_basic():
    """Test basic np.reshape operations"""

    # Test 1D to 2D reshape
    @native
    def reshape_1d_to_2d(a):
        return np.reshape(a, (2, 3))

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    result = reshape_1d_to_2d(a)
    expected = np.reshape(a, (2, 3))
    assert result.shape == (2, 3)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D to 1D reshape (flatten)
    @native
    def reshape_2d_to_1d(a):
        return np.reshape(a, (6,))

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = reshape_2d_to_1d(a)
    expected = np.reshape(a, (6,))
    assert result.shape == (6,)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)

    # Test 2D to different 2D shape
    @native
    def reshape_2d_to_2d(a):
        return np.reshape(a, (3, 2))

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = reshape_2d_to_2d(a)
    expected = np.reshape(a, (3, 2))
    assert result.shape == (3, 2)
    assert result.dtype == np.float64
    assert np.array_equal(result, expected)


# def test_reshape_3d():
#     """Test reshape with 3D arrays"""

#     # Test 1D to 3D
#     @native
#     def reshape_1d_to_3d(a):
#         return np.reshape(a, (2, 3, 4))

#     a = np.arange(24, dtype=np.float64)
#     result = reshape_1d_to_3d(a)
#     expected = np.reshape(a, (2, 3, 4))
#     assert result.shape == (2, 3, 4)
#     assert result.dtype == np.float64
#     assert np.array_equal(result, expected)

#     # Test 3D to 2D
#     @native
#     def reshape_3d_to_2d(a):
#         return np.reshape(a, (6, 4))

#     a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
#     result = reshape_3d_to_2d(a)
#     expected = np.reshape(a, (6, 4))
#     assert result.shape == (6, 4)
#     assert result.dtype == np.float64
#     assert np.array_equal(result, expected)

#     # Test 3D to 1D
#     @native
#     def reshape_3d_to_1d(a):
#         return np.reshape(a, (24,))

#     a = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
#     result = reshape_3d_to_1d(a)
#     expected = np.reshape(a, (24,))
#     assert result.shape == (24,)
#     assert np.array_equal(result, expected)


def test_reshape_orders():
    """Test reshape with different memory orders"""

    # Test reshape C-order input
    @native
    def reshape_c_order(a):
        return np.reshape(a, (3, 2))

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = reshape_c_order(a)
    expected = np.reshape(a, (3, 2))
    assert result.shape == (3, 2)
    assert np.array_equal(result, expected)

    # Note: F-order reshape requires copy semantics which is not yet implemented
    # The compiler creates views with C-order strides, so F-order input
    # reshape is not directly supported as a view operation


def test_reshape_dtypes():
    """Test reshape with various data types"""

    # Test int64
    @native
    def reshape_int64(a):
        return np.reshape(a, (2, 3))

    a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    result = reshape_int64(a)
    expected = np.reshape(a, (2, 3))
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)

    # Test float32
    @native
    def reshape_float32(a):
        return np.reshape(a, (2, 3))

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    result = reshape_float32(a)
    expected = np.reshape(a, (2, 3))
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)

    # Test int32
    @native
    def reshape_int32(a):
        return np.reshape(a, (3, 2))

    a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    result = reshape_int32(a)
    expected = np.reshape(a, (3, 2))
    assert result.dtype == np.int32
    assert np.array_equal(result, expected)


def test_reshape_large():
    """Test reshape with larger arrays"""

    # Test larger array reshape
    @native
    def reshape_large(a):
        return np.reshape(a, (10, 10))

    a = np.arange(100, dtype=np.float64)
    result = reshape_large(a)
    expected = np.reshape(a, (10, 10))
    assert result.shape == (10, 10)
    assert np.array_equal(result, expected)

    # Test 4D reshape
    @native
    def reshape_4d(a):
        return np.reshape(a, (2, 3, 4, 5))

    a = np.arange(120, dtype=np.float64)
    result = reshape_4d(a)
    expected = np.reshape(a, (2, 3, 4, 5))
    assert result.shape == (2, 3, 4, 5)
    assert np.array_equal(result, expected)


def test_view_expressions():

    # Test flip of 1D slice
    @native
    def flip_slice_1d(a):
        return np.flip(a[:5])

    a = np.arange(10, dtype=np.float64)
    result = flip_slice_1d(a)
    expected = np.flip(a[:5])
    assert result.shape == (5,)
    assert np.array_equal(result, expected)

    # Test flip of 2D slice (row slice)
    @native
    def flip_slice_2d_rows(a):
        return np.flip(a[:2, :], axis=0)

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = flip_slice_2d_rows(a)
    expected = np.flip(a[:2, :], axis=0)
    assert result.shape == (2, 4)
    assert np.array_equal(result, expected)

    # Test flip of 2D slice (column slice)
    @native
    def flip_slice_2d_cols(a):
        return np.flip(a[:, 1:3], axis=1)

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    result = flip_slice_2d_cols(a)
    expected = np.flip(a[:, 1:3], axis=1)
    assert result.shape == (3, 2)
    assert np.array_equal(result, expected)

    # Test scalar * flip(slice)
    @native
    def scalar_times_flip(a):
        return 2.0 * np.flip(a[:5])

    a = np.arange(10, dtype=np.float64)
    result = scalar_times_flip(a)
    expected = 2.0 * np.flip(a[:5])
    assert result.shape == (5,)
    assert np.allclose(result, expected)

    @native
    def flip_plus_flip(a):
        return np.flip(a[:5]) + np.flip(a[5:])

    a = np.arange(10, dtype=np.float64)
    result = flip_plus_flip(a)
    expected = np.flip(a[:5]) + np.flip(a[5:])
    assert result.shape == (5,)
    assert np.allclose(result, expected)

    @native
    def flip_times_array(a, b):
        return np.flip(a[:5]) * b

    a = np.arange(10, dtype=np.float64)
    b = np.ones(5, dtype=np.float64) * 3.0
    result = flip_times_array(a, b)
    expected = np.flip(a[:5]) * b
    assert result.shape == (5,)
    assert np.allclose(result, expected)

    @native
    def flip_assign_simple(y):
        y[:5] = 2.0 * np.flip(y[:5])

    a = np.arange(10, dtype=np.float64)
    expected = a.copy()
    expected[:5] = 2.0 * np.flip(expected[:5])
    flip_assign_simple(a)
    assert np.allclose(a, expected)

    @native
    def flip_aug_assign(y):
        y[:5] += 2.0 * np.flip(y[:5])

    a = np.arange(10, dtype=np.float64)
    expected = a.copy()
    expected[:5] += 2.0 * np.flip(expected[:5])
    flip_aug_assign(a)
    assert np.allclose(a, expected)

    @native
    def durbin_pattern(y, k, alpha):
        y[:k] += alpha * np.flip(y[:k])

    for k_val in [2, 5, 8]:
        a = np.arange(10, dtype=np.float64)
        expected = a.copy()
        expected[:k_val] += 0.5 * np.flip(expected[:k_val])
        durbin_pattern(a, k_val, 0.5)
        assert np.allclose(a, expected), f"Failed for k={k_val}"
