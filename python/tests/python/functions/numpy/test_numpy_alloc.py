from docc.python import native
import pytest
import numpy as np


def test_numpy_empty():

    @native
    def alloc_empty(n):
        a = np.empty(n, dtype=float)
        a[0] = 1.0
        return a

    result = alloc_empty(10)
    assert result.shape == (10,)
    assert result.strides == (8,)
    assert result[0] == 1.0

    @native
    def alloc_empty_c(n, m):
        a = np.empty((n, m), dtype=np.float64, order="C")
        a[0, 0] = 1.0
        return a

    result = alloc_empty_c(10, 10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)  # row stride > col stride for C-order
    assert result[0, 0] == 1.0

    @native
    def alloc_empty_f(n, m):
        a = np.empty((n, m), dtype=np.float64, order="F")
        a[0, 0] = 1.0
        return a

    result = alloc_empty_f(10, 10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)  # Return arrays are always C-order
    assert result[0, 0] == 1.0

    @native
    def alloc_empty_dtype(A, n):
        B = np.empty(n, dtype=A.dtype)
        B[0] = A[0]
        return B

    A = np.array([3.14], dtype=np.float64)
    res = alloc_empty_dtype(A, 10)
    assert res.shape == (10,)
    assert res.strides == (8,)
    assert res[0] == 3.14

    A_int = np.array([42], dtype=np.int64)
    res_int = alloc_empty_dtype(A_int, 10)
    assert res_int.shape == (10,)
    assert res_int.strides == (8,)
    assert res_int[0] == 42


def test_numpy_zeros():
    @native
    def alloc_zeros(n):
        a = np.zeros(n, dtype=float)
        return a

    result = alloc_zeros(10)
    assert result.shape == (10,)
    assert result.strides == (8,)
    assert result[0] == 0.0

    @native
    def alloc_zeros_c(n, m):
        a = np.zeros((n, m), dtype=np.float64, order="C")
        a[1, 2] = 5.0
        return a

    result = alloc_zeros_c(10, 10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)  # row stride > col stride for C-order
    assert result[1, 2] == 5.0

    @native
    def alloc_zeros_f(n, m):
        a = np.zeros((n, m), dtype=np.float64, order="F")
        a[1, 2] = 5.0
        return a

    result = alloc_zeros_f(10, 10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)  # Return arrays are always C-order
    assert result[1, 2] == 5.0


def test_numpy_eye():

    @native
    def alloc_eye(n):
        a = np.eye(n, dtype=float)
        return a

    result = alloc_eye(10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.0
    assert result[1, 1] == 1.0

    @native
    def alloc_eye_k(n):
        a = np.eye(n, k=1, dtype=float)
        return a

    result = alloc_eye_k(10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)
    assert result[0, 0] == 0.0
    assert result[0, 1] == 1.0
    assert result[1, 2] == 1.0

    @native
    def alloc_eye_rect(n):
        a = np.eye(n, M=n + 2, dtype=float)
        return a

    result = alloc_eye_rect(10)
    assert result.shape == (10, 12)
    assert result.strides == (96, 8)
    assert result[0, 0] == 1.0
    assert result[9, 9] == 1.0

    @native
    def alloc_eye_none(n):
        a = np.eye(n, M=None, dtype=float)
        return a

    result = alloc_eye_none(10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)
    assert result[0, 0] == 1.0


def test_numpy_ones():
    @native
    def alloc_ones(n):
        a = np.ones(n, dtype=float)
        return a

    result = alloc_ones(10)
    assert result.shape == (10,)
    assert result.strides == (8,)
    assert np.all(result == 1.0)

    @native
    def alloc_ones_int(n):
        a = np.ones(n, dtype=int)
        return a

    result = alloc_ones_int(10)
    assert result.shape == (10,)
    assert result.strides == (8,)
    assert np.all(result == 1)

    @native
    def alloc_ones_c(n, m):
        a = np.ones((n, m), dtype=np.float64, order="C")
        return a

    result = alloc_ones_c(10, 10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)
    assert np.all(result == 1.0)

    @native
    def alloc_ones_f(n, m):
        a = np.ones((n, m), dtype=np.float64, order="F")
        return a

    result = alloc_ones_f(10, 10)
    assert result.shape == (10, 10)
    assert result.strides == (80, 8)  # Return arrays are always C-order
    assert np.all(result == 1.0)


def test_zeros_like():
    @native
    def zeros_like_test(a):
        return np.zeros_like(a)

    a = np.random.rand(10, 10)
    res = zeros_like_test(a)
    assert res.shape == (10, 10)
    assert res.strides == (80, 8)
    assert np.all(res == 0)

    @native
    def zeros_like_dtype(a):
        return np.zeros_like(a, dtype=np.float64)

    a = np.random.rand(5, 5).astype(np.float32)
    res = zeros_like_dtype(a)
    assert res.shape == (5, 5)
    assert res.strides == (40, 8)
    assert res.dtype == np.float64
    assert np.all(res == 0)

    @native
    def zeros_like_c(a):
        return np.zeros_like(a, order="C")

    a = np.random.rand(10, 10)
    res = zeros_like_c(a)
    assert res.shape == (10, 10)
    assert res.strides == (80, 8)
    assert np.all(res == 0)

    @native
    def zeros_like_f(a):
        return np.zeros_like(a, order="F")

    a = np.random.rand(10, 10)
    res = zeros_like_f(a)
    assert res.shape == (10, 10)
    assert res.strides == (80, 8)  # Return arrays are always C-order
    assert np.all(res == 0)


def test_numpy_ndarray():

    @native
    def alloc_ndarray(n):
        a = np.ndarray((n,), dtype=np.float64)
        a[0] = 1.0
        return a

    result = alloc_ndarray(10)
    assert result.shape == (10,)
    assert result.strides == (8,)
    assert result[0] == 1.0

    @native
    def alloc_ndarray_c(n, m):
        a = np.ndarray((n, m), dtype=np.float64, order="C")
        for i in range(n):
            for j in range(m):
                a[i, j] = i * m + j
        return a

    result = alloc_ndarray_c(4, 5)
    assert result.shape == (4, 5)
    assert result.strides == (40, 8)  # row stride > col stride for C-order
    for i in range(4):
        for j in range(5):
            assert result[i, j] == i * 5 + j

    @native
    def alloc_ndarray_f(n, m):
        a = np.ndarray((n, m), dtype=np.float64, order="F")
        for i in range(n):
            for j in range(m):
                a[i, j] = i * m + j
        return a

    result = alloc_ndarray_f(4, 5)
    assert result.shape == (4, 5)
    assert result.strides == (40, 8)  # Return arrays are always C-order
    for i in range(4):
        for j in range(5):
            assert result[i, j] == i * 5 + j

    @native
    def alloc_with_custom_strides(n, m):
        # Create array with explicit F-order strides (1, n) in element units
        # For float64, byte strides would be (8, 8*n)
        a = np.ndarray((n, m), dtype=np.float64, strides=(8, 8 * n))
        for i in range(n):
            for j in range(m):
                a[i, j] = i * m + j
        return a

    result = alloc_with_custom_strides(3, 4)
    assert result.shape == (3, 4)
    assert result.strides == (32, 8)  # Return arrays are always C-order
    for i in range(3):
        for j in range(4):
            assert result[i, j] == i * 4 + j


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


def test_numpy_empty_like():
    """Test np.empty_like() allocation"""

    # Test 1D empty_like
    @native
    def empty_like_1d(a):
        b = np.empty_like(a)
        b[0] = 42.0
        return b

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = empty_like_1d(a)
    assert result.shape == (5,)
    assert result.strides == (8,)
    assert result.dtype == np.float64
    assert result[0] == 42.0

    # Test 2D C-order empty_like
    @native
    def empty_like_2d_c(a):
        b = np.empty_like(a)
        b[0, 0] = 7.0
        return b

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64, order="C")
    result = empty_like_2d_c(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # C-order strides
    assert result.dtype == np.float64
    assert result[0, 0] == 7.0

    # Test empty_like with different dtype
    @native
    def empty_like_dtype(a):
        b = np.empty_like(a, dtype=np.int32)
        b[0] = 123
        return b

    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = empty_like_dtype(a)
    assert result.shape == (3,)
    assert result.strides == (4,)
    assert result.dtype == np.int32
    assert result[0] == 123

    # Test empty_like with F-order - output is C-order
    @native
    def empty_like_f_order(a):
        b = np.empty_like(a, order="F")
        b[0, 0] = 99.0
        return b

    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    result = empty_like_f_order(a)
    assert result.shape == (2, 3)
    assert result.strides == (24, 8)  # Return arrays are always C-order
    assert result.dtype == np.float64
    assert result[0, 0] == 99.0
