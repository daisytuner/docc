from docc.python import native
import pytest
import numpy as np
import math


def test_scalar_array_broadcasting_mul():
    """Test scalar * array broadcasting (scalar on left side)."""

    @native
    def scalar_times_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = np.sqrt(x) * arr
        return result[1]

    # sqrt(4.0) * 2.0 = 2.0 * 2.0 = 4.0
    assert abs(scalar_times_array(4.0, 5) - 4.0) < 1e-10


def test_scalar_array_broadcasting_add():
    """Test scalar + array broadcasting (scalar on left side)."""

    @native
    def scalar_plus_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = np.sqrt(x) + arr
        return result[1]

    # sqrt(4.0) + 2.0 = 2.0 + 2.0 = 4.0
    assert abs(scalar_plus_array(4.0, 5) - 4.0) < 1e-10


def test_array_scalar_broadcasting_mul():
    """Test array * scalar broadcasting (scalar on right side)."""

    @native
    def array_times_scalar(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = arr * np.sqrt(x)
        return result[1]

    # 2.0 * sqrt(4.0) = 2.0 * 2.0 = 4.0
    assert abs(array_times_scalar(4.0, 5) - 4.0) < 1e-10


def test_scalar_array_broadcasting_sub():
    """Test scalar - array broadcasting."""

    @native
    def scalar_minus_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.0
        arr[1] = 2.0
        arr[2] = 3.0
        result = np.sqrt(x) - arr
        return result[0]

    # sqrt(9.0) - 1.0 = 3.0 - 1.0 = 2.0
    assert abs(scalar_minus_array(9.0, 5) - 2.0) < 1e-10


def test_scalar_array_broadcasting_div():
    """Test scalar / array broadcasting."""

    @native
    def scalar_div_array(x: float, n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[:] = 1.0
        arr[0] = 2.0
        arr[1] = 4.0
        result = np.sqrt(x) / arr
        return result[1]

    # sqrt(16.0) / 4.0 = 4.0 / 4.0 = 1.0
    assert abs(scalar_div_array(16.0, 5) - 1.0) < 1e-10


def test_2d_1d_broadcasting_inplace_sub():
    """Test 2D -= 1D broadcasting (row-wise subtraction)."""

    @native
    def array_2d_minus_1d(n, m) -> float:
        data = np.zeros((n, m), dtype=float)
        mean = np.zeros(m, dtype=float)
        # Set up data: row i has values i, i, i, ...
        for i in range(n):
            for j in range(m):
                data[i, j] = float(i)
        # mean = [1.0, 1.0, ...]
        for j in range(m):
            mean[j] = 1.0
        # After data -= mean, row i should have values i-1, i-1, ...
        data -= mean
        return data[2, 0]  # Should be 2.0 - 1.0 = 1.0

    result = array_2d_minus_1d(5, 4)
    assert abs(result - 1.0) < 1e-10


def test_int_scalar_times_float_array():
    """Test integer scalar * float array type promotion (int -> float)."""

    @native
    def int_scalar_times_float_array(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 2.5
        arr[1] = 3.5
        c = 2  # integer scalar
        result = c * arr
        return result[0]

    # 2 * 2.5 = 5.0
    assert abs(int_scalar_times_float_array(5) - 5.0) < 1e-10


def test_float_array_times_int_scalar():
    """Test float array * integer scalar type promotion (int -> float)."""

    @native
    def float_array_times_int_scalar(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 2.5
        arr[1] = 3.5
        c = 3  # integer scalar
        result = arr * c
        return result[1]

    # 3.5 * 3 = 10.5
    assert abs(float_array_times_int_scalar(5) - 10.5) < 1e-10


def test_int_scalar_plus_float_array():
    """Test integer scalar + float array type promotion."""

    @native
    def int_scalar_plus_float_array(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 1.5
        arr[1] = 2.5
        c = 10  # integer scalar
        result = c + arr
        return result[1]

    # 10 + 2.5 = 12.5
    assert abs(int_scalar_plus_float_array(5) - 12.5) < 1e-10


def test_float_array_minus_int_scalar():
    """Test float array - integer scalar type promotion."""

    @native
    def float_array_minus_int_scalar(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 5.5
        arr[1] = 10.5
        c = 3  # integer scalar
        result = arr - c
        return result[0]

    # 5.5 - 3 = 2.5
    assert abs(float_array_minus_int_scalar(5) - 2.5) < 1e-10


def test_int_scalar_div_float_array():
    """Test integer scalar / float array type promotion."""

    @native
    def int_scalar_div_float_array(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[:] = 1.0
        arr[0] = 2.0
        arr[1] = 4.0
        c = 8  # integer scalar
        result = c / arr
        return result[1]

    # 8 / 4.0 = 2.0
    assert abs(int_scalar_div_float_array(5) - 2.0) < 1e-10


def test_int_var_times_float_array_sum():
    """Test integer variable * (float array + float array) - mimics deriche pattern."""

    @native
    def int_var_times_array_sum(n) -> float:
        y1 = np.zeros(n, dtype=float)
        y2 = np.zeros(n, dtype=float)
        y1[0] = 1.5
        y1[1] = 2.5
        y2[0] = 0.5
        y2[1] = 1.5
        c1 = 1  # integer, like in deriche: c1 = c2 = 1
        result = c1 * (y1 + y2)
        return result[1]

    # 1 * (2.5 + 1.5) = 4.0
    assert abs(int_var_times_array_sum(5) - 4.0) < 1e-10


def test_chained_int_float_operations():
    """Test chained operations with mixed int/float types."""

    @native
    def chained_int_float_ops(n) -> float:
        arr = np.zeros(n, dtype=float)
        arr[0] = 2.0
        arr[1] = 3.0
        a = 2  # int
        b = 3  # int
        # (2 * arr + 3) should promote correctly
        result = a * arr + b
        return result[1]

    # 2 * 3.0 + 3 = 9.0
    assert abs(chained_int_float_ops(5) - 9.0) < 1e-10


def test_numpy_clip_float():
    """Test np.clip with float arrays."""

    @native
    def numpy_clip_float(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = -5.0
        a[1] = 0.5
        a[2] = 5.0
        a[3] = 15.0
        result = np.clip(a, 0.0, 10.0)
        return result[0]

    # -5.0 clipped to [0, 10] -> 0.0
    assert abs(numpy_clip_float(10) - 0.0) < 1e-10

    @native
    def numpy_clip_float_mid(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = -5.0
        a[1] = 0.5
        a[2] = 5.0
        a[3] = 15.0
        result = np.clip(a, 0.0, 10.0)
        return result[1]

    # 0.5 clipped to [0, 10] -> 0.5 (unchanged)
    assert abs(numpy_clip_float_mid(10) - 0.5) < 1e-10

    @native
    def numpy_clip_float_upper(n) -> float:
        a = np.zeros(n, dtype=float)
        a[0] = -5.0
        a[1] = 0.5
        a[2] = 5.0
        a[3] = 15.0
        result = np.clip(a, 0.0, 10.0)
        return result[3]

    # 15.0 clipped to [0, 10] -> 10.0
    assert abs(numpy_clip_float_upper(10) - 10.0) < 1e-10


def test_numpy_clip_int():
    """Test np.clip with integer arrays."""

    @native
    def numpy_clip_int_lower(n) -> int:
        a = np.zeros(n, dtype=int)
        a[0] = -5
        a[1] = 5
        a[2] = 15
        result = np.clip(a, 2, 10)
        return result[0]

    # -5 clipped to [2, 10] -> 2
    assert numpy_clip_int_lower(10) == 2

    @native
    def numpy_clip_int_mid(n) -> int:
        a = np.zeros(n, dtype=int)
        a[0] = -5
        a[1] = 5
        a[2] = 15
        result = np.clip(a, 2, 10)
        return result[1]

    # 5 clipped to [2, 10] -> 5 (unchanged)
    assert numpy_clip_int_mid(10) == 5

    @native
    def numpy_clip_int_upper(n) -> int:
        a = np.zeros(n, dtype=int)
        a[0] = -5
        a[1] = 5
        a[2] = 15
        result = np.clip(a, 2, 10)
        return result[2]

    # 15 clipped to [2, 10] -> 10
    assert numpy_clip_int_upper(10) == 10


def test_numpy_clip_with_expression():
    """Test np.clip combined with other operations (like in compute benchmark)."""

    @native
    def numpy_clip_with_ops(n) -> int:
        arr1 = np.zeros(n, dtype=int)
        arr2 = np.zeros(n, dtype=int)
        arr1[0] = 5
        arr1[1] = 500
        arr2[0] = 3
        arr2[1] = 7
        a = 4
        b = 3
        c = 9
        # Mimics: np.clip(array_1, 2, 10) * a + array_2 * b + c
        result = np.clip(arr1, 2, 10) * a + arr2 * b + c
        return result[0]

    # clip(5, 2, 10) * 4 + 3 * 3 + 9 = 5 * 4 + 9 + 9 = 20 + 9 + 9 = 38
    assert numpy_clip_with_ops(10) == 38

    @native
    def numpy_clip_with_ops_upper(n) -> int:
        arr1 = np.zeros(n, dtype=int)
        arr2 = np.zeros(n, dtype=int)
        arr1[0] = 5
        arr1[1] = 500
        arr2[0] = 3
        arr2[1] = 7
        a = 4
        b = 3
        c = 9
        result = np.clip(arr1, 2, 10) * a + arr2 * b + c
        return result[1]

    # clip(500, 2, 10) * 4 + 7 * 3 + 9 = 10 * 4 + 21 + 9 = 40 + 21 + 9 = 70
    assert numpy_clip_with_ops_upper(10) == 70


def test_numpy_clip_2d():
    """Test np.clip with 2D arrays."""

    @native
    def numpy_clip_2d(m, n) -> float:
        a = np.zeros((m, n), dtype=float)
        a[0, 0] = -10.0
        a[0, 1] = 5.0
        a[1, 0] = 20.0
        a[1, 1] = 8.0
        result = np.clip(a, 0.0, 10.0)
        return result[1, 0]

    # 20.0 clipped to [0, 10] -> 10.0
    assert abs(numpy_clip_2d(3, 3) - 10.0) < 1e-10
