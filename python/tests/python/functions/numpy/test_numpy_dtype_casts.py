"""Tests for explicit NumPy scalar-type casts, e.g. np.float32(x), np.int32(arr).

Covers both scalar casts (returning a value of the target dtype) and array
casts (returning a new array of the target dtype), mirroring NumPy semantics.

Every ``@native`` function has a unique name: the JIT compile cache keys on the
function name plus its argument signature (not its body), so two identically
named kernels with the same signature would otherwise alias to the same binary.
"""

from docc.python import native
import pytest
import numpy as np


# --------------------------------------------------------------------------- #
# Scalar casts
# --------------------------------------------------------------------------- #


def test_scalar_float_to_int32_truncates():
    @native
    def cast_f_to_i32(x: float):
        return np.int32(x)

    for v in (3.7, -2.9, 0.0, 5.0):
        result = cast_f_to_i32(v)
        expected = np.int32(v)
        assert int(result) == int(expected), f"np.int32({v}) -> {result}"


def test_scalar_float_to_int64_truncates():
    @native
    def cast_f_to_i64(x: float):
        return np.int64(x)

    for v in (10.9, -4.2, 0.0):
        result = cast_f_to_i64(v)
        expected = np.int64(v)
        assert int(result) == int(expected), f"np.int64({v}) -> {result}"


def test_scalar_int_to_float32():
    @native
    def cast_i_to_f32(x: int):
        return np.float32(x)

    for v in (42, -17, 0):
        result = cast_i_to_f32(v)
        assert abs(float(result) - float(np.float32(v))) < 1e-6


def test_scalar_int_to_float64():
    @native
    def cast_i_to_f64(x: int):
        return np.float64(x)

    for v in (42, -17, 0):
        result = cast_i_to_f64(v)
        assert abs(float(result) - float(v)) < 1e-12


def test_scalar_float64_to_float32_precision_loss():
    """np.float32 of a value not exactly representable loses precision."""

    @native
    def cast_f64_to_f32(x: float):
        return np.float32(x)

    v = 0.1
    result = cast_f64_to_f32(v)
    # float32(0.1) differs from float64(0.1); the cast must reproduce the
    # float32 rounding rather than keeping full double precision.
    assert abs(float(result) - float(np.float32(v))) < 1e-9
    assert abs(float(result) - v) > 1e-9


def test_scalar_int_narrowing_int16():
    @native
    def cast_i_to_i16(x: int):
        return np.int16(x)

    # 40000 wraps around in int16 (max 32767) exactly like NumPy.
    for v in (300, -300, 40000):
        result = cast_i_to_i16(v)
        expected = np.int16(np.int64(v))
        assert int(result) == int(expected), f"np.int16({v}) -> {result}"


def test_scalar_int_narrowing_int8():
    @native
    def cast_i_to_i8(x: int):
        return np.int8(x)

    for v in (100, -100, 200):
        result = cast_i_to_i8(v)
        expected = np.int8(np.int64(v))
        assert int(result) == int(expected), f"np.int8({v}) -> {result}"


def test_scalar_uint8_wraparound():
    @native
    def cast_i_to_u8(x: int):
        return np.uint8(x)

    for v in (0, 255, 256, 300):
        result = cast_i_to_u8(v)
        expected = np.uint8(np.int64(v))
        assert int(result) == int(expected), f"np.uint8({v}) -> {result}"


def test_scalar_bool_cast():
    @native
    def cast_f_to_bool(x: float):
        return np.bool_(x)

    assert bool(cast_f_to_bool(3.14)) is True
    assert bool(cast_f_to_bool(0.0)) is False
    assert bool(cast_f_to_bool(-2.5)) is True


def test_scalar_cast_in_expression():
    """A cast used inside a larger arithmetic expression."""

    @native
    def cast_expr(x: float, y: int):
        return x + np.float64(y) * 2.0

    result = cast_expr(3.5, 4)
    assert abs(float(result) - (3.5 + 4 * 2.0)) < 1e-9


def test_scalar_cast_of_computed_value():
    @native
    def cast_computed(x: float):
        acc = x * x + 1.5
        return np.int32(acc)

    result = cast_computed(2.0)
    assert int(result) == int(np.int32(2.0 * 2.0 + 1.5))


# --------------------------------------------------------------------------- #
# Array casts
# --------------------------------------------------------------------------- #


def test_array_float64_to_int32():
    @native
    def cast_arr_f64_to_i32(A):
        return np.int32(A)

    A = np.array([1.1, 2.9, 3.5, 4.2, 5.8], dtype=np.float64)
    result = cast_arr_f64_to_i32(A)
    expected = np.int32(A)
    assert result.dtype == np.int32
    assert result.shape == (5,)
    assert np.array_equal(result, expected)


def test_array_float64_to_int64():
    @native
    def cast_arr_f64_to_i64(A):
        return np.int64(A)

    A = np.array([1.9, -2.1, 3.999, 0.0], dtype=np.float64)
    result = cast_arr_f64_to_i64(A)
    expected = np.int64(A)
    assert result.dtype == np.int64
    assert np.array_equal(result, expected)


def test_array_int_to_float32():
    @native
    def cast_arr_i_to_f32(A):
        return np.float32(A)

    A = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    result = cast_arr_i_to_f32(A)
    expected = A.astype(np.float32)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)


def test_array_float64_to_float32():
    @native
    def cast_arr_f64_to_f32(A):
        return np.float32(A)

    A = np.array([0.1, 0.2, 0.3, 1.0 / 3.0], dtype=np.float64)
    result = cast_arr_f64_to_f32(A)
    expected = A.astype(np.float32)
    assert result.dtype == np.float32
    assert np.array_equal(result, expected)


def test_array_2d_cast():
    @native
    def cast_arr_2d(A):
        return np.int32(A)

    A = np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]], dtype=np.float64)
    result = cast_arr_2d(A)
    expected = np.int32(A)
    assert result.dtype == np.int32
    assert result.shape == (3, 2)
    assert np.array_equal(result, expected)


def test_array_cast_then_arithmetic():
    """Cast an int array to float32, then use it in an elementwise op."""

    @native
    def cast_then_mul(A):
        B = np.float32(A)
        return B * np.float32(2.0)

    A = np.array([1, 2, 3, 4], dtype=np.int32)
    result = cast_then_mul(A)
    expected = A.astype(np.float32) * np.float32(2.0)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)


def test_array_element_cast_in_loop():
    """np.int32 applied to individual array elements inside a loop."""

    @native
    def cast_elems(A, B):
        for i in range(A.shape[0]):
            B[i] = np.int32(A[i])

    A = np.array([1.2, 2.7, 3.9, 4.1, 5.8], dtype=np.float64)
    B = np.zeros(5, dtype=np.int32)
    cast_elems(A, B)
    assert np.array_equal(B, np.int32(A))


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_cast_wrong_arg_count_raises():
    @native
    def cast_bad_arity(x: float):
        return np.float32(x, x)

    with pytest.raises((NotImplementedError, TypeError)):
        cast_bad_arity(1.0)
