from docc.python import native
import numpy as np

# Module-level constants
GLOBAL_FLOAT = 0.5
GLOBAL_INT = 42
SCALE_FACTOR = 0.25
BET_M = 0.5
BET_P = 0.5


def test_global_float_constant():
    """Test that global float constants are captured."""

    @native
    def use_global_float(arr) -> float:
        return arr[0] * GLOBAL_FLOAT

    arr = np.array([2.0], dtype=np.float64)
    result = use_global_float(arr)
    assert result == 1.0  # 2.0 * 0.5


def test_global_int_constant():
    """Test that global int constants are captured."""

    @native
    def use_global_int(arr) -> float:
        return arr[0] + GLOBAL_INT

    arr = np.array([8.0], dtype=np.float64)
    result = use_global_int(arr)
    assert result == 50.0  # 8.0 + 42


def test_multiple_global_constants():
    """Test using multiple global constants in one function."""

    @native
    def use_multiple_globals(arr) -> float:
        x = arr[0] * BET_M
        y = arr[1] * BET_P
        return x + y

    arr = np.array([4.0, 6.0], dtype=np.float64)
    result = use_multiple_globals(arr)
    assert result == 5.0  # 4.0 * 0.5 + 6.0 * 0.5


def test_global_in_expression():
    """Test global constant used in complex expressions."""

    @native
    def global_in_expr(arr) -> float:
        result = SCALE_FACTOR * (arr[0] + arr[1])
        return result

    arr = np.array([3.0, 5.0], dtype=np.float64)
    result = global_in_expr(arr)
    assert result == 2.0  # 0.25 * (3.0 + 5.0)


def test_global_with_array_slice():
    """Test global constant with array slicing (vadv pattern)."""

    @native
    def global_with_slice(wcon, k) -> float:
        gcv = SCALE_FACTOR * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M
        return cs[0, 0]

    wcon = np.ones((4, 3, 5), dtype=np.float64)
    result = global_with_slice(wcon, 1)
    # 0.25 * (1 + 1) * 0.5 = 0.25
    assert result == 0.25


def test_global_constant_not_overwritten_by_local():
    """Test that global constant is used when no local variable exists."""

    @native
    def global_vs_local(arr) -> float:
        # Use global SCALE_FACTOR (0.25)
        return arr[0] * SCALE_FACTOR

    arr = np.array([8.0], dtype=np.float64)
    result = global_vs_local(arr)
    assert result == 2.0  # 8.0 * 0.25
