from docc.python import native
import numpy as np


def test_int_cast_from_float():
    """Test casting from float to int using int()"""

    @native
    def cast_float_to_int(x: float) -> int:
        return int(x)

    # Test positive float
    result = cast_float_to_int(3.7)
    assert result == 3, f"Expected 3, got {result}"

    # Test negative float
    result = cast_float_to_int(-2.9)
    assert result == -2, f"Expected -2, got {result}"

    # Test zero
    result = cast_float_to_int(0.0)
    assert result == 0, f"Expected 0, got {result}"


def test_int_cast_from_bool():
    """Test casting from bool to int using int()"""

    @native
    def cast_bool_to_int(x: bool) -> int:
        return int(x)

    # Test True
    result = cast_bool_to_int(True)
    assert result == 1, f"Expected 1, got {result}"

    # Test False
    result = cast_bool_to_int(False)
    assert result == 0, f"Expected 0, got {result}"


def test_float_cast_from_int():
    """Test casting from int to float using float()"""

    @native
    def cast_int_to_float(x: int) -> float:
        return float(x)

    # Test positive int
    result = cast_int_to_float(42)
    assert abs(result - 42.0) < 1e-9, f"Expected 42.0, got {result}"

    # Test negative int
    result = cast_int_to_float(-17)
    assert abs(result - (-17.0)) < 1e-9, f"Expected -17.0, got {result}"

    # Test zero
    result = cast_int_to_float(0)
    assert abs(result - 0.0) < 1e-9, f"Expected 0.0, got {result}"


def test_float_cast_from_bool():
    """Test casting from bool to float using float()"""

    @native
    def cast_bool_to_float(x: bool) -> float:
        return float(x)

    # Test True
    result = cast_bool_to_float(True)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"

    # Test False
    result = cast_bool_to_float(False)
    assert abs(result - 0.0) < 1e-9, f"Expected 0.0, got {result}"


def test_bool_cast_from_int():
    """Test casting from int to bool using bool()"""

    @native
    def cast_int_to_bool(x: int) -> bool:
        return bool(x)

    # Test non-zero int (should be True)
    result = cast_int_to_bool(42)
    assert result == True, f"Expected True, got {result}"

    # Test zero (should be False)
    result = cast_int_to_bool(0)
    assert result == False, f"Expected False, got {result}"

    # Test negative int (should be True)
    result = cast_int_to_bool(-5)
    assert result == True, f"Expected True, got {result}"


def test_bool_cast_from_float():
    """Test casting from float to bool using bool()"""

    @native
    def cast_float_to_bool(x: float) -> bool:
        return bool(x)

    # Test non-zero float (should be True)
    result = cast_float_to_bool(3.14)
    assert result == True, f"Expected True, got {result}"

    # Test zero (should be False)
    result = cast_float_to_bool(0.0)
    assert result == False, f"Expected False, got {result}"

    # Test negative float (should be True)
    result = cast_float_to_bool(-2.5)
    assert result == True, f"Expected True, got {result}"


def test_cast_in_expression():
    """Test using casts within expressions"""

    @native
    def cast_in_expr(x: float, y: int) -> float:
        # Cast int to float and add to float
        result = x + float(y)
        return result

    result = cast_in_expr(3.5, 2)
    assert abs(result - 5.5) < 1e-9, f"Expected 5.5, got {result}"


def test_cast_with_scalar_variable():
    """Test casting scalar variables"""

    @native
    def cast_variable(x: float) -> int:
        y = x + 1.5
        z = int(y)
        return z

    result = cast_variable(2.3)
    assert result == 3, f"Expected 3, got {result}"


def test_multiple_casts():
    """Test multiple casts in sequence"""

    @native
    def multiple_casts(x: float) -> bool:
        # float -> int -> bool
        y = int(x)
        z = bool(y)
        return z

    # Test non-zero value
    result = multiple_casts(3.7)
    assert result == True, f"Expected True, got {result}"

    # Test value that becomes zero
    result = multiple_casts(0.5)
    assert result == False, f"Expected False, got {result}"


def test_cast_with_array_element():
    """Test casting array elements"""

    @native
    def cast_array_element(A, B):
        for i in range(A.shape[0]):
            B[i] = int(A[i])

    N = 5
    A = np.array([1.2, 2.7, 3.9, 4.1, 5.8], dtype=np.float64)
    B = np.zeros(N, dtype=np.int64)

    cast_array_element(A, B)
    expected = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    assert np.array_equal(B, expected), f"Expected {expected}, got {B}"


def test_cast_in_condition():
    """Test using casts in conditional statements"""

    @native
    def cast_in_condition(x: float) -> int:
        if bool(x):
            return int(x)
        else:
            return 0

    result = cast_in_condition(5.5)
    assert result == 5, f"Expected 5, got {result}"

    result = cast_in_condition(0.0)
    assert result == 0, f"Expected 0, got {result}"


def test_cast_float_to_int_large_values():
    """Test casting large float values to int"""

    @native
    def cast_large_float(x: float) -> int:
        return int(x)

    result = cast_large_float(1000000.5)
    assert result == 1000000, f"Expected 1000000, got {result}"

    result = cast_large_float(-999999.9)
    assert result == -999999, f"Expected -999999, got {result}"
