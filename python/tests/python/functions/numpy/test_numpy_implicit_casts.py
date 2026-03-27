import pytest
import numpy as np

from docc.python import native


# =============================================================================
# Test: Python literals adapt to array (no promotion)
# =============================================================================


class TestLiteralAdaptsToArray:
    """Test that Python literals adapt to the array's dtype."""

    def test_float32_array_plus_float_literal(self):
        """float32[] + 1.0 → float32 (literal adapts to array)"""

        @native
        def func(a):
            return a + 1.0

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = func(a)
        expected = a + 1.0
        assert result.dtype == np.float32
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_float32_array_plus_int_literal(self):
        """float32[] + 1 → float32 (int literal adapts to array)"""

        @native
        def func(a):
            return a + 1

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = func(a)
        expected = a + 1
        assert result.dtype == np.float32
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_int32_array_plus_int_literal(self):
        """int32[] + 1 → int32 (int literal adapts to array)"""

        @native
        def func(a):
            return a + 1

        a = np.array([1, 2, 3], dtype=np.int32)
        result = func(a)
        expected = a + 1
        assert result.dtype == np.int32
        assert result.dtype == expected.dtype
        assert np.array_equal(result, expected)


# =============================================================================
# Test: Python scalars adapt to array (no promotion)
# =============================================================================


class TestPythonScalarAdaptsToArray:
    """Test that Python native scalars (int, float) passed as arguments
    adapt to the array's dtype, just like literals."""

    def test_float32_array_plus_python_float(self):
        """float32[] + float → float32 (Python float adapts to array)"""

        @native
        def float32_array_plus_python_float(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = 1.0  # Python float
        result = float32_array_plus_python_float(a, b)
        expected = a + b
        assert result.dtype == np.float32
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_float32_array_plus_python_int(self):
        """float32[] + int → float32 (Python int adapts to array)"""

        @native
        def float32_array_plus_python_int(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = 1  # Python int
        result = float32_array_plus_python_int(a, b)
        expected = a + b
        assert result.dtype == np.float32
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_float64_array_plus_python_float(self):
        """float64[] + float → float64 (Python float adapts to array)"""

        @native
        def float64_array_plus_python_float(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = 1.0  # Python float
        result = float64_array_plus_python_float(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_int32_array_plus_python_int(self):
        """int32[] + int → int32 (Python int adapts to array)"""

        @native
        def int32_array_plus_python_int(a, b):
            return a + b

        a = np.array([1, 2, 3], dtype=np.int32)
        b = 1  # Python int
        result = int32_array_plus_python_int(a, b)
        expected = a + b
        assert result.dtype == np.int32
        assert result.dtype == expected.dtype
        assert np.array_equal(result, expected)

    def test_int64_array_plus_python_int(self):
        """int64[] + int → int64 (Python int adapts to array)"""

        @native
        def int64_array_plus_python_int(a, b):
            return a + b

        a = np.array([1, 2, 3], dtype=np.int64)
        b = 1  # Python int
        result = int64_array_plus_python_int(a, b)
        expected = a + b
        assert result.dtype == np.int64
        assert result.dtype == expected.dtype
        assert np.array_equal(result, expected)


# =============================================================================
# Test: NumPy scalars trigger full promotion (like arrays)
# =============================================================================


class TestNumpyScalarPromotion:
    """Test that NumPy scalar types (np.float64, np.int32, etc.) trigger full
    type promotion, just like arrays. This differs from Python literals."""

    def test_float32_array_plus_float64_scalar(self):
        """float32[] + np.float64 → float64 (numpy scalar triggers promotion)"""

        @native
        def float32_array_plus_float64_scalar(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.float64(1.0)
        result = float32_array_plus_float64_scalar(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_float32_array_plus_int64_scalar(self):
        """float32[] + np.int64 → float64 (numpy scalar triggers promotion)"""

        @native
        def float32_array_plus_int64_scalar(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.int64(1)
        result = float32_array_plus_int64_scalar(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_float32_array_plus_int32_scalar(self):
        """float32[] + np.int32 → float64 (numpy scalar triggers promotion)"""

        @native
        def float32_array_plus_int32_scalar(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.int32(1)
        result = float32_array_plus_int32_scalar(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_float64_array_plus_float32_scalar(self):
        """float64[] + np.float32 → float64 (wider type wins)"""

        @native
        def float64_array_plus_float32_scalar(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.float32(1.0)
        result = float64_array_plus_float32_scalar(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert result.dtype == expected.dtype
        assert np.allclose(result, expected)

    def test_int64_array_plus_int32_scalar(self):
        """int64[] + np.int32 → int64 (wider int wins)"""

        @native
        def int64_array_plus_int32_scalar(a, b):
            return a + b

        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.int32(1)
        result = int64_array_plus_int32_scalar(a, b)
        expected = a + b
        assert result.dtype == np.int64
        assert result.dtype == expected.dtype
        assert np.array_equal(result, expected)


# =============================================================================
# Test: Array + Array promotion (full type promotion rules)
# =============================================================================


class TestArrayArrayPromotion:
    """Test type promotion when both operands are arrays."""

    # --- Float + Float ---

    def test_float32_array_plus_float64_array(self):
        """float32[] + float64[] → float64 (wider float wins)"""

        @native
        def float32_array_plus_float64_array(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        result = float32_array_plus_float64_array(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert expected.dtype == result.dtype
        assert np.allclose(result, expected)

    def test_float64_array_plus_float32_array(self):
        """float64[] + float32[] → float64 (wider float wins)"""

        @native
        def float64_array_plus_float32_array(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = float64_array_plus_float32_array(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert expected.dtype == result.dtype
        assert np.allclose(result, expected)

    def test_float32_array_plus_float32_array(self):
        """float32[] + float32[] → float32 (same type)"""

        @native
        def float32_array_plus_float32_array(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = float32_array_plus_float32_array(a, b)
        expected = a + b
        assert result.dtype == np.float32
        assert expected.dtype == result.dtype
        assert np.allclose(result, expected)

    # --- Int + Int ---

    def test_int32_array_plus_int64_array(self):
        """int32[] + int64[] → int64 (wider int wins)"""

        @native
        def int32_array_plus_int64_array(a, b):
            return a + b

        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([10, 20, 30], dtype=np.int64)
        result = int32_array_plus_int64_array(a, b)
        expected = a + b
        assert result.dtype == np.int64
        assert expected.dtype == result.dtype
        assert np.array_equal(result, expected)

    def test_int64_array_plus_int32_array(self):
        """int64[] + int32[] → int64 (wider int wins)"""

        @native
        def int64_array_plus_int32_array(a, b):
            return a + b

        a = np.array([1, 2, 3], dtype=np.int64)
        b = np.array([10, 20, 30], dtype=np.int32)
        result = int64_array_plus_int32_array(a, b)
        expected = a + b
        assert result.dtype == np.int64
        assert expected.dtype == result.dtype
        assert np.array_equal(result, expected)

    def test_int32_array_plus_int32_array(self):
        """int32[] + int32[] → int32 (same type)"""

        @native
        def int32_array_plus_int32_array(a, b):
            return a + b

        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([10, 20, 30], dtype=np.int32)
        result = int32_array_plus_int32_array(a, b)
        expected = a + b
        assert result.dtype == np.int32
        assert expected.dtype == result.dtype
        assert np.array_equal(result, expected)

    # --- Float + Int (mixed) ---

    def test_float32_array_plus_int32_array(self):
        """float32[] + int32[] → float64 (float32 can't represent all int32)"""

        @native
        def float32_array_plus_int32_array(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([10, 20, 30], dtype=np.int32)
        result = float32_array_plus_int32_array(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert expected.dtype == result.dtype
        assert np.allclose(result, expected)

    def test_float64_array_plus_int32_array(self):
        """float64[] + int32[] → float64 (float64 can represent int32)"""

        @native
        def float64_array_plus_int32_array(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([10, 20, 30], dtype=np.int32)
        result = float64_array_plus_int32_array(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert expected.dtype == result.dtype
        assert np.allclose(result, expected)

    def test_float64_array_plus_int64_array(self):
        """float64[] + int64[] → float64"""

        @native
        def float64_array_plus_int64_array(a, b):
            return a + b

        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([10, 20, 30], dtype=np.int64)
        result = float64_array_plus_int64_array(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert expected.dtype == result.dtype
        assert np.allclose(result, expected)

    def test_int32_array_plus_float32_array(self):
        """int32[] + float32[] → float64 (symmetric with float32+int32)"""

        @native
        def int32_array_plus_float32_array(a, b):
            return a + b

        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = int32_array_plus_float32_array(a, b)
        expected = a + b
        assert result.dtype == np.float64
        assert expected.dtype == result.dtype
        assert np.allclose(result, expected)
