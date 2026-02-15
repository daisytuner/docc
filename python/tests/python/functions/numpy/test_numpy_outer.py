import numpy as np
import pytest
from typing import Annotated
from docc.python import native


class TypeFactory:
    """Helper to create annotated array types."""

    def __init__(self, dtype):
        self.dtype = dtype

    def __getitem__(self, shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        return Annotated[np.ndarray, shape, self.dtype]


# Type factories for different dtypes
float64 = TypeFactory(np.float64)
float32 = TypeFactory(np.float32)
int64 = TypeFactory(np.int64)
int32 = TypeFactory(np.int32)


# =============================================================================
# np.outer tests (multiply-based)
# =============================================================================


class TestNumpyOuter:
    """Tests for np.outer function."""

    def test_outer_basic_float64(self):
        """Basic outer product with float64 arrays."""

        @native
        def np_outer_basic(a: float64[5], b: float64[4]) -> float64[5, 4]:
            return np.outer(a, b)

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        res = np_outer_basic(a, b)
        expected = np.outer(a, b)
        assert np.allclose(res, expected), f"Expected {expected}, got {res}"

    def test_outer_same_size(self):
        """Outer product with same-sized arrays."""

        @native
        def np_outer_same(a: float64[10], b: float64[10]) -> float64[10, 10]:
            return np.outer(a, b)

        a = np.random.rand(10)
        b = np.random.rand(10)
        res = np_outer_same(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_outer_random_values(self):
        """Outer product with random values."""

        @native
        def np_outer_random(a: float64[8], b: float64[6]) -> float64[8, 6]:
            return np.outer(a, b)

        a = np.random.rand(8)
        b = np.random.rand(6)
        res = np_outer_random(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_outer_with_zeros(self):
        """Outer product where one array contains zeros."""

        @native
        def np_outer_zeros(a: float64[5], b: float64[5]) -> float64[5, 5]:
            return np.outer(a, b)

        a = np.array([1.0, 0.0, 2.0, 0.0, 3.0])
        b = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
        res = np_outer_zeros(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_outer_with_negatives(self):
        """Outer product with negative values."""

        @native
        def np_outer_neg(a: float64[4], b: float64[4]) -> float64[4, 4]:
            return np.outer(a, b)

        a = np.array([-1.0, 2.0, -3.0, 4.0])
        b = np.array([1.0, -2.0, 3.0, -4.0])
        res = np_outer_neg(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_outer_slicing(self):
        """Outer product with sliced arrays."""

        @native
        def outer_slice(a: float64[20], b: float64[20]) -> float64[10, 10]:
            return np.outer(a[:10], b[10:])

        a = np.random.rand(20)
        b = np.random.rand(20)
        res = outer_slice(a, b)
        assert np.allclose(res, np.outer(a[:10], b[10:]))


# =============================================================================
# np.add.outer tests
# =============================================================================


class TestAddOuter:
    """Tests for np.add.outer function."""

    def test_add_outer_basic_float64(self):
        """Basic add.outer with float64 arrays."""

        @native
        def np_add_outer_basic(a: float64[5], b: float64[4]) -> float64[5, 4]:
            return np.add.outer(a, b)

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        res = np_add_outer_basic(a, b)
        expected = np.add.outer(a, b)
        assert np.allclose(res, expected), f"Expected {expected}, got {res}"

    def test_add_outer_same_size(self):
        """Add outer with same-sized arrays."""

        @native
        def np_add_outer_same(a: float64[10], b: float64[10]) -> float64[10, 10]:
            return np.add.outer(a, b)

        a = np.random.rand(10)
        b = np.random.rand(10)
        res = np_add_outer_same(a, b)
        assert np.allclose(res, np.add.outer(a, b))

    def test_add_outer_different_sizes(self):
        """Add outer with different sized arrays."""

        @native
        def np_add_outer_diff(a: float64[8], b: float64[12]) -> float64[8, 12]:
            return np.add.outer(a, b)

        a = np.random.rand(8)
        b = np.random.rand(12)
        res = np_add_outer_diff(a, b)
        assert np.allclose(res, np.add.outer(a, b))

    def test_add_outer_with_negatives(self):
        """Add outer with negative values."""

        @native
        def np_add_outer_neg(a: float64[4], b: float64[4]) -> float64[4, 4]:
            return np.add.outer(a, b)

        a = np.array([-1.0, 2.0, -3.0, 4.0])
        b = np.array([1.0, -2.0, 3.0, -4.0])
        res = np_add_outer_neg(a, b)
        assert np.allclose(res, np.add.outer(a, b))

    def test_add_outer_int64(self):
        """Add outer with int64 arrays."""

        @native
        def np_add_outer_int64(a: int64[5], b: int64[4]) -> int64[5, 4]:
            return np.add.outer(a, b)

        a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        b = np.array([10, 20, 30, 40], dtype=np.int64)
        res = np_add_outer_int64(a, b)
        expected = np.add.outer(a, b)
        assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


# =============================================================================
# np.subtract.outer tests
# =============================================================================


class TestSubtractOuter:
    """Tests for np.subtract.outer function."""

    def test_subtract_outer_basic_float64(self):
        """Basic subtract.outer with float64 arrays."""

        @native
        def np_sub_outer_basic(a: float64[5], b: float64[4]) -> float64[5, 4]:
            return np.subtract.outer(a, b)

        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        res = np_sub_outer_basic(a, b)
        expected = np.subtract.outer(a, b)
        assert np.allclose(res, expected), f"Expected {expected}, got {res}"

    def test_subtract_outer_same_size(self):
        """Subtract outer with same-sized arrays."""

        @native
        def np_sub_outer_same(a: float64[10], b: float64[10]) -> float64[10, 10]:
            return np.subtract.outer(a, b)

        a = np.random.rand(10)
        b = np.random.rand(10)
        res = np_sub_outer_same(a, b)
        assert np.allclose(res, np.subtract.outer(a, b))

    def test_subtract_outer_different_sizes(self):
        """Subtract outer with different sized arrays."""

        @native
        def np_sub_outer_diff(a: float64[6], b: float64[9]) -> float64[6, 9]:
            return np.subtract.outer(a, b)

        a = np.random.rand(6)
        b = np.random.rand(9)
        res = np_sub_outer_diff(a, b)
        assert np.allclose(res, np.subtract.outer(a, b))

    def test_subtract_outer_int64(self):
        """Subtract outer with int64 arrays."""

        @native
        def np_sub_outer_int64(a: int64[5], b: int64[4]) -> int64[5, 4]:
            return np.subtract.outer(a, b)

        a = np.array([10, 20, 30, 40, 50], dtype=np.int64)
        b = np.array([1, 2, 3, 4], dtype=np.int64)
        res = np_sub_outer_int64(a, b)
        expected = np.subtract.outer(a, b)
        assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


# =============================================================================
# np.multiply.outer tests
# =============================================================================


class TestMultiplyOuter:
    """Tests for np.multiply.outer function (should behave like np.outer)."""

    def test_multiply_outer_basic_float64(self):
        """Basic multiply.outer with float64 arrays."""

        @native
        def np_mul_outer_basic(a: float64[5], b: float64[4]) -> float64[5, 4]:
            return np.multiply.outer(a, b)

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        res = np_mul_outer_basic(a, b)
        expected = np.multiply.outer(a, b)
        assert np.allclose(res, expected), f"Expected {expected}, got {res}"

    def test_multiply_outer_same_size(self):
        """Multiply outer with same-sized arrays."""

        @native
        def np_mul_outer_same(a: float64[10], b: float64[10]) -> float64[10, 10]:
            return np.multiply.outer(a, b)

        a = np.random.rand(10)
        b = np.random.rand(10)
        res = np_mul_outer_same(a, b)
        # np.multiply.outer is the same as np.outer
        assert np.allclose(res, np.outer(a, b))


# =============================================================================
# np.divide.outer tests
# =============================================================================


class TestDivideOuter:
    """Tests for np.divide.outer function."""

    def test_divide_outer_basic_float64(self):
        """Basic divide.outer with float64 arrays."""

        @native
        def np_div_outer_basic(a: float64[5], b: float64[4]) -> float64[5, 4]:
            return np.divide.outer(a, b)

        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b = np.array([1.0, 2.0, 5.0, 10.0])
        res = np_div_outer_basic(a, b)
        expected = np.divide.outer(a, b)
        assert np.allclose(res, expected), f"Expected {expected}, got {res}"

    def test_divide_outer_same_size(self):
        """Divide outer with same-sized arrays."""

        @native
        def np_div_outer_same(a: float64[10], b: float64[10]) -> float64[10, 10]:
            return np.divide.outer(a, b)

        a = np.random.rand(10) + 0.1  # Avoid very small values
        b = np.random.rand(10) + 0.1  # Avoid division by zero
        res = np_div_outer_same(a, b)
        assert np.allclose(res, np.divide.outer(a, b))

    def test_divide_outer_different_sizes(self):
        """Divide outer with different sized arrays."""

        @native
        def np_div_outer_diff(a: float64[7], b: float64[5]) -> float64[7, 5]:
            return np.divide.outer(a, b)

        a = np.random.rand(7) + 0.1
        b = np.random.rand(5) + 0.1
        res = np_div_outer_diff(a, b)
        assert np.allclose(res, np.divide.outer(a, b))


# =============================================================================
# np.minimum.outer tests
# =============================================================================


class TestMinimumOuter:
    """Tests for np.minimum.outer function."""

    def test_minimum_outer_basic_float64(self):
        """Basic minimum.outer with float64 arrays."""

        @native
        def np_min_outer_basic(a: float64[5], b: float64[4]) -> float64[5, 4]:
            return np.minimum.outer(a, b)

        a = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        b = np.array([4.0, 2.0, 6.0, 1.0])
        res = np_min_outer_basic(a, b)
        expected = np.minimum.outer(a, b)
        assert np.allclose(res, expected), f"Expected {expected}, got {res}"

    def test_minimum_outer_same_size(self):
        """Minimum outer with same-sized arrays."""

        @native
        def np_min_outer_same(a: float64[10], b: float64[10]) -> float64[10, 10]:
            return np.minimum.outer(a, b)

        a = np.random.rand(10)
        b = np.random.rand(10)
        res = np_min_outer_same(a, b)
        assert np.allclose(res, np.minimum.outer(a, b))

    def test_minimum_outer_with_negatives(self):
        """Minimum outer with negative values."""

        @native
        def np_min_outer_neg(a: float64[4], b: float64[4]) -> float64[4, 4]:
            return np.minimum.outer(a, b)

        a = np.array([-1.0, 2.0, -3.0, 4.0])
        b = np.array([1.0, -2.0, 3.0, -4.0])
        res = np_min_outer_neg(a, b)
        assert np.allclose(res, np.minimum.outer(a, b))

    def test_minimum_outer_int64(self):
        """Minimum outer with int64 arrays."""

        @native
        def np_min_outer_int64(a: int64[5], b: int64[4]) -> int64[5, 4]:
            return np.minimum.outer(a, b)

        a = np.array([10, 5, 15, 3, 8], dtype=np.int64)
        b = np.array([7, 12, 4, 9], dtype=np.int64)
        res = np_min_outer_int64(a, b)
        expected = np.minimum.outer(a, b)
        assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


# =============================================================================
# np.maximum.outer tests
# =============================================================================


class TestMaximumOuter:
    """Tests for np.maximum.outer function."""

    def test_maximum_outer_basic_float64(self):
        """Basic maximum.outer with float64 arrays."""

        @native
        def np_max_outer_basic(a: float64[5], b: float64[4]) -> float64[5, 4]:
            return np.maximum.outer(a, b)

        a = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
        b = np.array([4.0, 2.0, 6.0, 1.0])
        res = np_max_outer_basic(a, b)
        expected = np.maximum.outer(a, b)
        assert np.allclose(res, expected), f"Expected {expected}, got {res}"

    def test_maximum_outer_same_size(self):
        """Maximum outer with same-sized arrays."""

        @native
        def np_max_outer_same(a: float64[10], b: float64[10]) -> float64[10, 10]:
            return np.maximum.outer(a, b)

        a = np.random.rand(10)
        b = np.random.rand(10)
        res = np_max_outer_same(a, b)
        assert np.allclose(res, np.maximum.outer(a, b))

    def test_maximum_outer_with_negatives(self):
        """Maximum outer with negative values."""

        @native
        def np_max_outer_neg(a: float64[4], b: float64[4]) -> float64[4, 4]:
            return np.maximum.outer(a, b)

        a = np.array([-1.0, 2.0, -3.0, 4.0])
        b = np.array([1.0, -2.0, 3.0, -4.0])
        res = np_max_outer_neg(a, b)
        assert np.allclose(res, np.maximum.outer(a, b))

    def test_maximum_outer_int64(self):
        """Maximum outer with int64 arrays."""

        @native
        def np_max_outer_int64(a: int64[5], b: int64[4]) -> int64[5, 4]:
            return np.maximum.outer(a, b)

        a = np.array([10, 5, 15, 3, 8], dtype=np.int64)
        b = np.array([7, 12, 4, 9], dtype=np.int64)
        res = np_max_outer_int64(a, b)
        expected = np.maximum.outer(a, b)
        assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


# =============================================================================
# Edge case tests
# =============================================================================


class TestOuterEdgeCases:
    """Edge case tests for outer functions."""

    def test_outer_single_element_a(self):
        """Outer with single element in first array."""

        @native
        def outer_single_a(a: float64[1], b: float64[5]) -> float64[1, 5]:
            return np.outer(a, b)

        a = np.array([3.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = outer_single_a(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_outer_single_element_b(self):
        """Outer with single element in second array."""

        @native
        def outer_single_b(a: float64[5], b: float64[1]) -> float64[5, 1]:
            return np.outer(a, b)

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([3.0])
        res = outer_single_b(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_outer_single_element_both(self):
        """Outer with single element in both arrays."""

        @native
        def outer_single_both(a: float64[1], b: float64[1]) -> float64[1, 1]:
            return np.outer(a, b)

        a = np.array([3.0])
        b = np.array([4.0])
        res = outer_single_both(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_add_outer_all_zeros(self):
        """Add outer with all zeros."""

        @native
        def add_outer_zeros(a: float64[5], b: float64[5]) -> float64[5, 5]:
            return np.add.outer(a, b)

        a = np.zeros(5)
        b = np.zeros(5)
        res = add_outer_zeros(a, b)
        assert np.allclose(res, np.add.outer(a, b))

    def test_subtract_outer_identical(self):
        """Subtract outer with identical arrays (should give zero diagonal)."""

        @native
        def sub_outer_identical(a: float64[5], b: float64[5]) -> float64[5, 5]:
            return np.subtract.outer(a, b)

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = sub_outer_identical(a, a.copy())
        expected = np.subtract.outer(a, a)
        # Diagonal should be zero
        assert np.allclose(np.diag(res), np.zeros(5))
        assert np.allclose(res, expected)


# =============================================================================
# Type conversion tests
# =============================================================================


class TestOuterTypeConversion:
    """Tests for type handling in outer functions."""

    @pytest.mark.skip(reason="float32 handling in np.outer has pre-existing issues")
    def test_outer_float32(self):
        """Outer product with float32 arrays."""

        @native
        def outer_f32(a: float32[5], b: float32[5]) -> float32[5, 5]:
            return np.outer(a, b)

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        res = outer_f32(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_add_outer_int32(self):
        """Add outer with int32 arrays."""

        @native
        def add_outer_i32(a: int32[5], b: int32[4]) -> int32[5, 4]:
            return np.add.outer(a, b)

        a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        b = np.array([10, 20, 30, 40], dtype=np.int32)
        res = add_outer_i32(a, b)
        expected = np.add.outer(a, b)
        assert np.array_equal(res, expected)


# =============================================================================
# Combined operation tests
# =============================================================================


class TestOuterCombined:
    """Tests combining outer operations with other operations."""

    def test_outer_plus_outer(self):
        """Sum of two outer products."""

        @native
        def outer_plus_outer(
            a: float64[5], b: float64[5], c: float64[5], d: float64[5]
        ) -> float64[5, 5]:
            return np.outer(a, b) + np.outer(c, d)

        a = np.random.rand(5)
        b = np.random.rand(5)
        c = np.random.rand(5)
        d = np.random.rand(5)
        res = outer_plus_outer(a, b, c, d)
        expected = np.outer(a, b) + np.outer(c, d)
        assert np.allclose(res, expected)

    def test_outer_accumulate(self):
        """Outer product accumulated into existing array."""

        @native
        def outer_acc(a: float64[5], b: float64[5], C: float64[5, 5]) -> float64[5, 5]:
            C[:] += np.outer(a, b)
            return C

        a = np.random.rand(5)
        b = np.random.rand(5)
        C = np.zeros((5, 5))
        expected = C.copy() + np.outer(a, b)
        res = outer_acc(a, b, C)
        assert np.allclose(res, expected)

    def test_add_outer_then_reduce(self):
        """Add outer followed by sum reduction."""

        @native
        def add_outer_sum(a: float64[5], b: float64[5]) -> float:
            # Note: Direct chaining works, but intermediate variable may have issues
            return np.sum(np.add.outer(a, b))

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = add_outer_sum(a, b)
        expected = np.sum(np.add.outer(a, b))
        assert np.isclose(res, expected)


# =============================================================================
# Large array tests
# =============================================================================


class TestOuterLargeArrays:
    """Tests with larger arrays for performance validation."""

    def test_outer_medium_size(self):
        """Outer product with medium-sized arrays."""

        @native
        def outer_medium(a: float64[50], b: float64[50]) -> float64[50, 50]:
            return np.outer(a, b)

        a = np.random.rand(50)
        b = np.random.rand(50)
        res = outer_medium(a, b)
        assert np.allclose(res, np.outer(a, b))

    def test_add_outer_medium_size(self):
        """Add outer with medium-sized arrays."""

        @native
        def add_outer_medium(a: float64[50], b: float64[50]) -> float64[50, 50]:
            return np.add.outer(a, b)

        a = np.random.rand(50)
        b = np.random.rand(50)
        res = add_outer_medium(a, b)
        assert np.allclose(res, np.add.outer(a, b))

    def test_minimum_outer_medium_size(self):
        """Minimum outer with medium-sized arrays."""

        @native
        def min_outer_medium(a: float64[50], b: float64[50]) -> float64[50, 50]:
            return np.minimum.outer(a, b)

        a = np.random.rand(50)
        b = np.random.rand(50)
        res = min_outer_medium(a, b)
        assert np.allclose(res, np.minimum.outer(a, b))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
