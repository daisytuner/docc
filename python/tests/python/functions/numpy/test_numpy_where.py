"""Unit tests for np.where support in the Python frontend."""

import numpy as np
import pytest
from docc.python import native


class TestNumpyWhereComparison:
    """Tests for np.where with comparison conditions - most common pattern."""

    def test_where_greater_than_zero_keep_positive(self):
        """Test np.where(arr > 0, arr, 0.0) - keep positive values."""

        @native
        def where_gt_zero_keep_pos(arr):
            return np.where(arr > 0, arr, 0.0)

        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        compiled = where_gt_zero_keep_pos.compile(arr)
        result = compiled(arr)
        expected = np.where(arr > 0, arr, 0.0)
        np.testing.assert_array_equal(result, expected)

    def test_where_less_than_zero_keep_negative(self):
        """Test np.where(arr < 0, arr, 0.0) - keep negative values."""

        @native
        def where_lt_zero_keep_neg(arr):
            return np.where(arr < 0, arr, 0.0)

        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        compiled = where_lt_zero_keep_neg.compile(arr)
        result = compiled(arr)
        expected = np.where(arr < 0, arr, 0.0)
        np.testing.assert_array_equal(result, expected)

    def test_where_replace_positive_with_zero(self):
        """Test np.where(arr > 0, 0.0, arr) - replace positive with zero (hdiff pattern)."""

        @native
        def where_replace_pos_zero(arr):
            return np.where(arr > 0, 0.0, arr)

        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        compiled = where_replace_pos_zero.compile(arr)
        result = compiled(arr)
        expected = np.where(arr > 0, 0.0, arr)
        np.testing.assert_array_equal(result, expected)

    def test_where_replace_negative_with_zero(self):
        """Test np.where(arr < 0, 0.0, arr) - replace negative with zero."""

        @native
        def where_replace_neg_zero(arr):
            return np.where(arr < 0, 0.0, arr)

        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        compiled = where_replace_neg_zero.compile(arr)
        result = compiled(arr)
        expected = np.where(arr < 0, 0.0, arr)
        np.testing.assert_array_equal(result, expected)


class TestNumpyWhere2D:
    """Tests for np.where with 2D arrays."""

    def test_where_2d_keep_positive(self):
        """Test np.where with 2D array - keep positive values."""

        @native
        def where_2d_keep_pos(arr):
            return np.where(arr > 0, arr, 0.0)

        arr = np.array([[1.0, -2.0], [-3.0, 4.0]])
        compiled = where_2d_keep_pos.compile(arr)
        result = compiled(arr)
        expected = np.where(arr > 0, arr, 0.0)
        np.testing.assert_array_equal(result, expected)

    def test_where_2d_replace_positive(self):
        """Test np.where with 2D array - replace positive with zero."""

        @native
        def where_2d_replace_pos(arr):
            return np.where(arr > 0, 0.0, arr)

        arr = np.array([[1.0, -2.0], [-3.0, 4.0]])
        compiled = where_2d_replace_pos.compile(arr)
        result = compiled(arr)
        expected = np.where(arr > 0, 0.0, arr)
        np.testing.assert_array_equal(result, expected)


class TestNumpyWhereInputArrays:
    """Tests for np.where with multiple input arrays."""

    def test_where_all_arrays(self):
        """Test np.where(cond, x, y) - all input arrays."""

        @native
        def where_all_arrays(cond, x, y):
            return np.where(cond, x, y)

        cond = np.array([True, False, True, False, True])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        compiled = where_all_arrays.compile(cond, x, y)
        result = compiled(cond, x, y)
        expected = np.where(cond, x, y)
        np.testing.assert_array_equal(result, expected)

    def test_where_scalar_x(self):
        """Test np.where(cond, scalar, y) - scalar x value."""

        @native
        def where_scalar_x(cond, y):
            return np.where(cond, 0.0, y)

        cond = np.array([True, False, True, False, True])
        y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

        compiled = where_scalar_x.compile(cond, y)
        result = compiled(cond, y)
        expected = np.where(cond, 0.0, y)
        np.testing.assert_array_equal(result, expected)

    def test_where_scalar_y(self):
        """Test np.where(cond, x, scalar) - scalar y value."""

        @native
        def where_scalar_y(cond, x):
            return np.where(cond, x, 0.0)

        cond = np.array([True, False, True, False, True])
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        compiled = where_scalar_y.compile(cond, x)
        result = compiled(cond, x)
        expected = np.where(cond, x, 0.0)
        np.testing.assert_array_equal(result, expected)


class TestNumpyWhereSliced:
    """Tests for np.where with sliced arrays (like hdiff pattern)."""

    def test_where_sliced_1d(self):
        """Test np.where with sliced input array."""

        @native
        def where_sliced_1d(a):
            left = a[:-1]
            right = a[1:]
            diff = right - left
            return np.where(diff > 0, diff, 0.0)

        a = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        compiled = where_sliced_1d.compile(a)
        result = compiled(a)
        # diff = [2, -1, 3, -1] -> where > 0 -> [2, 0, 3, 0]
        expected = np.array([2.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_where_sliced_replace_positive(self):
        """Test np.where replacing positive with zero on sliced array."""

        @native
        def where_sliced_replace_pos(a):
            left = a[:-1]
            right = a[1:]
            diff = right - left
            return np.where(diff > 0, 0.0, diff)

        a = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        compiled = where_sliced_replace_pos.compile(a)
        result = compiled(a)
        # diff = [2, -1, 3, -1] -> where > 0 use 0 else diff -> [0, -1, 0, -1]
        expected = np.array([0.0, -1.0, 0.0, -1.0])
        np.testing.assert_array_equal(result, expected)

    def test_where_sliced_2d(self):
        """Test np.where with 2D sliced arrays."""

        @native
        def where_sliced_2d(a):
            center = a[1:-1, 1:-1]
            return np.where(center > 0, center, 0.0)

        a = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]])
        compiled = where_sliced_2d.compile(a)
        result = compiled(a)
        # center = [[5.0]] -> where > 0 -> [[5.0]]
        expected = np.array([[5.0]])
        np.testing.assert_array_equal(result, expected)


class TestNumpyWhereChained:
    """Tests for np.where in more complex expressions."""

    def test_where_followed_by_operation(self):
        """Test np.where result used in another operation."""

        @native
        def where_then_multiply(arr):
            clipped = np.where(arr > 0, 0.0, arr)
            return clipped * 2.0

        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        compiled = where_then_multiply.compile(arr)
        result = compiled(arr)
        expected = np.where(arr > 0, 0.0, arr) * 2.0
        np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
