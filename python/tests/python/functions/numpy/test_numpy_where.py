import numpy as np
import pytest
from docc.python import native


def test_where_contiguous():

    @native
    def where_arrays(cond, x, y):
        return np.where(cond, x, y)

    cond = np.array([True, False, True, False, True])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    compiled = where_arrays.compile(cond, x, y)
    result = compiled(cond, x, y)
    expected = np.where(cond, x, y)
    np.testing.assert_array_equal(result, expected)

    @native
    def where_scalar_x(cond, y):
        return np.where(cond, 0.0, y)

    cond = np.array([True, False, True, False, True])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    compiled = where_scalar_x.compile(cond, y)
    result = compiled(cond, y)
    expected = np.where(cond, 0.0, y)
    np.testing.assert_array_equal(result, expected)

    @native
    def where_scalar_y(cond, x):
        return np.where(cond, x, 0.0)

    cond = np.array([True, False, True, False, True])
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    compiled = where_scalar_y.compile(cond, x)
    result = compiled(cond, x)
    expected = np.where(cond, x, 0.0)
    np.testing.assert_array_equal(result, expected)

    @native
    def positive_mask(arr):
        return np.where(arr > 0, arr, 0.0)

    arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    compiled = positive_mask.compile(arr)
    result = compiled(arr)
    expected = np.where(arr > 0, arr, 0.0)
    np.testing.assert_array_equal(result, expected)

    @native
    def negative_mask(arr):
        return np.where(arr < 0, arr, 0.0)

    arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    compiled = negative_mask.compile(arr)
    result = compiled(arr)
    expected = np.where(arr < 0, arr, 0.0)
    np.testing.assert_array_equal(result, expected)

    @native
    def clip_negative(arr):
        return np.where(arr > 0, 0.0, arr)

    arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    compiled = clip_negative.compile(arr)
    result = compiled(arr)
    expected = np.where(arr > 0, 0.0, arr)
    np.testing.assert_array_equal(result, expected)

    @native
    def clip_positive(arr):
        return np.where(arr < 0, 0.0, arr)

    arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
    compiled = clip_positive.compile(arr)
    result = compiled(arr)
    expected = np.where(arr < 0, 0.0, arr)
    np.testing.assert_array_equal(result, expected)

    @native
    def where_2d(arr):
        return np.where(arr > 0, arr, 0.0)

    arr = np.array([[1.0, -2.0], [-3.0, 4.0]])
    compiled = where_2d.compile(arr)
    result = compiled(arr)
    expected = np.where(arr > 0, arr, 0.0)
    np.testing.assert_array_equal(result, expected)

    @native
    def clip_2d(arr):
        return np.where(arr > 0, 0.0, arr)

    arr = np.array([[1.0, -2.0], [-3.0, 4.0]])
    compiled = clip_2d.compile(arr)
    result = compiled(arr)
    expected = np.where(arr > 0, 0.0, arr)
    np.testing.assert_array_equal(result, expected)


class TestWhereOffsetViews:
    """Tests for np.where with offset views (contiguous slices)."""

    def test_condition_is_offset_view(self):
        """Condition is an offset view: np.where(a[1:4] > 0, ...)"""

        @native
        def where_offset_cond(a):
            view = a[1:4]
            return np.where(view > 0, view, 0.0)

        a = np.array([10.0, -1.0, 2.0, -3.0, 4.0])
        compiled = where_offset_cond.compile(a)
        result = compiled(a)
        # view = [-1.0, 2.0, -3.0]
        expected = np.array([0.0, 2.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_x_is_offset_view(self):
        """X value is an offset view."""

        @native
        def where_offset_x(a, cond):
            view = a[1:4]
            return np.where(cond, view, 0.0)

        a = np.array([10.0, 1.0, 2.0, 3.0, 40.0])
        cond = np.array([True, False, True])
        compiled = where_offset_x.compile(a, cond)
        result = compiled(a, cond)
        # view = [1.0, 2.0, 3.0], cond selects [0]=1.0, [2]=3.0
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_y_is_offset_view(self):
        """Y value is an offset view."""

        @native
        def where_offset_y(a, cond):
            view = a[2:5]
            return np.where(cond, 0.0, view)

        a = np.array([10.0, 20.0, 1.0, 2.0, 3.0])
        cond = np.array([False, True, False])
        compiled = where_offset_y.compile(a, cond)
        result = compiled(a, cond)
        # view = [1.0, 2.0, 3.0], cond selects [0]=1.0, [2]=3.0
        expected = np.array([1.0, 0.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_all_offset_views(self):
        """All inputs are offset views from different positions."""

        @native
        def where_all_offset(a, b, c):
            cond_view = a[0:3] > 0  # Boolean array
            x_view = b[1:4]
            y_view = c[2:5]
            return np.where(cond_view, x_view, y_view)

        a = np.array([1.0, -2.0, 3.0, 4.0])
        b = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        c = np.array([0.0, 0.0, 100.0, 200.0, 300.0])
        compiled = where_all_offset.compile(a, b, c)
        result = compiled(a, b, c)
        # cond_view = [True, False, True] (from a[0:3] > 0)
        # x_view = [10.0, 20.0, 30.0]
        # y_view = [100.0, 200.0, 300.0]
        expected = np.array([10.0, 200.0, 30.0])
        np.testing.assert_array_equal(result, expected)

    def test_2d_offset_view(self):
        """2D offset view: a[1:3, 1:3]"""

        @native
        def where_2d_offset(a):
            view = a[1:3, 1:3]
            return np.where(view > 0, view, 0.0)

        a = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, -6.0, 7.0, 8.0],
                [9.0, 10.0, -11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
        compiled = where_2d_offset.compile(a)
        result = compiled(a)
        # view = [[-6, 7], [10, -11]]
        expected = np.array([[0.0, 7.0], [10.0, 0.0]])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 3. STRIDED VIEWS (Non-unit step: a[::2])
# =============================================================================


class TestWhereStridedViews:
    """Tests for np.where with strided views (non-contiguous)."""

    def test_condition_is_strided_view(self):
        """Condition is a strided view: np.where(a[::2] > 0, ...)"""

        @native
        def where_strided_cond(a):
            view = a[::2]  # [a[0], a[2], a[4]]
            return np.where(view > 0, view, 0.0)

        a = np.array([-1.0, 100.0, 2.0, 200.0, -3.0])
        compiled = where_strided_cond.compile(a)
        result = compiled(a)
        # view = [-1.0, 2.0, -3.0] -> where > 0 -> [0.0, 2.0, 0.0]
        expected = np.array([0.0, 2.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_x_is_strided_view(self):
        """X value is a strided view."""

        @native
        def where_strided_x(a, cond):
            view = a[::2]
            return np.where(cond, view, 0.0)

        a = np.array([10.0, 0.0, 20.0, 0.0, 30.0])
        cond = np.array([True, False, True])
        compiled = where_strided_x.compile(a, cond)
        result = compiled(a, cond)
        # view = [10.0, 20.0, 30.0]
        expected = np.array([10.0, 0.0, 30.0])
        np.testing.assert_array_equal(result, expected)

    def test_strided_with_offset(self):
        """Strided view with offset: a[1::2]"""

        @native
        def where_strided_offset(a):
            view = a[1::2]  # [a[1], a[3], a[5]]
            return np.where(view > 0, view, 0.0)

        a = np.array([0.0, -1.0, 0.0, 2.0, 0.0, -3.0])
        compiled = where_strided_offset.compile(a)
        result = compiled(a)
        # view = [-1.0, 2.0, -3.0]
        expected = np.array([0.0, 2.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_2d_strided_rows(self):
        """2D strided view on rows: a[::2, :]"""

        @native
        def where_2d_strided_rows(a):
            view = a[::2, :]
            return np.where(view > 0, view, 0.0)

        a = np.array(
            [
                [-1.0, 2.0],
                [100.0, 200.0],  # skipped
                [3.0, -4.0],
                [300.0, 400.0],  # skipped
                [-5.0, 6.0],
            ]
        )
        compiled = where_2d_strided_rows.compile(a)
        result = compiled(a)
        # view = [[-1, 2], [3, -4], [-5, 6]]
        expected = np.array([[0.0, 2.0], [3.0, 0.0], [0.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_2d_strided_cols(self):
        """2D strided view on columns: a[:, ::2]"""

        @native
        def where_2d_strided_cols(a):
            view = a[:, ::2]
            return np.where(view > 0, view, 0.0)

        a = np.array([[-1.0, 100.0, 2.0, 200.0], [3.0, 300.0, -4.0, 400.0]])
        compiled = where_2d_strided_cols.compile(a)
        result = compiled(a)
        # view = [[-1, 2], [3, -4]]
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 4. NEGATIVE STEP VIEWS (Reversed: a[::-1])
# =============================================================================


class TestWhereNegativeStepViews:
    """Tests for np.where with negative step views (reversed arrays)."""

    def test_condition_is_reversed(self):
        """Condition is reversed: np.where(a[::-1] > 0, ...)"""

        @native
        def where_reversed_cond(a):
            view = a[::-1]
            return np.where(view > 0, view, 0.0)

        a = np.array([1.0, -2.0, 3.0])
        compiled = where_reversed_cond.compile(a)
        result = compiled(a)
        # view = [3.0, -2.0, 1.0] -> where > 0 -> [3.0, 0.0, 1.0]
        expected = np.array([3.0, 0.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_x_is_reversed(self):
        """X value is reversed."""

        @native
        def where_reversed_x(a, cond):
            view = a[::-1]
            return np.where(cond, view, 0.0)

        a = np.array([10.0, 20.0, 30.0])
        cond = np.array([True, False, True])
        compiled = where_reversed_x.compile(a, cond)
        result = compiled(a, cond)
        # view = [30.0, 20.0, 10.0]
        expected = np.array([30.0, 0.0, 10.0])
        np.testing.assert_array_equal(result, expected)

    def test_reversed_step_2(self):
        """Reversed with step=-2: a[::-2]"""

        @native
        def where_reversed_step2(a):
            view = a[::-2]  # [a[4], a[2], a[0]]
            return np.where(view > 0, view, 0.0)

        a = np.array([-1.0, 100.0, 2.0, 200.0, -3.0])
        compiled = where_reversed_step2.compile(a)
        result = compiled(a)
        # view = [-3.0, 2.0, -1.0] -> where > 0 -> [0.0, 2.0, 0.0]
        expected = np.array([0.0, 2.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_2d_reversed_rows(self):
        """2D reversed rows: a[::-1, :]"""

        @native
        def where_2d_reversed_rows(a):
            view = a[::-1, :]
            return np.where(view > 0, view, 0.0)

        a = np.array([[-1.0, 2.0], [3.0, -4.0]])
        compiled = where_2d_reversed_rows.compile(a)
        result = compiled(a)
        # view = [[3, -4], [-1, 2]] -> [[3, 0], [0, 2]]
        expected = np.array([[3.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 5. MIXED VIEW/SCALAR (View condition with scalar values)
# =============================================================================


class TestWhereMixedViewScalar:
    """Tests for np.where with views and scalars mixed."""

    def test_view_cond_scalar_values(self):
        """View condition, scalar x and y."""

        @native
        def where_view_scalars(a):
            view = a[1:4]
            return np.where(view > 0, 1.0, -1.0)

        a = np.array([100.0, -1.0, 2.0, -3.0, 200.0])
        compiled = where_view_scalars.compile(a)
        result = compiled(a)
        # view = [-1.0, 2.0, -3.0] -> [False, True, False] -> [-1, 1, -1]
        expected = np.array([-1.0, 1.0, -1.0])
        np.testing.assert_array_equal(result, expected)

    def test_view_x_scalar_y(self):
        """View x, scalar y."""

        @native
        def where_view_x_scalar_y(a, cond):
            view = a[::2]
            return np.where(cond, view, 999.0)

        a = np.array([10.0, 0.0, 20.0, 0.0, 30.0])
        cond = np.array([True, False, True])
        compiled = where_view_x_scalar_y.compile(a, cond)
        result = compiled(a, cond)
        # view = [10.0, 20.0, 30.0]
        expected = np.array([10.0, 999.0, 30.0])
        np.testing.assert_array_equal(result, expected)

    def test_scalar_x_view_y(self):
        """Scalar x, view y."""

        @native
        def where_scalar_x_view_y(a, cond):
            view = a[1::2]
            return np.where(cond, 0.0, view)

        a = np.array([0.0, 10.0, 0.0, 20.0, 0.0, 30.0])
        cond = np.array([True, False, True])
        compiled = where_scalar_x_view_y.compile(a, cond)
        result = compiled(a, cond)
        # view = [10.0, 20.0, 30.0]
        expected = np.array([0.0, 20.0, 0.0])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 6. MULTIPLE VIEWS (All inputs are views from different sources)
# =============================================================================


class TestWhereMultipleViews:
    """Tests for np.where where all inputs are views."""

    def test_all_from_same_array(self):
        """All views from the same source array."""

        @native
        def where_same_source(a):
            cond = a[0:3] > 0
            x = a[1:4]
            y = a[2:5]
            return np.where(cond, x, y)

        a = np.array([1.0, -2.0, 3.0, 4.0, 5.0, 6.0])
        compiled = where_same_source.compile(a)
        result = compiled(a)
        # cond = a[0:3] > 0 = [True, False, True]
        # x = a[1:4] = [-2.0, 3.0, 4.0]
        # y = a[2:5] = [3.0, 4.0, 5.0]
        # result = [-2.0, 4.0, 4.0]
        expected = np.array([-2.0, 4.0, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_different_strides(self):
        """Views with different strides."""

        @native
        def where_diff_strides(a, b, c):
            v1 = a[::2]  # stride 2
            v2 = b[1::2]  # stride 2, offset 1
            cond = c[0:3] > 0
            return np.where(cond, v1, v2)

        a = np.array([10.0, 0.0, 20.0, 0.0, 30.0])  # v1 = [10, 20, 30]
        b = np.array([0.0, 100.0, 0.0, 200.0, 0.0, 300.0])  # v2 = [100, 200, 300]
        c = np.array([1.0, -1.0, 1.0, 1.0])  # cond = [True, False, True]
        compiled = where_diff_strides.compile(a, b, c)
        result = compiled(a, b, c)
        expected = np.array([10.0, 200.0, 30.0])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 7. CHAINED VIEWS (View of view)
# =============================================================================


class TestWhereChainedViews:
    """Tests for np.where with chained views (view of view)."""

    def test_view_of_view(self):
        """Chained view: (a[::2])[1:]"""

        @native
        def where_chained(a):
            v1 = a[::2]  # [a[0], a[2], a[4], a[6]]
            v2 = v1[1:]  # [a[2], a[4], a[6]]
            return np.where(v2 > 0, v2, 0.0)

        a = np.array([0.0, 1.0, -2.0, 3.0, 4.0, 5.0, -6.0, 7.0])
        compiled = where_chained.compile(a)
        result = compiled(a)
        # v1 = [0.0, -2.0, 4.0, -6.0]
        # v2 = [-2.0, 4.0, -6.0]
        expected = np.array([0.0, 4.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_view_of_reversed(self):
        """View of reversed: (a[::-1])[1:3]"""

        @native
        def where_view_of_reversed(a):
            rev = a[::-1]
            view = rev[1:3]
            return np.where(view > 0, view, 0.0)

        a = np.array([1.0, -2.0, 3.0, -4.0])
        compiled = where_view_of_reversed.compile(a)
        result = compiled(a)
        # rev = [-4.0, 3.0, -2.0, 1.0]
        # view = [3.0, -2.0]
        expected = np.array([3.0, 0.0])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 8. 2D VIEWS (Multi-dimensional slicing)
# =============================================================================


class TestWhere2DViews:
    """Tests for np.where with 2D views."""

    def test_2d_row_slice(self):
        """2D view slicing rows: a[1:-1, :]"""

        @native
        def where_2d_rows(a):
            view = a[1:-1, :]
            return np.where(view > 0, view, 0.0)

        a = np.array(
            [
                [100.0, 200.0, 300.0],
                [-1.0, 2.0, -3.0],
                [4.0, -5.0, 6.0],
                [400.0, 500.0, 600.0],
            ]
        )
        compiled = where_2d_rows.compile(a)
        result = compiled(a)
        expected = np.array([[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.skip(
        reason="Memory management bug with column slicing a[:, start:stop] - pre-existing issue"
    )
    def test_2d_col_slice(self):
        """2D view slicing columns: a[:, 1:-1]"""

        @native
        def where_2d_cols(a):
            view = a[:, 1:-1]
            return np.where(view > 0, view, 0.0)

        a = np.array([[100.0, -1.0, 2.0, 200.0], [300.0, 3.0, -4.0, 400.0]])
        compiled = where_2d_cols.compile(a)
        result = compiled(a)
        expected = np.array([[0.0, 2.0], [3.0, 0.0]])
        np.testing.assert_array_equal(result, expected)

    def test_2d_both_dims(self):
        """2D view slicing both dimensions: a[1:3, 1:3]"""

        @native
        def where_2d_both(a):
            view = a[1:3, 1:3]
            return np.where(view > 0, view, 0.0)

        a = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, -6.0, 7.0, 8.0],
                [9.0, 10.0, -11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]
        )
        compiled = where_2d_both.compile(a)
        result = compiled(a)
        expected = np.array([[0.0, 7.0], [10.0, 0.0]])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 9. HDIFF PATTERN (Real-world use case)
# =============================================================================


class TestWhereHdiffPattern:
    """Tests for np.where with hdiff-like patterns (stencil computations)."""

    def test_1d_forward_diff_clipped(self):
        """1D forward difference with positive values clipped."""

        @native
        def hdiff_1d(a):
            left = a[:-1]
            right = a[1:]
            diff = right - left
            return np.where(diff > 0, diff, 0.0)

        a = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        compiled = hdiff_1d.compile(a)
        result = compiled(a)
        # diff = [2, -1, 3, -1] -> [2, 0, 3, 0]
        expected = np.array([2.0, 0.0, 3.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_replace_positive_with_zero(self):
        """Replace positive differences with zero (inverse hdiff)."""

        @native
        def hdiff_inv(a):
            left = a[:-1]
            right = a[1:]
            diff = right - left
            return np.where(diff > 0, 0.0, diff)

        a = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        compiled = hdiff_inv.compile(a)
        result = compiled(a)
        # diff = [2, -1, 3, -1] -> [0, -1, 0, -1]
        expected = np.array([0.0, -1.0, 0.0, -1.0])
        np.testing.assert_array_equal(result, expected)

    def test_2d_center_slice(self):
        """2D center slice with where."""

        @native
        def where_2d_center(a):
            center = a[1:-1, 1:-1]
            return np.where(center > 0, center, 0.0)

        a = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, -8.0, 9.0]])
        compiled = where_2d_center.compile(a)
        result = compiled(a)
        expected = np.array([[5.0]])
        np.testing.assert_array_equal(result, expected)


# =============================================================================
# 10. CHAINED OPERATIONS
# =============================================================================


class TestWhereChainedOperations:
    """Tests for np.where result used in further operations."""

    def test_where_then_multiply(self):
        """np.where result multiplied by scalar."""

        @native
        def where_then_mul(arr):
            clipped = np.where(arr > 0, 0.0, arr)
            return clipped * 2.0

        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        compiled = where_then_mul.compile(arr)
        result = compiled(arr)
        expected = np.where(arr > 0, 0.0, arr) * 2.0
        np.testing.assert_array_equal(result, expected)

    def test_where_then_sum(self):
        """np.where result summed."""

        @native
        def where_then_sum(arr):
            clipped = np.where(arr > 0, arr, 0.0)
            return np.sum(clipped)

        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0])
        compiled = where_then_sum.compile(arr)
        result = compiled(arr)
        expected = np.sum(np.where(arr > 0, arr, 0.0))
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
