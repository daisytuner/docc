import numpy as np
from docc.python import native


class TestSliceRewriterSubscript:
    """Tests for SliceRewriter handling of subscript expressions."""

    def test_array_element_to_1d_slice(self):
        """Test assigning a scalar array element to a 1D slice: arr[:] = src[0]"""

        @native
        def array_element_to_1d_slice(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[0]

        src = np.array([42.0, 1.0, 2.0], dtype=np.float64)
        dst = np.zeros(5, dtype=np.float64)
        array_element_to_1d_slice(src, dst)
        # All elements should be 42.0
        np.testing.assert_array_equal(dst, np.full(5, 42.0))

    def test_array_element_to_2d_row_slice(self):
        """Test assigning 1D array element to a row of 2D array: ey[0, :] = _fict_[0]"""

        @native
        def array_element_to_2d_row_slice(_fict_: np.ndarray, ey: np.ndarray):
            ey[0, :] = _fict_[0]

        _fict_ = np.array([99.0, 1.0, 2.0], dtype=np.float64)
        ey = np.zeros((3, 4), dtype=np.float64)
        array_element_to_2d_row_slice(_fict_, ey)
        # First row should be all 99.0
        np.testing.assert_array_equal(ey[0, :], np.full(4, 99.0))
        # Other rows unchanged
        np.testing.assert_array_equal(ey[1:, :], np.zeros((2, 4)))

    def test_array_element_to_2d_col_slice(self):
        """Test assigning array element to a column: arr[:, 0] = src[1]"""

        @native
        def array_element_to_2d_col_slice(src: np.ndarray, dst: np.ndarray):
            dst[:, 0] = src[1]

        src = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        dst = np.zeros((4, 3), dtype=np.float64)
        array_element_to_2d_col_slice(src, dst)
        # First column should be all 20.0
        np.testing.assert_array_equal(dst[:, 0], np.full(4, 20.0))
        # Other columns unchanged
        np.testing.assert_array_equal(dst[:, 1:], np.zeros((4, 2)))

    def test_scalar_to_slice(self):
        """Test assigning a scalar variable to a slice."""

        @native
        def scalar_to_slice(val: float, dst: np.ndarray):
            dst[:] = val

        dst = np.zeros(5, dtype=np.float64)
        scalar_to_slice(3.14, dst)
        np.testing.assert_array_almost_equal(dst, np.full(5, 3.14))

    def test_constant_to_slice(self):
        """Test assigning a constant to a slice."""

        @native
        def constant_to_slice(dst: np.ndarray):
            dst[:] = 2.5

        dst = np.zeros(4, dtype=np.float64)
        constant_to_slice(dst)
        np.testing.assert_array_almost_equal(dst, np.full(4, 2.5))

    def test_indexed_expr_to_slice(self):
        """Test assigning indexed expression to slice: dst[:] = src[i] where i is variable."""

        @native
        def indexed_expr_to_slice(src: np.ndarray, idx: int, dst: np.ndarray):
            dst[:] = src[idx]

        src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        dst = np.zeros(5, dtype=np.float64)
        indexed_expr_to_slice(src, 2, dst)
        np.testing.assert_array_equal(dst, np.full(5, 3.0))

    def test_mixed_fixed_and_slice_indices(self):
        """Test 3D array with mixed fixed and slice indices."""

        @native
        def mixed_fixed_and_slice_indices(src: np.ndarray, dst: np.ndarray):
            # dst[0, :, 1] = scalar from src[0]
            dst[0, :, 1] = src[0]

        src = np.array([7.0, 8.0], dtype=np.float64)
        dst = np.zeros((2, 3, 2), dtype=np.float64)
        mixed_fixed_and_slice_indices(src, dst)
        # dst[0, :, 1] should all be 7.0
        np.testing.assert_array_equal(dst[0, :, 1], np.full(3, 7.0))

    def test_fdtd_2d_pattern(self):
        """Test the actual fdtd_2d pattern: ey[0, :] = _fict_[t] inside loop."""

        @native
        def fdtd_2d_pattern(TMAX: int, _fict_: np.ndarray, ey: np.ndarray):
            for t in range(TMAX):
                ey[0, :] = _fict_[t]

        TMAX = 3
        _fict_ = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ey = np.zeros((2, 4), dtype=np.float64)
        fdtd_2d_pattern(TMAX, _fict_, ey)
        # After loop, ey[0, :] should be _fict_[2] = 3.0
        np.testing.assert_array_equal(ey[0, :], np.full(4, 3.0))

    def test_slice_to_slice_same_size(self):
        """Test slice-to-slice assignment of same size (no broadcasting)."""

        @native
        def slice_to_slice_same_size(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[:]

        src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        dst = np.zeros(3, dtype=np.float64)
        slice_to_slice_same_size(src, dst)
        np.testing.assert_array_equal(dst, src)


class TestSliceRewriterNoTransform:
    """Tests verifying that fully-indexed arrays are NOT transformed."""

    def test_point_indexed_array_unchanged(self):
        """Verify that arr[i] (point index) is not transformed to arr[loop_var][i]."""

        @native
        def point_indexed_array_unchanged(src: np.ndarray, dst: np.ndarray):
            # src[0] should remain src[0], not become src[loop_var][0]
            dst[:] = src[0] + 1.0

        src = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        dst = np.zeros(4, dtype=np.float64)
        point_indexed_array_unchanged(src, dst)
        np.testing.assert_array_equal(dst, np.full(4, 11.0))

    def test_2d_point_indexed_unchanged(self):
        """Verify that arr[i, j] (2D point indices) is not transformed."""

        @native
        def point_indexed_2d_unchanged(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[1, 2]

        src = np.arange(12.0, dtype=np.float64).reshape(3, 4)
        dst = np.zeros(5, dtype=np.float64)
        point_indexed_2d_unchanged(src, dst)
        np.testing.assert_array_equal(dst, np.full(5, src[1, 2]))


class TestNegativeIndexHandling:
    """Tests for negative index normalization in slice assignments."""

    def test_negative_index_last_column(self):
        """Test assigning to last column using -1 index: arr[:, -1] = value."""

        @native
        def assign_last_column(dst: np.ndarray):
            dst[:, -1] = 0.0

        dst = np.ones((4, 5), dtype=np.float64)
        assign_last_column(dst)
        # Last column should be all 0.0
        np.testing.assert_array_equal(dst[:, -1], np.zeros(4))
        # Other columns unchanged
        np.testing.assert_array_equal(dst[:, :-1], np.ones((4, 4)))

    def test_negative_index_second_last_column(self):
        """Test assigning to second-to-last column using -2 index: arr[:, -2] = value."""

        @native
        def assign_second_last_column(dst: np.ndarray):
            dst[:, -2] = 5.0

        dst = np.zeros((3, 4), dtype=np.float64)
        assign_second_last_column(dst)
        # Second-to-last column should be all 5.0
        np.testing.assert_array_equal(dst[:, -2], np.full(3, 5.0))
        # Other columns unchanged
        np.testing.assert_array_equal(dst[:, -1], np.zeros(3))
        np.testing.assert_array_equal(dst[:, :-2], np.zeros((3, 2)))

    def test_negative_index_last_row(self):
        """Test assigning to last row using -1 index: arr[-1, :] = value."""

        @native
        def assign_last_row(dst: np.ndarray):
            dst[-1, :] = 3.0

        dst = np.zeros((4, 5), dtype=np.float64)
        assign_last_row(dst)
        # Last row should be all 3.0
        np.testing.assert_array_equal(dst[-1, :], np.full(5, 3.0))
        # Other rows unchanged
        np.testing.assert_array_equal(dst[:-1, :], np.zeros((3, 5)))

    def test_negative_index_second_last_row(self):
        """Test assigning to second-to-last row using -2 index: arr[-2, :] = value."""

        @native
        def assign_second_last_row(dst: np.ndarray):
            dst[-2, :] = 7.0

        dst = np.zeros((5, 3), dtype=np.float64)
        assign_second_last_row(dst)
        # Second-to-last row should be all 7.0
        np.testing.assert_array_equal(dst[-2, :], np.full(3, 7.0))
        # Other rows unchanged
        np.testing.assert_array_equal(dst[-1, :], np.zeros(3))
        np.testing.assert_array_equal(dst[:-2, :], np.zeros((3, 3)))

    def test_negative_index_with_expression(self):
        """Test assigning expression to slice with negative index."""

        @native
        def negative_index_with_expr(src: np.ndarray, dst: np.ndarray):
            dst[:, -1] = src[:, 0] * 2.0

        src = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        dst = np.zeros((3, 4), dtype=np.float64)
        negative_index_with_expr(src, dst)
        # Last column should be 2 * first column of src
        np.testing.assert_array_equal(dst[:, -1], np.array([2.0, 6.0, 10.0]))

    def test_negative_index_read_access(self):
        """Test reading from array with negative index: dst[:] = src[:, -1]."""

        @native
        def read_last_column(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[:, -1]

        src = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        dst = np.zeros(2, dtype=np.float64)
        read_last_column(src, dst)
        # Should get last column of src
        np.testing.assert_array_equal(dst, np.array([3.0, 6.0]))

    def test_negative_index_read_second_last(self):
        """Test reading from array with -2 index: dst[:] = src[:, -2]."""

        @native
        def read_second_last_column(src: np.ndarray, dst: np.ndarray):
            dst[:] = src[:, -2]

        src = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        dst = np.zeros(2, dtype=np.float64)
        read_second_last_column(src, dst)
        # Should get second-to-last column of src
        np.testing.assert_array_equal(dst, np.array([2.0, 5.0]))

    def test_negative_index_deriche_pattern(self):
        """Test the deriche benchmark pattern: y2[:, -1] = 0.0; y2[:, -2] = a3 * imgIn[:, -1]."""

        @native
        def deriche_pattern(imgIn: np.ndarray, a3: float):
            y2 = np.empty_like(imgIn)
            y2[:, -1] = 0.0
            y2[:, -2] = a3 * imgIn[:, -1]
            return y2

        imgIn = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
        a3 = 2.0
        y2 = deriche_pattern(imgIn, a3)

        # Check last column is 0.0
        np.testing.assert_array_equal(y2[:, -1], np.zeros(2))
        # Check second-to-last column is a3 * last column of imgIn
        np.testing.assert_array_equal(y2[:, -2], np.array([8.0, 16.0]))

    def test_negative_index_row_pattern(self):
        """Test row-based negative index pattern: y1[0, :] = ...; y2[-1, :] = 0.0."""

        @native
        def row_pattern(imgOut: np.ndarray, a5: float):
            y1 = np.empty_like(imgOut)
            y2 = np.empty_like(imgOut)
            y1[0, :] = a5 * imgOut[0, :]
            y2[-1, :] = 0.0
            y2[-2, :] = a5 * imgOut[-1, :]
            return y1, y2

        imgOut = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
        )
        a5 = 0.5
        y1, y2 = row_pattern(imgOut, a5)

        # Check y1[0, :] = a5 * imgOut[0, :]
        np.testing.assert_array_equal(y1[0, :], np.array([0.5, 1.0, 1.5]))
        # Check y2[-1, :] = 0.0
        np.testing.assert_array_equal(y2[-1, :], np.zeros(3))
        # Check y2[-2, :] = a5 * imgOut[-1, :]
        np.testing.assert_array_equal(y2[-2, :], np.array([3.5, 4.0, 4.5]))
