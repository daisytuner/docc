import pytest
import numpy as np
from docc.python import native


class TestSimpleGather:
    """Tests for basic gather operations: result = x[indices]"""

    def test_gather_1d_simple(self):
        """Test basic 1D gather: y = x[indices]"""

        @native
        def gather_simple(x, indices, y):
            for i in range(indices.shape[0]):
                y[i] = x[indices[i]]

        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        indices = np.array([2, 0, 4, 1], dtype=np.int64)
        y = np.zeros(4, dtype=np.float64)

        gather_simple(x, indices, y)
        np.testing.assert_array_equal(y, x[indices])

    def test_gather_1d_return(self):
        """Test gather with return value"""

        @native
        def gather_return(x, indices):
            result = np.empty(indices.shape[0], dtype=np.float64)
            for i in range(indices.shape[0]):
                result[i] = x[indices[i]]
            return result

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        indices = np.array([4, 2, 0], dtype=np.int64)

        result = gather_return(x, indices)
        np.testing.assert_array_equal(result, x[indices])

    def test_gather_int32_indices(self):
        """Test gather with int32 indices"""

        @native
        def gather_int32(x, indices, y):
            for i in range(indices.shape[0]):
                y[i] = x[indices[i]]

        x = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        indices = np.array([2, 0, 1, 0], dtype=np.int32)
        y = np.zeros(4, dtype=np.float64)

        gather_int32(x, indices, y)
        np.testing.assert_array_equal(y, x[indices])


class TestGatherInLoop:
    """Tests for gather operations inside loops with dynamic indices"""

    def test_gather_with_slice_indices(self):
        """Test gather where indices come from a slice (SpMV pattern)"""

        @native
        def gather_slice_indices(x, row_ptr, col_idx, result):
            for i in range(row_ptr.shape[0] - 1):
                # Get indices for this row
                start = row_ptr[i]
                end = row_ptr[i + 1]
                # Simple sum of gathered values
                s = 0.0
                for j in range(start, end):
                    s = s + x[col_idx[j]]
                result[i] = s

        # CSR-like structure: 3 rows with variable number of entries
        row_ptr = np.array([0, 2, 5, 7], dtype=np.int64)
        col_idx = np.array([0, 2, 1, 2, 3, 0, 3], dtype=np.int64)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = np.zeros(3, dtype=np.float64)

        gather_slice_indices(x, row_ptr, col_idx, result)

        # Expected: row 0: x[0]+x[2]=4, row 1: x[1]+x[2]+x[3]=9, row 2: x[0]+x[3]=5
        expected = np.array([4.0, 9.0, 5.0])
        np.testing.assert_array_equal(result, expected)

    def test_gather_indirect_slice_bounds(self):
        """Test slicing with indirect bounds: arr[row_ptr[i]:row_ptr[i+1]]"""

        @native
        def indirect_slice_sum(arr, row_ptr, result):
            for i in range(row_ptr.shape[0] - 1):
                vals = arr[row_ptr[i] : row_ptr[i + 1]]
                result[i] = np.sum(vals)

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        row_ptr = np.array([0, 2, 5, 6], dtype=np.int64)
        result = np.zeros(3, dtype=np.float64)

        indirect_slice_sum(arr, row_ptr, result)

        # Expected: sum([1,2])=3, sum([3,4,5])=12, sum([6])=6
        expected = np.array([3.0, 12.0, 6.0])
        np.testing.assert_array_equal(result, expected)


class TestGatherWithOperations:
    """Tests for gather combined with arithmetic operations"""

    def test_gather_scale(self):
        """Test gather followed by scaling: y = alpha * x[indices]"""

        @native
        def gather_scale(x, indices, alpha, y):
            for i in range(indices.shape[0]):
                y[i] = alpha * x[indices[i]]

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        indices = np.array([3, 1, 0], dtype=np.int64)
        y = np.zeros(3, dtype=np.float64)
        alpha = 2.5

        gather_scale(x, indices, alpha, y)
        np.testing.assert_array_equal(y, alpha * x[indices])

    def test_gather_add(self):
        """Test gather with addition: y = x[indices] + z[indices]"""

        @native
        def gather_add(x, z, indices, y):
            for i in range(indices.shape[0]):
                y[i] = x[indices[i]] + z[indices[i]]

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        z = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        indices = np.array([2, 0, 3], dtype=np.int64)
        y = np.zeros(3, dtype=np.float64)

        gather_add(x, z, indices, y)
        np.testing.assert_array_equal(y, x[indices] + z[indices])

    def test_gather_multiply_accumulate(self):
        """Test gather with multiply-accumulate (dot product pattern)"""

        @native
        def gather_mac(vals, x, indices):
            result = 0.0
            for i in range(indices.shape[0]):
                result = result + vals[i] * x[indices[i]]
            return result

        vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        indices = np.array([3, 0, 2], dtype=np.int64)

        result = gather_mac(vals, x, indices)
        expected = np.dot(vals, x[indices])  # 1*40 + 2*10 + 3*30 = 150
        assert result == expected


class TestGatherWithMatmul:
    """Tests for gather combined with matrix operations (SpMV-like)"""

    def test_gather_dot_product(self):
        """Test gather followed by dot product: y[i] = vals @ x[cols]"""

        @native
        def spmv_row(vals, x, cols):
            # Single row of SpMV
            gathered = np.empty(cols.shape[0], dtype=np.float64)
            for i in range(cols.shape[0]):
                gathered[i] = x[cols[i]]
            return vals @ gathered

        vals = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        x = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        cols = np.array([4, 1, 3], dtype=np.int64)

        result = spmv_row(vals, x, cols)
        expected = np.dot(vals, x[cols])  # 1*50 + 2*20 + 3*40 = 210
        np.testing.assert_almost_equal(result, expected)

    @pytest.mark.skip(
        reason="Full SpMV pattern with gather and matmul is not yet supported"
    )
    def test_full_spmv_pattern(self):
        """Test full SpMV kernel pattern with indirect slicing and gather"""

        @native
        def spmv_kernel(row_ptr, col_idx, values, x):
            y = np.empty(row_ptr.shape[0] - 1, dtype=np.float64)
            for i in range(row_ptr.shape[0] - 1):
                cols = col_idx[row_ptr[i] : row_ptr[i + 1]]
                vals = values[row_ptr[i] : row_ptr[i + 1]]
                y[i] = vals @ x[cols]
            return y

        # Simple 3x4 sparse matrix in CSR format
        # Row 0: [1, 0, 2, 0]  -> indices [0, 2], values [1, 2]
        # Row 1: [0, 3, 0, 4]  -> indices [1, 3], values [3, 4]
        # Row 2: [5, 0, 0, 6]  -> indices [0, 3], values [5, 6]
        row_ptr = np.array([0, 2, 4, 6], dtype=np.int64)
        col_idx = np.array([0, 2, 1, 3, 0, 3], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

        result = spmv_kernel(row_ptr, col_idx, values, x)

        # Expected: y = A @ x
        # y[0] = 1*1 + 2*3 = 7
        # y[1] = 3*2 + 4*4 = 22
        # y[2] = 5*1 + 6*4 = 29
        expected = np.array([7.0, 22.0, 29.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestScatter:
    """Tests for scatter operations (writing with indirect indices)"""

    def test_scatter_simple(self):
        """Test basic scatter: y[indices] = x"""

        @native
        def scatter_simple(x, indices, y):
            for i in range(indices.shape[0]):
                y[indices[i]] = x[i]

        x = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        indices = np.array([2, 0, 4], dtype=np.int64)
        y = np.zeros(5, dtype=np.float64)

        scatter_simple(x, indices, y)

        expected = np.zeros(5)
        expected[indices] = x
        np.testing.assert_array_equal(y, expected)

    def test_scatter_accumulate(self):
        """Test scatter with accumulation: y[indices[i]] += x[i]"""

        @native
        def scatter_accumulate(x, indices, y):
            for i in range(indices.shape[0]):
                y[indices[i]] = y[indices[i]] + x[i]

        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        indices = np.array([0, 1, 0, 1], dtype=np.int64)  # Duplicate indices
        y = np.zeros(3, dtype=np.float64)

        scatter_accumulate(x, indices, y)

        # y[0] = 1 + 3 = 4, y[1] = 2 + 4 = 6
        expected = np.array([4.0, 6.0, 0.0])
        np.testing.assert_array_equal(y, expected)


class TestGatherEdgeCases:
    """Edge cases and special scenarios for gather operations"""

    def test_gather_single_element(self):
        """Test gather with single element"""

        @native
        def gather_single(x, indices, y):
            for i in range(indices.shape[0]):
                y[i] = x[indices[i]]

        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        indices = np.array([1], dtype=np.int64)
        y = np.zeros(1, dtype=np.float64)

        gather_single(x, indices, y)
        np.testing.assert_array_equal(y, x[indices])

    def test_gather_all_same_index(self):
        """Test gather with all same indices (broadcast-like)"""

        @native
        def gather_broadcast(x, indices, y):
            for i in range(indices.shape[0]):
                y[i] = x[indices[i]]

        x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        indices = np.array([1, 1, 1, 1], dtype=np.int64)
        y = np.zeros(4, dtype=np.float64)

        gather_broadcast(x, indices, y)
        np.testing.assert_array_equal(y, np.array([2.0, 2.0, 2.0, 2.0]))

    def test_gather_reverse_order(self):
        """Test gather that reverses array order"""

        @native
        def gather_reverse(x, y):
            n = x.shape[0]
            for i in range(n):
                y[i] = x[n - 1 - i]

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        y = np.zeros(5, dtype=np.float64)

        gather_reverse(x, y)
        np.testing.assert_array_equal(y, x[::-1])
