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
# Full-slice expression tests (path[:] in expressions)
# =============================================================================


class TestFullSliceExpressions:
    """Tests for full-slice array access in expressions."""

    def test_full_slice_in_minimum(self):
        """Test np.minimum with full slice argument."""

        @native
        def full_slice_minimum(
            a: float64["n", "n"], b: float64["n", "n"]
        ) -> float64["n", "n"]:
            return np.minimum(a[:], b[:])

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        b = np.random.rand(n, n).astype(np.float64)

        result = full_slice_minimum(a, b)
        expected = np.minimum(a, b)

        assert np.allclose(result, expected)

    def test_full_slice_in_maximum(self):
        """Test np.maximum with full slice argument."""

        @native
        def full_slice_maximum(
            a: float64["n", "n"], b: float64["n", "n"]
        ) -> float64["n", "n"]:
            return np.maximum(a[:], b[:])

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        b = np.random.rand(n, n).astype(np.float64)

        result = full_slice_maximum(a, b)
        expected = np.maximum(a, b)

        assert np.allclose(result, expected)

    def test_full_slice_in_add(self):
        """Test addition with full slice arguments."""

        @native
        def full_slice_add(
            a: float64["n", "n"], b: float64["n", "n"]
        ) -> float64["n", "n"]:
            return a[:] + b[:]

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        b = np.random.rand(n, n).astype(np.float64)

        result = full_slice_add(a, b)
        expected = a + b

        assert np.allclose(result, expected)

    def test_full_slice_1d(self):
        """Test full slice on 1D array."""

        @native
        def full_slice_1d(a: float64["n"], b: float64["n"]) -> float64["n"]:
            return np.minimum(a[:], b[:])

        n = 10
        a = np.random.rand(n).astype(np.float64)
        b = np.random.rand(n).astype(np.float64)

        result = full_slice_1d(a, b)
        expected = np.minimum(a, b)

        assert np.allclose(result, expected)


# =============================================================================
# Slice assignment with ufunc outer tests
# =============================================================================


class TestSliceAssignmentWithUfuncOuter:
    """Tests for slice assignment where RHS contains ufunc outer."""

    @pytest.mark.skip(
        reason="Direct ufunc outer assignment without wrapping operation not yet supported"
    )
    def test_basic_add_outer_assignment(self):
        """Test basic slice assignment with np.add.outer."""

        @native
        def add_outer_assign(a: float64["n", "n"]) -> float64["n", "n"]:
            a[:] = np.add.outer(a[:, 0], a[0, :])
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = add_outer_assign(a)
        expected = np.add.outer(a_copy[:, 0], a_copy[0, :])

        assert np.allclose(result, expected)

    def test_minimum_with_add_outer(self):
        """Test np.minimum wrapping np.add.outer in slice assignment."""

        @native
        def minimum_add_outer(a: float64["n", "n"]) -> float64["n", "n"]:
            a[:] = np.minimum(a[:], np.add.outer(a[:, 0], a[0, :]))
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = minimum_add_outer(a)
        expected = np.minimum(a_copy, np.add.outer(a_copy[:, 0], a_copy[0, :]))

        assert np.allclose(result, expected)

    def test_maximum_with_add_outer(self):
        """Test np.maximum wrapping np.add.outer in slice assignment."""

        @native
        def maximum_add_outer(a: float64["n", "n"]) -> float64["n", "n"]:
            a[:] = np.maximum(a[:], np.add.outer(a[:, 0], a[0, :]))
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = maximum_add_outer(a)
        expected = np.maximum(a_copy, np.add.outer(a_copy[:, 0], a_copy[0, :]))

        assert np.allclose(result, expected)

    def test_minimum_with_subtract_outer(self):
        """Test np.minimum wrapping np.subtract.outer."""

        @native
        def minimum_sub_outer(a: float64["n", "n"]) -> float64["n", "n"]:
            a[:] = np.minimum(a[:], np.subtract.outer(a[:, 0], a[0, :]))
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = minimum_sub_outer(a)
        expected = np.minimum(a_copy, np.subtract.outer(a_copy[:, 0], a_copy[0, :]))

        assert np.allclose(result, expected)

    @pytest.mark.xfail(reason="np.minimum.outer not yet supported")
    def test_minimum_with_minimum_outer(self):
        """Test np.minimum wrapping np.minimum.outer."""

        @native
        def minimum_min_outer(a: float64["n", "n"]) -> float64["n", "n"]:
            a[:] = np.minimum(a[:], np.minimum.outer(a[:, 0], a[0, :]))
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = minimum_min_outer(a)
        expected = np.minimum(a_copy, np.minimum.outer(a_copy[:, 0], a_copy[0, :]))

        assert np.allclose(result, expected)


# =============================================================================
# Floyd-Warshall style loop tests
# =============================================================================


class TestFloydWarshallPattern:
    """Tests for Floyd-Warshall style patterns with ufunc outer in loops."""

    def test_floyd_warshall_single_iteration(self):
        """Test single iteration of Floyd-Warshall pattern."""

        @native
        def floyd_single_iter(path: float64["n", "n"]) -> float64["n", "n"]:
            k = 0
            path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
            return path

        n = 4
        path = np.random.rand(n, n).astype(np.float64) * 10
        np.fill_diagonal(path, 0)
        path_copy = path.copy()

        result = floyd_single_iter(path)
        expected = np.minimum(path_copy, np.add.outer(path_copy[:, 0], path_copy[0, :]))

        assert np.allclose(result, expected)

    def test_floyd_warshall_full(self):
        """Test full Floyd-Warshall algorithm."""

        @native
        def floyd_warshall(path: float64["n", "n"]) -> float64["n", "n"]:
            for k in range(path.shape[0]):
                path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
            return path

        def floyd_warshall_numpy(path):
            result = path.copy()
            for k in range(path.shape[0]):
                result = np.minimum(result, np.add.outer(result[:, k], result[k, :]))
            return result

        n = 5
        path = np.random.rand(n, n).astype(np.float64) * 10
        np.fill_diagonal(path, 0)

        result = floyd_warshall(path.copy())
        expected = floyd_warshall_numpy(path.copy())

        assert np.allclose(result, expected)

    def test_floyd_warshall_different_sizes(self):
        """Test Floyd-Warshall with different matrix sizes."""

        @native
        def floyd_warshall(path: float64["n", "n"]) -> float64["n", "n"]:
            for k in range(path.shape[0]):
                path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
            return path

        def floyd_warshall_numpy(path):
            result = path.copy()
            for k in range(path.shape[0]):
                result = np.minimum(result, np.add.outer(result[:, k], result[k, :]))
            return result

        for n in [3, 4, 6, 8]:
            path = np.random.rand(n, n).astype(np.float64) * 10
            np.fill_diagonal(path, 0)

            result = floyd_warshall(path.copy())
            expected = floyd_warshall_numpy(path.copy())

            assert np.allclose(result, expected), f"Failed for n={n}"

    def test_floyd_warshall_int64(self):
        """Test Floyd-Warshall with int64 dtype."""

        @native
        def floyd_warshall_int(path: int64["n", "n"]) -> int64["n", "n"]:
            for k in range(path.shape[0]):
                path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
            return path

        def floyd_warshall_numpy(path):
            result = path.copy()
            for k in range(path.shape[0]):
                result = np.minimum(result, np.add.outer(result[:, k], result[k, :]))
            return result

        n = 4
        path = np.random.randint(1, 100, size=(n, n)).astype(np.int64)
        np.fill_diagonal(path, 0)

        result = floyd_warshall_int(path.copy())
        expected = floyd_warshall_numpy(path.copy())

        assert np.array_equal(result, expected)


# =============================================================================
# Column and row slicing with ufunc outer
# =============================================================================


class TestColumnRowSlicing:
    """Tests for column and row slicing combined with ufunc outer."""

    @pytest.mark.xfail(reason="Returning ufunc outer result directly not yet supported")
    def test_column_slice_first_dim(self):
        """Test slicing first column with add outer."""

        @native
        def col_slice_outer(a: float64["m", "n"]) -> float64["m", "n"]:
            result = np.add.outer(a[:, 0], a[0, :])
            return result

        m, n = 5, 4
        a = np.random.rand(m, n).astype(np.float64)

        result = col_slice_outer(a)
        expected = np.add.outer(a[:, 0], a[0, :])

        assert np.allclose(result, expected)
        assert result.shape == (m, n)

    @pytest.mark.xfail(reason="Returning ufunc outer result directly not yet supported")
    def test_row_slice_last_dim(self):
        """Test slicing last row with add outer."""

        @native
        def row_slice_outer(a: float64["m", "n"]) -> float64["n", "m"]:
            result = np.add.outer(a[-1, :], a[:, -1])
            return result

        m, n = 5, 4
        a = np.random.rand(m, n).astype(np.float64)

        result = row_slice_outer(a)
        expected = np.add.outer(a[-1, :], a[:, -1])

        assert np.allclose(result, expected)

    @pytest.mark.xfail(
        reason="Complex loop accumulation with ufunc outer not yet supported"
    )
    def test_variable_index_column(self):
        """Test column slicing with variable index in loop."""

        @native
        def var_col_outer(a: float64["n", "n"]) -> float64["n", "n"]:
            result = np.zeros((a.shape[0], a.shape[0]), dtype=np.float64)
            for k in range(a.shape[0]):
                result[:] = result[:] + np.add.outer(a[:, k], a[k, :])
            return result

        n = 4
        a = np.random.rand(n, n).astype(np.float64)

        result = var_col_outer(a)

        expected = np.zeros((n, n), dtype=np.float64)
        for k in range(n):
            expected = expected + np.add.outer(a[:, k], a[k, :])

        assert np.allclose(result, expected)


# =============================================================================
# Edge cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in ufunc outer slice assignments."""

    def test_small_matrix_2x2(self):
        """Test with minimal 2x2 matrix."""

        @native
        def small_floyd(path: float64["n", "n"]) -> float64["n", "n"]:
            for k in range(path.shape[0]):
                path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
            return path

        path = np.array([[0, 5], [3, 0]], dtype=np.float64)
        result = small_floyd(path.copy())

        # Manual Floyd-Warshall
        expected = path.copy()
        for k in range(2):
            expected = np.minimum(
                expected, np.add.outer(expected[:, k], expected[k, :])
            )

        assert np.allclose(result, expected)

    def test_single_element_matrix(self):
        """Test with 1x1 matrix."""

        @native
        def single_element(path: float64["n", "n"]) -> float64["n", "n"]:
            path[:] = np.minimum(path[:], np.add.outer(path[:, 0], path[0, :]))
            return path

        path = np.array([[5.0]], dtype=np.float64)
        result = single_element(path.copy())
        expected = np.minimum(path, np.add.outer(path[:, 0], path[0, :]))

        assert np.allclose(result, expected)

    def test_zeros_matrix(self):
        """Test with all zeros matrix."""

        @native
        def zeros_floyd(path: float64["n", "n"]) -> float64["n", "n"]:
            path[:] = np.minimum(path[:], np.add.outer(path[:, 0], path[0, :]))
            return path

        n = 4
        path = np.zeros((n, n), dtype=np.float64)
        result = zeros_floyd(path.copy())
        expected = np.minimum(path, np.add.outer(path[:, 0], path[0, :]))

        assert np.allclose(result, expected)

    def test_negative_values(self):
        """Test with negative values."""

        @native
        def negative_floyd(path: float64["n", "n"]) -> float64["n", "n"]:
            path[:] = np.minimum(path[:], np.add.outer(path[:, 0], path[0, :]))
            return path

        n = 4
        path = np.random.rand(n, n).astype(np.float64) * 10 - 5  # Range: [-5, 5]
        result = negative_floyd(path.copy())
        expected = np.minimum(path, np.add.outer(path[:, 0], path[0, :]))

        assert np.allclose(result, expected)

    def test_inf_values(self):
        """Test with infinity values (common in distance matrices)."""

        @native
        def inf_floyd(path: float64["n", "n"]) -> float64["n", "n"]:
            for k in range(path.shape[0]):
                path[:] = np.minimum(path[:], np.add.outer(path[:, k], path[k, :]))
            return path

        n = 4
        path = np.full((n, n), np.inf, dtype=np.float64)
        np.fill_diagonal(path, 0)
        # Add some finite edges
        path[0, 1] = 1
        path[1, 2] = 2
        path[2, 3] = 3

        result = inf_floyd(path.copy())

        expected = path.copy()
        for k in range(n):
            expected = np.minimum(
                expected, np.add.outer(expected[:, k], expected[k, :])
            )

        assert np.allclose(result, expected)


# =============================================================================
# Different operations combining
# =============================================================================


class TestCombinedOperations:
    """Tests for different operation combinations with ufunc outer."""

    def test_add_after_minimum_outer(self):
        """Test addition after minimum with outer."""

        @native
        def add_after_min(
            a: float64["n", "n"], b: float64["n", "n"]
        ) -> float64["n", "n"]:
            a[:] = np.minimum(a[:], np.add.outer(a[:, 0], a[0, :])) + b[:]
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        b = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = add_after_min(a, b)
        expected = np.minimum(a_copy, np.add.outer(a_copy[:, 0], a_copy[0, :])) + b

        assert np.allclose(result, expected)

    def test_multiply_after_outer(self):
        """Test multiplication after outer operation."""

        @native
        def mul_after_outer(
            a: float64["n", "n"], scale: float64["n", "n"]
        ) -> float64["n", "n"]:
            a[:] = np.add.outer(a[:, 0], a[0, :]) * scale[:]
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        scale = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = mul_after_outer(a, scale)
        expected = np.add.outer(a_copy[:, 0], a_copy[0, :]) * scale

        assert np.allclose(result, expected)

    def test_chain_minimum_operations(self):
        """Test chaining multiple minimum operations."""

        @native
        def chain_minimum(
            a: float64["n", "n"], b: float64["n", "n"]
        ) -> float64["n", "n"]:
            a[:] = np.minimum(np.minimum(a[:], b[:]), np.add.outer(a[:, 0], a[0, :]))
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        b = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = chain_minimum(a, b)
        expected = np.minimum(
            np.minimum(a_copy, b), np.add.outer(a_copy[:, 0], a_copy[0, :])
        )

        assert np.allclose(result, expected)


# =============================================================================
# Multiple ufunc outer in same function
# =============================================================================


class TestMultipleUfuncOuter:
    """Tests for multiple ufunc outer operations."""

    @pytest.mark.xfail(
        reason="Multiple ufunc outer in single expression not yet supported"
    )
    def test_two_outer_sum(self):
        """Test sum of two outer products."""

        @native
        def two_outer_sum(a: float64["n", "n"]) -> float64["n", "n"]:
            result = np.add.outer(a[:, 0], a[0, :]) + np.add.outer(a[:, 1], a[1, :])
            return result

        n = 4
        a = np.random.rand(n, n).astype(np.float64)

        result = two_outer_sum(a)
        expected = np.add.outer(a[:, 0], a[0, :]) + np.add.outer(a[:, 1], a[1, :])

        assert np.allclose(result, expected)

    def test_sequential_slice_assignments(self):
        """Test sequential slice assignments with outer."""

        @native
        def sequential_outer(a: float64["n", "n"]) -> float64["n", "n"]:
            a[:] = np.minimum(a[:], np.add.outer(a[:, 0], a[0, :]))
            a[:] = np.minimum(a[:], np.add.outer(a[:, 1], a[1, :]))
            return a

        n = 4
        a = np.random.rand(n, n).astype(np.float64)
        a_copy = a.copy()

        result = sequential_outer(a)

        expected = np.minimum(a_copy, np.add.outer(a_copy[:, 0], a_copy[0, :]))
        expected = np.minimum(expected, np.add.outer(expected[:, 1], expected[1, :]))

        assert np.allclose(result, expected)
