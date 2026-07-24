import numpy as np
import pytest
from docc.python import native


def test_outer():
    """Basic outer product with float64 arrays."""

    @native
    def np_outer(a, b):
        return np.outer(a, b)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = np_outer(a, b)
    expected = np.outer(a, b)
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"


def test_outer_float32():
    """Outer product with float32 arrays."""

    @native
    def outer_f32(a, b):
        return np.outer(a, b)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    res = outer_f32(a, b)
    assert res.dtype == np.float32, f"Expected dtype float32, got {res.dtype}"
    assert np.allclose(res, np.outer(a, b))


def test_outer_slicing():
    """Outer product with sliced arrays."""

    @native
    def outer_slice(a, b):
        return np.outer(a[:10], b[10:])

    a = np.random.rand(20)
    b = np.random.rand(20)
    res = outer_slice(a, b)
    assert np.allclose(res, np.outer(a[:10], b[10:]))


def test_outer_single_element_a():
    """Outer with single element in first array."""

    @native
    def outer_single_a(a, b):
        return np.outer(a, b)

    a = np.array([3.0])
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    res = outer_single_a(a, b)
    assert np.allclose(res, np.outer(a, b))


def test_outer_single_element_b():
    """Outer with single element in second array."""

    @native
    def outer_single_b(a, b):
        return np.outer(a, b)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([3.0])
    res = outer_single_b(a, b)
    assert np.allclose(res, np.outer(a, b))


def test_outer_single_element_both():
    """Outer with single element in both arrays."""

    @native
    def outer_single_both(a, b):
        return np.outer(a, b)

    a = np.array([3.0])
    b = np.array([4.0])
    res = outer_single_both(a, b)
    assert np.allclose(res, np.outer(a, b))


def test_add_outer():
    """Basic add.outer with float64 arrays."""

    @native
    def np_add_outer(a, b):
        return np.add.outer(a, b)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    res = np_add_outer(a, b)
    expected = np.add.outer(a, b)
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"


def test_add_outer_int64():
    """Add outer with int64 arrays."""

    @native
    def np_add_outer_int64(a, b):
        return np.add.outer(a, b)

    a = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    b = np.array([10, 20, 30, 40], dtype=np.int64)
    res = np_add_outer_int64(a, b)
    expected = np.add.outer(a, b)
    assert res.dtype == np.int64, f"Expected dtype int64, got {res.dtype}"
    assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


def test_add_outer_int32():
    """Add outer with int32 arrays."""

    @native
    def add_outer_i32(a, b):
        return np.add.outer(a, b)

    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    b = np.array([10, 20, 30, 40], dtype=np.int32)
    res = add_outer_i32(a, b)
    expected = np.add.outer(a, b)
    assert res.dtype == np.int32, f"Expected dtype int32, got {res.dtype}"
    assert np.array_equal(res, expected)


def test_subtract_outer():
    """Basic subtract.outer with float64 arrays."""

    @native
    def np_sub_outer(a, b):
        return np.subtract.outer(a, b)

    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = np_sub_outer(a, b)
    expected = np.subtract.outer(a, b)
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"


def test_subtract_outer_int64():
    """Subtract outer with int64 arrays."""

    @native
    def np_sub_outer_int64(a, b):
        return np.subtract.outer(a, b)

    a = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    b = np.array([1, 2, 3, 4], dtype=np.int64)
    res = np_sub_outer_int64(a, b)
    expected = np.subtract.outer(a, b)
    assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


def test_multiply_outer():
    """Basic multiply.outer with float64 arrays."""

    @native
    def np_mul_outer(a, b):
        return np.multiply.outer(a, b)

    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    res = np_mul_outer(a, b)
    expected = np.multiply.outer(a, b)
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"


def test_divide_outer():
    """Basic divide.outer with float64 arrays."""

    @native
    def np_div_outer(a, b):
        return np.divide.outer(a, b)

    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    b = np.array([1.0, 2.0, 5.0, 10.0])
    res = np_div_outer(a, b)
    expected = np.divide.outer(a, b)
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"


def test_minimum_outer_basic_float64():
    """Basic minimum.outer with float64 arrays."""

    @native
    def np_min_outer_basic(a, b):
        return np.minimum.outer(a, b)

    a = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
    b = np.array([4.0, 2.0, 6.0, 1.0])
    res = np_min_outer_basic(a, b)
    expected = np.minimum.outer(a, b)
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"


def test_minimum_outer_int64():
    """Minimum outer with int64 arrays."""

    @native
    def np_min_outer_int64(a, b):
        return np.minimum.outer(a, b)

    a = np.array([10, 5, 15, 3, 8], dtype=np.int64)
    b = np.array([7, 12, 4, 9], dtype=np.int64)
    res = np_min_outer_int64(a, b)
    expected = np.minimum.outer(a, b)
    assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


def test_maximum_outer_basic_float64():
    """Basic maximum.outer with float64 arrays."""

    @native
    def np_max_outer_basic(a, b):
        return np.maximum.outer(a, b)

    a = np.array([1.0, 5.0, 3.0, 7.0, 2.0])
    b = np.array([4.0, 2.0, 6.0, 1.0])
    res = np_max_outer_basic(a, b)
    expected = np.maximum.outer(a, b)
    assert np.allclose(res, expected), f"Expected {expected}, got {res}"


def test_maximum_outer_int64():
    """Maximum outer with int64 arrays."""

    @native
    def np_max_outer_int64(a, b):
        return np.maximum.outer(a, b)

    a = np.array([10, 5, 15, 3, 8], dtype=np.int64)
    b = np.array([7, 12, 4, 9], dtype=np.int64)
    res = np_max_outer_int64(a, b)
    expected = np.maximum.outer(a, b)
    assert np.array_equal(res, expected), f"Expected {expected}, got {res}"


class TestOuterCombined:
    """Tests combining outer operations with other operations."""

    def test_outer_plus_outer(self):
        """Sum of two outer products."""

        @native
        def outer_plus_outer(a, b, c, d):
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
        def outer_acc(a, b, C):
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
        def add_outer_sum(a, b) -> float:
            # Note: Direct chaining works, but intermediate variable may have issues
            return np.sum(np.add.outer(a, b))

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = add_outer_sum(a, b)
        expected = np.sum(np.add.outer(a, b))
        assert np.isclose(res, expected)


class TestFullSliceExpressions:
    """Tests for full-slice array access in expressions."""

    def test_full_slice_in_minimum(self):
        """Test np.minimum with full slice argument."""

        @native
        def full_slice_minimum(a, b):
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
        def full_slice_maximum(a, b):
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
        def full_slice_add(a, b):
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
        def full_slice_1d(a, b):
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

    def test_basic_add_outer_assignment(self):
        """Test basic slice assignment with np.add.outer."""

        @native
        def add_outer_assign(a):
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
        def minimum_add_outer(a):
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
        def maximum_add_outer(a):
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
        def minimum_sub_outer(a):
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
        def minimum_min_outer(a):
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
        def floyd_single_iter(path):
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
        def floyd_warshall(path):
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
        def floyd_warshall(path):
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
        def floyd_warshall_int(path):
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

    def test_column_slice_first_dim(self):
        """Test slicing first column with add outer."""

        @native
        def col_slice_outer(a):
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
        def row_slice_outer(a):
            result = np.add.outer(a[-1, :], a[:, -1])
            return result

        m, n = 5, 4
        a = np.random.rand(m, n).astype(np.float64)

        result = row_slice_outer(a)
        expected = np.add.outer(a[-1, :], a[:, -1])

        assert np.allclose(result, expected)

    def test_variable_index_column(self):
        """Test column slicing with variable index in loop."""

        @native
        def var_col_outer(a):
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
        def small_floyd(path):
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
        def single_element(path):
            path[:] = np.minimum(path[:], np.add.outer(path[:, 0], path[0, :]))
            return path

        path = np.array([[5.0]], dtype=np.float64)
        result = single_element(path.copy())
        expected = np.minimum(path, np.add.outer(path[:, 0], path[0, :]))

        assert np.allclose(result, expected)

    def test_zeros_matrix(self):
        """Test with all zeros matrix."""

        @native
        def zeros_floyd(path):
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
        def negative_floyd(path):
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
        def inf_floyd(path):
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
        def add_after_min(a, b):
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
        def mul_after_outer(a, scale):
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
        def chain_minimum(a, b):
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

    def test_two_outer_sum(self):
        """Test sum of two outer products."""

        @native
        def two_outer_sum(a):
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
        def sequential_outer(a):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
