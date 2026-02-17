from docc.python import native
import os
import shutil
import pytest
import numpy as np


def test_array_loop():
    @native
    def array_loop(A, n):
        for i in range(n):
            A[i] = A[i] + 1

    arr = np.zeros(10, dtype=np.int32)
    array_loop(arr, 10)

    expected = np.ones(10, dtype=np.int32)
    np.testing.assert_array_equal(arr, expected)


def test_negative_step_loop_array():
    @native
    def negative_step_loop_array(n):
        A = np.zeros((n,), dtype=np.int64)
        for i in range(n - 1, -1, -1):
            A[i] = i
        return A

    res = negative_step_loop_array(5)
    np.testing.assert_array_equal(res, [0, 1, 2, 3, 4])


def test_slice_assignment_loop_var():
    @native
    def slice_assignment(n):
        A = np.zeros((n, n), dtype=np.float64)
        B = np.ones((n, n), dtype=np.float64)

        # Pattern from ADI: A[row, slice] = B[row, slice]
        for i in range(n):
            A[i, 1 : n - 1] = B[i, 1 : n - 1]

        return A

    res = slice_assignment(10)
    expected = np.zeros((10, 10))
    expected[:, 1:9] = 1.0
    np.testing.assert_array_equal(res, expected)


def test_constant_index_with_slice():
    """Test pattern: A[n-1, 1:n-1] = value (constant index before slice)"""

    @native
    def constant_index_slice(n):
        A = np.zeros((n, n), dtype=np.float64)
        # Set last row's middle elements to 1.0
        A[n - 1, 1 : n - 1] = 1.0
        return A

    res = constant_index_slice(10)
    expected = np.zeros((10, 10))
    expected[9, 1:9] = 1.0
    np.testing.assert_array_equal(res, expected)


def test_constant_index_with_slice_in_loop():
    """Test pattern from ADI: v[N-1, 1:N-1] = 1.0 inside a loop"""

    @native
    def constant_index_in_loop(n, steps):
        A = np.zeros((n, n), dtype=np.float64)
        for t in range(steps):
            # First and last row, middle columns
            A[0, 1 : n - 1] = 1.0
            A[n - 1, 1 : n - 1] = 2.0
        return A

    res = constant_index_in_loop(10, 3)
    expected = np.zeros((10, 10))
    expected[0, 1:9] = 1.0
    expected[9, 1:9] = 2.0
    np.testing.assert_array_equal(res, expected)


def test_slice_bounds_depend_on_loop_var():
    """Test where slice bounds depend on the loop variable"""

    @native
    def slice_depends_on_loop(n):
        A = np.zeros((n, n), dtype=np.float64)
        # Each row i gets values set from column 0 to i
        for i in range(1, n):
            A[i, 0:i] = float(i)
        return A

    res = slice_depends_on_loop(5)
    expected = np.zeros((5, 5))
    for i in range(1, 5):
        expected[i, 0:i] = float(i)
    np.testing.assert_array_equal(res, expected)


def test_nested_loop_with_slice():
    """Test nested loops with slice assignment"""

    @native
    def nested_loop_slice(n, m):
        A = np.zeros((n, m), dtype=np.float64)
        for i in range(n):
            for j in range(1, m - 1):
                A[i, j : j + 1] = float(i + j)
        return A

    res = nested_loop_slice(4, 6)
    expected = np.zeros((4, 6))
    for i in range(4):
        for j in range(1, 5):
            expected[i, j : j + 1] = float(i + j)
    np.testing.assert_array_equal(res, expected)


def test_reverse_loop_with_slice():
    """Test reverse loop with slice assignment (ADI pattern)"""

    @native
    def reverse_loop_slice(n):
        A = np.zeros((n, n), dtype=np.float64)
        B = np.ones((n, n), dtype=np.float64)

        # Forward pass: sets rows 1 to n-2 (middle rows)
        for j in range(n):
            A[1 : n - 1, j] = B[1 : n - 1, j]

        # Backward pass (like ADI's v[j, 1:N-1] = ... for j in range(N-2, 0, -1))
        for j in range(n - 2, 0, -1):
            A[j, 1 : n - 1] = A[j, 1 : n - 1] + 1.0

        return A

    res = reverse_loop_slice(6)
    expected = np.zeros((6, 6))
    # Forward pass sets rows 1-4 to 1.0
    expected[1:5, :] = 1.0
    # Backward pass adds 1.0 to rows 1-4, columns 1-4
    for j in range(4, 0, -1):
        expected[j, 1:5] += 1.0
    np.testing.assert_array_equal(res, expected)


def test_slice_with_expression_index():
    """Test slice combined with expression index (j+1, N-j, etc.)"""

    @native
    def expr_index_slice(n):
        A = np.zeros((n, n), dtype=np.float64)
        for j in range(n - 2):
            # Access row (n - 2 - j), similar to ADI pattern
            A[n - 2 - j, 1 : n - 1] = float(j + 1)
        return A

    res = expr_index_slice(6)
    expected = np.zeros((6, 6))
    for j in range(4):
        expected[4 - j, 1:5] = float(j + 1)
    np.testing.assert_array_equal(res, expected)


def test_multiple_slice_assignments_in_loop():
    """Test multiple slice assignments in same loop iteration"""

    @native
    def multi_slice_loop(n):
        A = np.zeros((n, n), dtype=np.float64)
        B = np.zeros((n, n), dtype=np.float64)

        for t in range(3):
            # Multiple slice assignments like in ADI
            A[0, 1 : n - 1] = 1.0
            B[1 : n - 1, 0] = 2.0
            A[n - 1, 1 : n - 1] = 3.0
            B[1 : n - 1, n - 1] = 4.0

        return A, B

    a, b = multi_slice_loop(8)

    expected_a = np.zeros((8, 8))
    expected_a[0, 1:7] = 1.0
    expected_a[7, 1:7] = 3.0

    expected_b = np.zeros((8, 8))
    expected_b[1:7, 0] = 2.0
    expected_b[1:7, 7] = 4.0

    np.testing.assert_array_equal(a, expected_a)
    np.testing.assert_array_equal(b, expected_b)


# def test_reverse_loop_dependency():
#     @native
#     def reverse_dep(n):
#         A = np.zeros(n, dtype=np.float64)
#         # Init A with 1.0 at end
#         A[n - 1] = 1.0

#         # Propagate backwards: A[i] = A[i+1]
#         for i in range(n - 2, -1, -1):
#             A[i] = A[i + 1]

#         return A

#     res = reverse_dep(10)
#     expected = np.ones(10)
#     np.testing.assert_array_equal(res, expected)
