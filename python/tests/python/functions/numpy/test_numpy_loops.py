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
