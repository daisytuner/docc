from docc.python import native
import pytest
import numpy as np


def test_slice_with_index_2d():
    """Test slicing with a mix of slice and index: arr[1:, k]"""

    @native
    def slice_with_index(arr, k):
        # arr is (N, M), arr[1:, k] gives a 1D array of size N-1
        result = np.zeros((arr.shape[0] - 1,), dtype=np.float64)
        result[:] = arr[1:, k]
        return result[0]

    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    # arr[1:, 2] = [7, 12, 17]
    assert slice_with_index(arr, 2) == 7.0


def test_slice_negative_bound():
    """Test slicing with negative upper bound: arr[:-1, k]"""

    @native
    def slice_negative(arr, k):
        # arr is (N, M), arr[:-1, k] gives a 1D array of size N-1
        result = np.zeros((arr.shape[0] - 1,), dtype=np.float64)
        result[:] = arr[:-1, k]
        return result[0]

    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    # arr[:-1, 2] = [2, 7, 12]
    assert slice_negative(arr, 2) == 2.0


def test_slice_in_expression_3d():
    """Test slicing in expression: arr[1:, :, k+1] + arr[:-1, :, k+1]"""

    @native
    def slice_expression(arr, k) -> float:
        # This is the pattern from vadv: wcon[1:, :, k+1] + wcon[:-1, :, k+1]
        # Use lowercase to avoid SymEngine interpreting I/J as imaginary
        ni = arr.shape[0] - 1  # Result is (ni, nj)
        nj = arr.shape[1]
        result = np.zeros((ni, nj), dtype=np.float64)
        result[:, :] = arr[1:, :, k + 1] + arr[:-1, :, k + 1]
        return result[0, 0]

    # arr shape is (I+1, J, K)
    arr = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    # arr[1:, :, 2] has shape (2, 4), values starting at arr[1, 0, 2] = 22
    # arr[:-1, :, 2] has shape (2, 4), values starting at arr[0, 0, 2] = 2
    # result[0, 0] = arr[1, 0, 2] + arr[0, 0, 2] = 22 + 2 = 24
    assert slice_expression(arr, 1) == 24.0


def test_full_slice_with_index():
    """Test full slice with index: arr[:, :, k]"""

    @native
    def full_slice_index(arr, k) -> float:
        # arr[:, :, k] extracts a 2D slice
        ni, nj = arr.shape[0], arr.shape[1]
        result = np.zeros((ni, nj), dtype=np.float64)
        result[:, :] = arr[:, :, k]
        return result[0, 0]

    arr = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    # arr[:, :, 2] gives shape (3, 4), arr[0, 0, 2] = 2
    assert full_slice_index(arr, 2) == 2.0


def test_slice_arithmetic():
    """Test arithmetic with sliced arrays: scalar * (slice + slice)"""

    @native
    def slice_arithmetic(arr, k) -> float:
        ni = arr.shape[0] - 1
        nj = arr.shape[1]
        result = np.zeros((ni, nj), dtype=np.float64)
        # Directly assign the arithmetic expression instead of using intermediate variable
        result[:, :] = 0.25 * (arr[1:, :, k + 1] + arr[:-1, :, k + 1])
        return result[0, 0]

    arr = np.ones((3, 4, 5), dtype=np.float64)
    # 0.25 * (1 + 1) = 0.5
    assert slice_arithmetic(arr, 1) == 0.5


def test_slice_to_intermediate_variable():
    """Test assigning slice expression to intermediate variable."""

    @native
    def slice_to_var(arr, k) -> float:
        ni = arr.shape[0] - 1
        nj = arr.shape[1]
        # Assign slice expression to intermediate variable
        gcv = 0.25 * (arr[1:, :, k + 1] + arr[:-1, :, k + 1])
        result = np.zeros((ni, nj), dtype=np.float64)
        result[:, :] = gcv
        return result[0, 0]

    arr = np.ones((3, 4, 5), dtype=np.float64)
    assert slice_to_var(arr, 1) == 0.5
