import pytest
import numpy as np
import math
from docc.python import native


def test_scalar_1d():

    @native
    def scalar_1d(A):
        A[0] = np.sqrt(A[1])

    a = np.array([4.0, 9.0, 16.0], dtype=np.float64)
    scalar_1d(a)
    assert abs(a[0] - 3.0) < 1e-10


def test_scalar_2d_c():

    @native
    def sqrt_c_00(A):
        A[0, 0] = np.sqrt(A[0, 0])

    a = np.array(
        [[4.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64, order="C"
    )
    sqrt_c_00(a)
    assert abs(a[0, 0] - 2.0) < 1e-10

    # Non-corner element [0,1]
    @native
    def sqrt_c_01(A):
        A[0, 1] = np.sqrt(A[0, 1])

    a = np.array(
        [[1.0, 9.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64, order="C"
    )
    sqrt_c_01(a)
    assert abs(a[0, 1] - 3.0) < 1e-10

    # Non-corner element [1,0]
    @native
    def sqrt_c_10(A):
        A[1, 0] = np.sqrt(A[1, 0])

    a = np.array(
        [[1.0, 1.0, 1.0], [16.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
        order="C",
    )
    sqrt_c_10(a)
    assert abs(a[1, 0] - 4.0) < 1e-10

    # Center element [1,1]
    @native
    def sqrt_c_11(A):
        A[1, 1] = np.sqrt(A[1, 1])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 25.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
        order="C",
    )
    sqrt_c_11(a)
    assert abs(a[1, 1] - 5.0) < 1e-10

    # Non-corner element [1,2]
    @native
    def sqrt_c_12(A):
        A[1, 2] = np.sqrt(A[1, 2])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 36.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
        order="C",
    )
    sqrt_c_12(a)
    assert abs(a[1, 2] - 6.0) < 1e-10

    # Corner element [2,2]
    @native
    def sqrt_c_22(A):
        A[2, 2] = np.sqrt(A[2, 2])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 49.0]],
        dtype=np.float64,
        order="C",
    )
    sqrt_c_22(a)
    assert abs(a[2, 2] - 7.0) < 1e-10


def test_scalar_2d_f():

    # Corner element [0,0]
    @native
    def sqrt_f_00(A):
        A[0, 0] = np.sqrt(A[0, 0])

    a = np.array(
        [[4.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64, order="F"
    )
    sqrt_f_00(a)
    assert abs(a[0, 0] - 2.0) < 1e-10

    # Non-corner element [0,1]
    @native
    def sqrt_f_01(A):
        A[0, 1] = np.sqrt(A[0, 1])

    a = np.array(
        [[1.0, 9.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64, order="F"
    )
    sqrt_f_01(a)
    assert abs(a[0, 1] - 3.0) < 1e-10

    # Non-corner element [1,0]
    @native
    def sqrt_f_10(A):
        A[1, 0] = np.sqrt(A[1, 0])

    a = np.array(
        [[1.0, 1.0, 1.0], [16.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
        order="F",
    )
    sqrt_f_10(a)
    assert abs(a[1, 0] - 4.0) < 1e-10

    # Center element [1,1]
    @native
    def sqrt_f_11(A):
        A[1, 1] = np.sqrt(A[1, 1])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 25.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
        order="F",
    )
    sqrt_f_11(a)
    assert abs(a[1, 1] - 5.0) < 1e-10

    # Non-corner element [2,1]
    @native
    def sqrt_f_21(A):
        A[2, 1] = np.sqrt(A[2, 1])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 36.0, 1.0]],
        dtype=np.float64,
        order="F",
    )
    sqrt_f_21(a)
    assert abs(a[2, 1] - 6.0) < 1e-10

    # Corner element [2,2]
    @native
    def sqrt_f_22(A):
        A[2, 2] = np.sqrt(A[2, 2])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 49.0]],
        dtype=np.float64,
        order="F",
    )
    sqrt_f_22(a)
    assert abs(a[2, 2] - 7.0) < 1e-10


def test_scalar_slice_row():

    @native
    def sqrt_from_row0(A):
        row = A[0, :]
        A[1, 1] = np.sqrt(row[1])

    a = np.array([[1.0, 36.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64, order="C")
    sqrt_from_row0(a)
    assert abs(a[1, 1] - 6.0) < 1e-10

    @native
    def sqrt_from_row1(A):
        row = A[1, :]
        A[0, 0] = np.sqrt(row[2])

    a = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 49.0]], dtype=np.float64, order="C")
    sqrt_from_row1(a)
    assert abs(a[0, 0] - 7.0) < 1e-10

    # @native
    # def sqrt_from_row0_f(A):
    #     row = A[0, :]
    #     A[1, 1] = np.sqrt(row[1])

    # a = np.array([[1.0, 36.0, 1.0], [1.0, 0.0, 1.0]], dtype=np.float64, order="F")
    # sqrt_from_row0_f(a)
    # assert abs(a[1, 1] - 6.0) < 1e-10

    # @native
    # def sqrt_from_row1_f(A):
    #     row = A[1, :]
    #     A[0, 0] = np.sqrt(row[2])

    # a = np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 64.0]], dtype=np.float64, order="F")
    # sqrt_from_row1_f(a)
    # assert abs(a[0, 0] - 8.0) < 1e-10


def test_scalar_slice_col():

    @native
    def sqrt_from_col0(A):
        col = A[:, 0]
        A[0, 1] = np.sqrt(col[1])

    a = np.array([[1.0, 0.0], [81.0, 1.0]], dtype=np.float64, order="C")
    sqrt_from_col0(a)
    assert abs(a[0, 1] - 9.0) < 1e-10

    @native
    def sqrt_from_col1(A):
        col = A[:, 1]
        A[1, 0] = np.sqrt(col[0])

    a = np.array([[1.0, 100.0], [0.0, 1.0]], dtype=np.float64, order="C")
    sqrt_from_col1(a)
    assert abs(a[1, 0] - 10.0) < 1e-10

    # @native
    # def sqrt_from_col0_f(A):
    #     col = A[:, 0]
    #     A[0, 1] = np.sqrt(col[1])

    # a = np.array([[1.0, 0.0], [81.0, 1.0]], dtype=np.float64, order="F")
    # sqrt_from_col0_f(a)
    # assert abs(a[0, 1] - 9.0) < 1e-10

    # @native
    # def sqrt_from_col1_f(A):
    #     col = A[:, 1]
    #     A[1, 0] = np.sqrt(col[0])

    # a = np.array([[1.0, 121.0], [0.0, 1.0]], dtype=np.float64, order="F")
    # sqrt_from_col1_f(a)
    # assert abs(a[1, 0] - 11.0) < 1e-10


def test_scalar_return():

    # C-order non-corner
    @native
    def sqrt_return_c_11(A) -> float:
        return np.sqrt(A[1, 1])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 676.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
        order="C",
    )
    result = sqrt_return_c_11(a)
    assert abs(result - 26.0) < 1e-10

    # F-order non-corner
    @native
    def sqrt_return_f_11(A) -> float:
        return np.sqrt(A[1, 1])

    a = np.array(
        [[1.0, 1.0, 1.0], [1.0, 729.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
        order="F",
    )
    result = sqrt_return_f_11(a)
    assert abs(result - 27.0) < 1e-10

    # From slice element
    @native
    def sqrt_return_slice(A) -> float:
        row = A[1, :]
        return np.sqrt(row[2])

    a = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 784.0]], dtype=np.float64, order="C")
    result = sqrt_return_slice(a)
    assert abs(result - 28.0) < 1e-10
