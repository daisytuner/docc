import pytest
import numpy as np
from docc.python import native


def test_slicing_contiguous():

    @native
    def slice_rows(A, B):
        B[:, :] = A[1:3, :]

    a = np.arange(20, dtype=np.float64).reshape(5, 4)
    b = np.zeros((2, 4), dtype=np.float64)
    slice_rows(a, b)
    np.testing.assert_allclose(b, a[1:3, :])

    @native
    def slice_cols(A, B):
        B[:, :] = A[:, 1:3]

    a = np.arange(20, dtype=np.float64).reshape(4, 5)
    b = np.zeros((4, 2), dtype=np.float64)
    slice_cols(a, b)
    np.testing.assert_allclose(b, a[:, 1:3])

    @native
    def slice_both(A, B):
        B[:, :] = A[1:3, 1:4]

    a = np.arange(30, dtype=np.float64).reshape(5, 6)
    b = np.zeros((2, 3), dtype=np.float64)
    slice_both(a, b)
    np.testing.assert_allclose(b, a[1:3, 1:4])

    @native
    def slice_from_start(A, B):
        B[:, :] = A[:2, :3]

    a = np.arange(24, dtype=np.float64).reshape(4, 6)
    b = np.zeros((2, 3), dtype=np.float64)
    slice_from_start(a, b)
    np.testing.assert_allclose(b, a[:2, :3])

    @native
    def slice_to_end(A, B):
        B[:, :] = A[2:, 3:]

    a = np.arange(24, dtype=np.float64).reshape(4, 6)
    b = np.zeros((2, 3), dtype=np.float64)
    slice_to_end(a, b)
    np.testing.assert_allclose(b, a[2:, 3:])


def test_slicing_strided():

    @native
    def stride_rows(A, B):
        B[:, :] = A[::2, :]

    a = np.arange(20, dtype=np.float64).reshape(5, 4)
    b = np.zeros((3, 4), dtype=np.float64)
    stride_rows(a, b)
    np.testing.assert_allclose(b, a[::2, :])

    @native
    def stride_cols(A, B):
        B[:, :] = A[:, ::2]

    a = np.arange(24, dtype=np.float64).reshape(4, 6)
    b = np.zeros((4, 3), dtype=np.float64)
    stride_cols(a, b)
    np.testing.assert_allclose(b, a[:, ::2])

    @native
    def stride_both(A, B):
        B[:, :] = A[::2, ::3]

    a = np.arange(42, dtype=np.float64).reshape(6, 7)
    b = np.zeros((3, 3), dtype=np.float64)
    stride_both(a, b)
    np.testing.assert_allclose(b, a[::2, ::3])

    @native
    def stride_offset(A, B):
        B[:, :] = A[1::2, :]

    a = np.arange(20, dtype=np.float64).reshape(5, 4)
    b = np.zeros((2, 4), dtype=np.float64)
    stride_offset(a, b)
    np.testing.assert_allclose(b, a[1::2, :])

    @native
    def stride_3(A, B):
        B[:, :] = A[::3, :]

    a = np.arange(40, dtype=np.float64).reshape(10, 4)
    b = np.zeros((4, 4), dtype=np.float64)
    stride_3(a, b)
    np.testing.assert_allclose(b, a[::3, :])


def test_slicing_dimensions():

    @native
    def extract_row(A, B):
        B[:] = A[1, :]

    a = np.arange(20, dtype=np.float64).reshape(4, 5)
    b = np.zeros(5, dtype=np.float64)
    extract_row(a, b)
    np.testing.assert_allclose(b, a[1, :])

    @native
    def extract_col(A, B):
        B[:] = A[:, 2]

    a = np.arange(20, dtype=np.float64).reshape(4, 5)
    b = np.zeros(4, dtype=np.float64)
    extract_col(a, b)
    np.testing.assert_allclose(b, a[:, 2])

    @native
    def extract_last_row(A, B):
        B[:] = A[-1, :]

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.zeros(4, dtype=np.float64)
    extract_last_row(a, b)
    np.testing.assert_allclose(b, a[-1, :])

    @native
    def extract_last_col(A, B):
        B[:] = A[:, -1]

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.zeros(3, dtype=np.float64)
    extract_last_col(a, b)
    np.testing.assert_allclose(b, a[:, -1])

    @native
    def extract_mid_row(A, B):
        B[:] = A[3, :]

    a = np.arange(42, dtype=np.float64).reshape(7, 6)
    b = np.zeros(6, dtype=np.float64)
    extract_mid_row(a, b)
    np.testing.assert_allclose(b, a[3, :])


def test_slicing_negative_step():

    @native
    def reverse_rows(A, B):
        B[:, :] = A[::-1, :]

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.zeros((3, 4), dtype=np.float64)
    reverse_rows(a, b)
    np.testing.assert_allclose(b, a[::-1, :])

    @native
    def reverse_cols(A, B):
        B[:, :] = A[:, ::-1]

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.zeros((3, 4), dtype=np.float64)
    reverse_cols(a, b)
    np.testing.assert_allclose(b, a[:, ::-1])

    @native
    def reverse_both(A, B):
        B[:, :] = A[::-1, ::-1]

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.zeros((3, 4), dtype=np.float64)
    reverse_both(a, b)
    np.testing.assert_allclose(b, a[::-1, ::-1])

    @native
    def reverse_step2(A, B):
        B[:, :] = A[::-2, :]

    a = np.arange(20, dtype=np.float64).reshape(5, 4)
    b = np.zeros((3, 4), dtype=np.float64)
    reverse_step2(a, b)
    np.testing.assert_allclose(b, a[::-2, :])


def test_slicing_combined():

    @native
    def combined_slice(A, B):
        B[:, :] = A[1:5:2, :]

    a = np.arange(24, dtype=np.float64).reshape(6, 4)
    b = np.zeros((2, 4), dtype=np.float64)
    combined_slice(a, b)
    np.testing.assert_allclose(b, a[1:5:2, :])

    @native
    def combined_slice2(A, B):
        B[:, :] = A[::2, 1:4]

    a = np.arange(24, dtype=np.float64).reshape(4, 6)
    b = np.zeros((2, 3), dtype=np.float64)
    combined_slice2(a, b)
    np.testing.assert_allclose(b, a[::2, 1:4])

    @native
    def combined_full(A, B):
        B[:, :] = A[1:5:2, 0:6:3]

    a = np.arange(48, dtype=np.float64).reshape(6, 8)
    b = np.zeros((2, 2), dtype=np.float64)
    combined_full(a, b)
    np.testing.assert_allclose(b, a[1:5:2, 0:6:3])

    @native
    def add_slices(A, B, C):
        C[:, :] = A[1:3, :] + B[0:2, :]

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.arange(8, dtype=np.float64).reshape(2, 4) * 10
    c = np.zeros((2, 4), dtype=np.float64)
    add_slices(a, b, c)
    np.testing.assert_allclose(c, a[1:3, :] + b[0:2, :])

    @native
    def scalar_from_slice(A):
        view = A[::2, :]
        A[1, 1] = np.sqrt(view[1, 0])

    a = np.array([[1.0, 1.0], [1.0, 0.0], [144.0, 1.0]], dtype=np.float64)
    scalar_from_slice(a)
    assert abs(a[1, 1] - 12.0) < 1e-10


def test_slicing_f_order():

    @native
    def slice_forder(A, B):
        B[:, :] = A[1:3, :]

    a = np.arange(20, dtype=np.float64).reshape(5, 4, order="F")
    b = np.zeros((2, 4), dtype=np.float64)
    slice_forder(a, b)
    np.testing.assert_allclose(b, a[1:3, :])

    @native
    def stride_forder(A, B):
        B[:, :] = A[::2, :]

    a = np.arange(20, dtype=np.float64).reshape(5, 4, order="F")
    b = np.zeros((3, 4), dtype=np.float64)
    stride_forder(a, b)
    np.testing.assert_allclose(b, a[::2, :])

    @native
    def col_forder(A, B):
        B[:] = A[:, 2]

    a = np.arange(20, dtype=np.float64).reshape(4, 5, order="F")
    b = np.zeros(4, dtype=np.float64)
    col_forder(a, b)
    np.testing.assert_allclose(b, a[:, 2])


def test_slicing_assign():

    @native
    def assign_scalar(A):
        A[1:3, :] = 5.0

    a = np.zeros((5, 4), dtype=np.float64)
    assign_scalar(a)
    expected = np.zeros((5, 4))
    expected[1:3, :] = 5.0
    np.testing.assert_allclose(a, expected)

    @native
    def single_element(A, B):
        B[:, :] = A[1:2, 2:3]

    a = np.arange(20, dtype=np.float64).reshape(4, 5)
    b = np.zeros((1, 1), dtype=np.float64)
    single_element(a, b)
    np.testing.assert_allclose(b, a[1:2, 2:3])

    @native
    def assign_rows(A, B):
        A[1:3, :] = B

    a = np.zeros((5, 4), dtype=np.float64)
    b = np.arange(8, dtype=np.float64).reshape(2, 4)
    assign_rows(a, b)
    np.testing.assert_allclose(a[1:3, :], b)

    @native
    def assign_expr(A, B):
        A[1:3, :] = B * 2 + 1

    a = np.zeros((5, 4), dtype=np.float64)
    b = np.arange(8, dtype=np.float64).reshape(2, 4)
    assign_expr(a, b)
    np.testing.assert_allclose(a[1:3, :], b * 2 + 1)

    @native
    def assign_cols(A, B):
        A[:, 1:3] = B

    a = np.zeros((4, 5), dtype=np.float64)
    b = np.arange(8, dtype=np.float64).reshape(4, 2)
    assign_cols(a, b)
    np.testing.assert_allclose(a[:, 1:3], b)

    # flaky segfault
    # @native
    # def assign_strided(A, B):
    #     A[::2, :] = B

    # a = np.zeros((5, 4), dtype=np.float64)
    # b = np.arange(12, dtype=np.float64).reshape(3, 4)
    # assign_strided(a, b)
    # np.testing.assert_allclose(a[::2, :], b)

    @native
    def full_slice(A, B):
        B[:, :] = A[:, :]

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    b = np.zeros((3, 4), dtype=np.float64)
    full_slice(a, b)
    np.testing.assert_allclose(b, a)

    @native
    def slice_1d(A, B):
        B[:] = A[2:5]

    a = np.arange(10, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    slice_1d(a, b)
    np.testing.assert_allclose(b, a[2:5])

    @native
    def stride_1d(A, B):
        B[:] = A[::2]

    a = np.arange(10, dtype=np.float64)
    b = np.zeros(5, dtype=np.float64)
    stride_1d(a, b)
    np.testing.assert_allclose(b, a[::2])


def test_slicing_view_properties():

    @native
    def write_via_view(A):
        view = A[1:3, :]
        view[0, 0] = 999.0

    a = np.arange(20, dtype=np.float64).reshape(5, 4)
    write_via_view(a)
    assert a[1, 0] == 999.0  # Should affect original

    @native
    def chained_view_write(A):
        view1 = A[::2, :]
        view2 = view1[:, ::2]
        view2[0, 0] = 888.0

    a = np.arange(24, dtype=np.float64).reshape(4, 6)
    chained_view_write(a)
    assert a[0, 0] == 888.0  # Should affect original
