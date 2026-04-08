from docc.python import native
import pytest
import numpy as np
import math


def test_einsum_dot_product():

    @native
    def dot_product(a, b) -> float:
        return np.einsum("i,i->", a, b)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    # (1.0 * 4.0) + (2.0 * 5.0) + (3.0 * 6.0) = 4.0 + 10.0 + 18.0 = 32.0
    assert dot_product(a, b) == 32.0


def test_einsum_1d_sum():

    @native
    def array_sum(a) -> float:
        return np.einsum("i->", a)

    a = np.array([1.0, 2.0, 3.0, 4.0])
    # 1.0 + 2.0 + 3.0 + 4.0 = 10.0
    assert array_sum(a) == 10.0


def test_einsum_outer_product():

    @native
    def outer_product(a, b):
        return np.einsum("i,j->ij", a, b)

    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    result = outer_product(a, b)
    expected = np.array([[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])
    assert np.allclose(result, expected)


def test_einsum_matrix_transpose():

    @native
    def transpose(m):
        return np.einsum("ij->ji", m)

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = transpose(m)
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert np.allclose(result, expected)


def test_einsum_matrix_trace():

    @native
    def trace(m) -> float:
        return np.einsum("ii->", m)

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    # 1.0 + 5.0 + 9.0 = 15.0
    assert trace(m) == 15.0


def test_einsum_matrix_vector_mul():

    @native
    def matvec(m, v):
        return np.einsum("ij,j->i", m, v)

    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.array([5.0, 6.0])
    result = matvec(m, v)
    # [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    expected = np.array([17.0, 39.0])
    assert np.allclose(result, expected)


def test_einsum_matrix_mul():

    @native
    def matmul(a, b):
        return np.einsum("ij,jk->ik", a, b)

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = matmul(a, b)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(result, expected)


def test_einsum_3d_sum_all():

    @native
    def sum_all(t) -> float:
        return np.einsum("ijk->", t)

    t = np.arange(1.0, 9.0).reshape(2, 2, 2)
    # 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36.0
    assert sum_all(t) == 36.0


def test_einsum_3d_sum_axis():

    @native
    def sum_last_axis(t):
        return np.einsum("ijk->ij", t)

    t = np.arange(1.0, 13.0).reshape(2, 2, 3)
    result = sum_last_axis(t)
    # Sum along last axis: [[1+2+3, 4+5+6], [7+8+9, 10+11+12]] = [[6, 15], [24, 33]]
    expected = np.array([[6.0, 15.0], [24.0, 33.0]])
    assert np.allclose(result, expected)


def test_einsum_3d_permute():

    @native
    def permute(t):
        return np.einsum("ijk->kji", t)

    t = np.arange(1.0, 25.0).reshape(2, 3, 4)
    result = permute(t)
    expected = np.transpose(t, (2, 1, 0))
    assert np.allclose(result, expected)


def test_einsum_3d_diagonal():

    @native
    def diagonal_sum(t) -> float:
        return np.einsum("iii->", t)

    t = np.arange(1.0, 28.0).reshape(3, 3, 3)
    # Diagonal elements: t[0,0,0]=1, t[1,1,1]=14, t[2,2,2]=27
    # 1 + 14 + 27 = 42
    assert diagonal_sum(t) == 42.0


def test_einsum_batch_matmul():

    @native
    def batch_matmul(a, b):
        return np.einsum("bij,bjk->bik", a, b)

    # Batch of 2 matrices: 2x2 @ 2x3
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = np.array(
        [[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    result = batch_matmul(a, b)
    expected = np.array(
        [[[1.0, 2.0, 1.0], [3.0, 4.0, 3.0]], [[11.0, 11.0, 11.0], [15.0, 15.0, 15.0]]]
    )
    assert np.allclose(result, expected)


def test_einsum_3d_contract():

    @native
    def contract_middle(a, b):
        return np.einsum("ijk,jl->ikl", a, b)

    a = np.ones((2, 3, 4))
    b = np.ones((3, 5))
    result = contract_middle(a, b)
    # Contracting over j (size 3), result shape is (2, 4, 5)
    expected = np.full((2, 4, 5), 3.0)
    assert np.allclose(result, expected)
