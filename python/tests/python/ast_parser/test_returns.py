import numpy as np
import pytest
from docc.python import native


def test_no_annotation_return_scalar():
    print("Testing scalar return without annotation...")

    @native
    def func(a: int, b: int):
        return a + b

    res = func(1, 2)
    print(f"Result: {res}")
    assert res == 3


def test_no_annotation_return_array():
    print("Testing array return without annotation...")

    @native
    def func(A):
        return A

    A = np.ones((10,))
    res = func(A)
    print(f"Result type: {type(res)}")
    assert np.allclose(res, A)


def test_matmul_return():
    print("Testing matmul return...")

    @native
    def func(A, x):
        return A @ x

    M, N = 10, 10
    A = np.ones((M, N))
    x = np.ones((N,))
    res = func(A, x)
    print(f"Result type: {type(res)}")

    assert np.allclose(res, A @ x)


def test_multiple_scalar_return():
    @native
    def multi_scalar(a: int, b: int):
        return a + b, a - b

    res = multi_scalar(10, 5)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert res[0] == 15
    assert res[1] == 5


def test_multiple_array_return():
    @native
    def multi_array(a, b):
        return a, b

    a = np.random.rand(10)
    b = np.random.rand(10)
    res = multi_array(a, b)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert np.allclose(res[0], a)
    assert np.allclose(res[1], b)


def test_mixed_return():
    @native
    def mixed_ret(a):
        x = a[0]
        return a, x

    a = np.random.rand(10)
    res = mixed_ret(a)
    assert isinstance(res, tuple)
    assert len(res) == 2
    assert np.allclose(res[0], a)
    assert res[1] == a[0]


def test_gramschmidt_style_return():
    @native
    def gs_ret(A):
        Q = np.zeros_like(A)
        R = np.zeros_like(A)
        # Dummy op
        Q[0, 0] = 1.0
        R[0, 0] = 1.0
        return Q, R

    A = np.random.rand(10, 10)
    Q, R = gs_ret(A)
    assert Q.shape == (10, 10)
    assert R.shape == (10, 10)
    assert Q[0, 0] == 1.0
