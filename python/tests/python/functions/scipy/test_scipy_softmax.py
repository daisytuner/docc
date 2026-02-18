import pytest
import numpy as np
import scipy.special

from docc.python import native


def test_softmax_simple():
    def numpy_softmax(x, axis=None):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @native
    def softmax_simple(
        A,
    ):
        return scipy.special.softmax(A, axis=0)

    A = np.random.rand(10, 10).astype(np.float64)
    A_ = A.copy()
    res = softmax_simple(A)
    assert np.allclose(res, numpy_softmax(A_, axis=0))


def test_softmax_all():
    def numpy_softmax(x, axis=None):
        x_max = np.max(x, axis=axis, keepdims=True)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x, axis=axis, keepdims=True)

    @native
    def softmax_all(
        A,
    ):
        return scipy.special.softmax(A)

    A = np.random.rand(10, 10).astype(np.float64)
    A_ = A.copy()
    res = softmax_all(A)
    assert np.allclose(res, numpy_softmax(A_))
