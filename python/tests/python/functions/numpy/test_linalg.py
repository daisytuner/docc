import numpy as np
from docc.python import native


def test_matmul_operator():
    @native
    def matmul_op(a, b):
        return a @ b

    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    res = matmul_op(a, b)
    assert np.allclose(res, a @ b)


def test_numpy_matmul():
    @native
    def np_matmul(a, b):
        return np.matmul(a, b)

    a = np.random.rand(10, 10)
    b = np.random.rand(10, 10)
    res = np_matmul(a, b)
    assert np.allclose(res, np.matmul(a, b))


def test_numpy_dot_matvec():
    @native
    def np_dot_mv(a, b):
        return np.dot(a, b)

    a = np.random.rand(10, 10)
    b = np.random.rand(10)
    res = np_dot_mv(a, b)
    assert np.allclose(res, np.dot(a, b))


def test_matmul_slicing():
    @native
    def matmul_slice(a, b):
        return a[:10, :10] @ b[:10, :10]

    a = np.random.rand(20, 20)
    b = np.random.rand(20, 20)
    res = matmul_slice(a, b)
    assert np.allclose(res, a[:10, :10] @ b[:10, :10])


def test_matmul_broadcasting():
    # (2, 10, 10) @ (2, 10, 10) -> (2, 10, 10)
    @native
    def matmul_broadcast(a, b):
        return np.matmul(a, b)

    a = np.random.rand(2, 10, 10)
    b = np.random.rand(2, 10, 10)
    res = matmul_broadcast(a, b)
    assert np.allclose(res, np.matmul(a, b))


def test_matmul_matvec():
    @native
    def matmul_mv(a, b):
        return np.matmul(a, b)

    a = np.random.rand(10, 10)
    b = np.random.rand(10)
    res = matmul_mv(a, b)
    assert np.allclose(res, np.matmul(a, b))


def test_dot_product_operator():
    @native
    def dot_op(a, b) -> float:
        return a @ b

    a = np.random.rand(10)
    b = np.random.rand(10)
    res = dot_op(a, b)
    assert np.allclose(res, a @ b)


def test_dot_product_slicing_scalar():
    @native
    def dot_slice(a, b) -> float:
        return a[:5] @ b[:5]

    a = np.random.rand(10)
    b = np.random.rand(10)
    res = dot_slice(a, b)
    assert np.allclose(res, a[:5] @ b[:5])


def test_numpy_outer():
    @native
    def np_outer(a, b):
        return np.outer(a, b)

    a = np.random.rand(10)
    b = np.random.rand(10)
    res = np_outer(a, b)
    assert np.allclose(res, np.outer(a, b))


def test_outer_slicing():
    @native
    def outer_slice(a, b):
        return np.outer(a[:10], b[10:])

    a = np.random.rand(20)
    b = np.random.rand(20)
    res = outer_slice(a, b)
    assert np.allclose(res, np.outer(a[:10], b[10:]))


def test_outer_accumulate():
    @native
    def outer_acc(a, b, C):
        C[:] += np.outer(a, b)
        return C

    a = np.random.rand(10)
    b = np.random.rand(10)
    C = np.zeros((10, 10))
    expected = C.copy() + np.outer(a, b)

    res = outer_acc(a, b, C)
    assert np.allclose(res, expected)


def test_outer_double_accumulate():
    @native
    def outer_double_acc(
        a,
        b,
        c,
        d,
        C,
    ):
        C[:] += np.outer(a, b) + np.outer(c, d)
        return C

    a = np.random.rand(10)
    b = np.random.rand(10)
    c_arr = np.random.rand(10)
    d = np.random.rand(10)
    C = np.zeros((10, 10))
    expected = C.copy() + np.outer(a, b) + np.outer(c_arr, d)

    res = outer_double_acc(a, b, c_arr, d, C)
    assert np.allclose(res, expected)


def test_2d_addition():
    @native
    def add_2d(n: int):
        a = np.zeros((10, 10), dtype=float)
        b = np.zeros((10, 10), dtype=float)
        a[0, 0] = 1.0
        b[0, 0] = 2.0
        c = a + b
        return c

    res = add_2d(10)
    assert res[0, 0] == 3.0


def test_matmul_operator_float32():
    @native
    def matmul_op_f32(a, b):
        return a @ b

    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10, 10).astype(np.float32)
    res = matmul_op_f32(a, b)
    assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
    assert np.allclose(res, a @ b, rtol=1e-5)


def test_numpy_matmul_float32():
    @native
    def np_matmul_f32(a, b):
        return np.matmul(a, b)

    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10, 10).astype(np.float32)
    res = np_matmul_f32(a, b)
    assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
    assert np.allclose(res, np.matmul(a, b), rtol=1e-5)


def test_numpy_dot_matvec_float32():
    @native
    def np_dot_mv_f32(a, b):
        return np.dot(a, b)

    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10).astype(np.float32)
    res = np_dot_mv_f32(a, b)
    assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
    assert np.allclose(res, np.dot(a, b), rtol=1e-5)


def test_matmul_matvec_float32():
    @native
    def matmul_mv_f32(a, b):
        return np.matmul(a, b)

    a = np.random.rand(10, 10).astype(np.float32)
    b = np.random.rand(10).astype(np.float32)
    res = matmul_mv_f32(a, b)
    assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
    assert np.allclose(res, np.matmul(a, b), rtol=1e-5)


def test_matmul_chained_float32():
    @native
    def matmul_chained_f32(a, b, c):
        return (a @ b) @ c

    a = np.random.rand(8, 10).astype(np.float32)
    b = np.random.rand(10, 12).astype(np.float32)
    c = np.random.rand(12, 6).astype(np.float32)
    res = matmul_chained_f32(a, b, c)
    expected = (a @ b) @ c
    assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
    assert np.allclose(res, expected, rtol=1e-4)


def test_numpy_outer_float32():
    @native
    def np_outer_f32(a, b):
        return np.outer(a, b)

    a = np.random.rand(10).astype(np.float32)
    b = np.random.rand(10).astype(np.float32)
    res = np_outer_f32(a, b)
    assert res.dtype == np.float32, f"Expected float32, got {res.dtype}"
    assert np.allclose(res, np.outer(a, b), rtol=1e-5)
