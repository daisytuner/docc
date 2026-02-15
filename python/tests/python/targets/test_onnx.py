import sys
import pytest
import numpy as np

from docc.python import native


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_add():
    @native(target="onnx", category="server")
    def onnx_add(a, b):
        return a + b

    N = 64
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    C = onnx_add(A, B)
    assert np.allclose(C, A + B)


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_sub():
    @native(target="onnx", category="server")
    def onnx_sub(a, b):
        return a - b

    N = 64
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    C = onnx_sub(A, B)
    assert np.allclose(C, A - B)


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_mul():
    @native(target="onnx", category="server")
    def onnx_mul(a, b):
        return a * b

    N = 64
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    C = onnx_mul(A, B)
    assert np.allclose(C, A * B)


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_div():
    @native(target="onnx", category="server")
    def onnx_div(a, b):
        return a / b

    N = 64
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    C = onnx_div(A, B)
    assert np.allclose(C, A / B)


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_transpose():
    @native(target="onnx", category="server")
    def onnx_transpose(a):
        b = np.transpose(a)
        return b

    A = np.random.rand(64, 32).astype(np.float64)
    C = np.zeros((32, 64), dtype=np.float64)

    C = onnx_transpose(A)
    assert np.allclose(C, A.T)


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_multiple():
    @native(target="onnx", category="server")
    def onnx_multiple(a, b):
        return (a * b) + (a * b)

    N = 64
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    C = onnx_multiple(A, B)
    assert np.allclose(C, (A * B) + (A * B))


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_dot():
    @native(target="onnx", category="server")
    def onnx_dot(x, y):
        return np.dot(x, y)

    N = 128
    X = np.random.rand(N).astype(np.float64)
    Y = np.random.rand(N).astype(np.float64)

    result = onnx_dot(X, Y)
    assert np.allclose(result, np.dot(X, Y))


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_matmul():
    @native(target="onnx", category="server")
    def onnx_matmul(a, b):
        return a @ b

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(np.float64)
    B = np.random.rand(K, N).astype(np.float64)

    C = onnx_matmul(A, B)
    assert np.allclose(C, A @ B)


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_dot_float32():
    @native(target="onnx", category="server")
    def onnx_dot_f32(x, y):
        return np.dot(x, y)

    N = 128
    X = np.random.rand(N).astype(np.float32)
    Y = np.random.rand(N).astype(np.float32)

    result = onnx_dot_f32(X, Y)
    assert np.allclose(result, np.dot(X, Y), rtol=1e-5)


@pytest.mark.skip()
# @pytest.mark.skipif(
#     sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
# )
def test_onnx_matmul_float32():
    @native(target="onnx", category="server")
    def onnx_matmul_f32(a, b):
        return a @ b

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)

    C = onnx_matmul_f32(A, B)
    assert np.allclose(C, A @ B, rtol=1e-5)
