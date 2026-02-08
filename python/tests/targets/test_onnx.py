import sys
import pytest
import numpy as np

from docc.python import native


@pytest.mark.skipif(
    sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
)
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


@pytest.mark.skipif(
    sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
)
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


@pytest.mark.skipif(
    sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
)
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


@pytest.mark.skipif(
    sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
)
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


@pytest.mark.skipif(
    sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
)
def test_onnx_transpose():
    @native(target="onnx", category="server")
    def onnx_transpose(a):
        b = np.transpose(a)
        return b

    A = np.random.rand(64, 32).astype(np.float64)
    C = np.zeros((32, 64), dtype=np.float64)

    C = onnx_transpose(A)
    assert np.allclose(C, A.T)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="ONNX target not yet supported on macOS"
)
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
