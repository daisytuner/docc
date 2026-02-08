from docc.python import native
import numpy as np
import pytest
import sys


def test_scheduling_default():
    @native
    def vec_add_default(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add_default(A, B, C, N)
    assert np.allclose(C, A + B)


def test_scheduling_sequential():
    @native(target="sequential", category="desktop")
    def vec_add(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add(A, B, C, N)
    assert np.allclose(C, A + B)


def test_scheduling_openmp():
    # Assuming OpenMP is available and supported
    @native(target="openmp", category="desktop")
    def vec_add_omp(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add_omp(A, B, C, N)
    assert np.allclose(C, A + B)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Instrumentation not supported on macOS"
)
def test_scheduling_cuda():
    # Assuming CUDA is available and supported
    @native(target="cuda", category="server")
    def vec_add_cuda(A, B, C, N):
        for i in range(N):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.zeros(N, dtype=np.float64)

    vec_add_cuda(A, B, C, N)
    assert np.allclose(C, A + B)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Instrumentation not supported on macOS"
)
def test_scheduling_cuda_gemm():
    # Assuming CUDA is available and supported
    @native(target="cuda", category="server")
    def matmul_cuda(A, B, C, N):
        for i in range(N):
            for j in range(N):
                C[i, j] = 0
                for k in range(N):
                    C[i, j] += A[i, k] * B[k, j]

    N = 64
    A = np.random.rand(N, N).astype(np.float64)
    B = np.random.rand(N, N).astype(np.float64)
    C = np.zeros((N, N), dtype=np.float64)

    matmul_cuda(A, B, C, N)
    assert np.allclose(C, A @ B)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Instrumentation not supported on macOS"
)
def test_scheduling_cuda_dot():
    # Assuming CUDA is available and supported
    @native(target="cuda", category="server")
    def dot_cuda(x, y, result, N):
        result[0] = 0
        for i in range(N):
            result[0] += x[i] * y[i]

    N = 1024
    x = np.random.rand(N).astype(np.float64)
    y = np.random.rand(N).astype(np.float64)
    result = np.zeros(1, dtype=np.float64)

    dot_cuda(x, y, result, N)
    assert np.allclose(result[0], x @ y)
