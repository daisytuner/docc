from docc.python import native
import numpy as np
import pytest
import sys


def test_scheduling_etsoc_matmul():
    # Assuming CUDA is available and supported
    @native(target="etsoc", category="server")
    def matmul_etsoc(A, B, C):
        C = A @ B

    N = 64
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    matmul_etsoc(A, B, C)
    assert np.allclose(C, A @ B)
