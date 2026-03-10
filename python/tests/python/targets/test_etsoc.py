import os

from docc.python import native
import numpy as np
import pytest
import sys


# COR-1607 find a way to depend on target plugins being available or not for pytest
@pytest.mark.skipif(
    os.environ.get("ETSOC_TESTS") != "1", reason="ETSoC tests are disabled"
)
def test_scheduling_etsoc_matmul_mini():
    @native(target="etsoc", category="server")
    def matmul_etsoc(A, B, C):
        C = A @ B

    N = 16
    A = np.random.rand(N, N).astype(np.float32)
    A.fill(0.5)
    B = np.random.rand(N, N).astype(np.float32)
    B.fill(2)
    C = np.zeros((N, N), dtype=np.float32)

    print("Input: ", A)
    matmul_etsoc(A, B, C)
    print("Result: ", C)
    assert np.allclose(C, A @ B)


@pytest.mark.skipif(
    os.environ.get("ETSOC_SLOW_TESTS") != "1", reason="ETSoC Slow tests are disabled"
)
def test_scheduling_etsoc_matmul_large():
    @native(target="etsoc", category="server")
    def matmul_etsoc(A, B, C):
        C = A @ B

    M = 1024
    N = 1024
    K = 1024
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul_etsoc(A, B, C)

    assert np.allclose(C, A @ B)
