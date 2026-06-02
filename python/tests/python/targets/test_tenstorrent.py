import os

from docc.python import native
import numpy as np
import pytest
import sys


# COR-1607 find a way to depend on target plugins being available or not for pytest
@pytest.mark.skipif(
    not os.path.exists("/dev/tenstorrent"),
    reason="No TT Accelerator present",
)
@pytest.mark.tenstorrent()
def test_scheduling_tt_matmul_large():
    @native(target="tenstorrent", category="server")
    def matmul_tt(A, B, C):
        C = A @ B

    M = 1024
    N = 1024
    K = 1024
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    matmul_tt(A, B, C)

    assert np.allclose(C, A @ B, rtol=1e-3, atol=1e-3)
