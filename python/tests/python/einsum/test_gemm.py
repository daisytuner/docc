import numpy as np

from docc.python import native


def _check(matmul, A_trans=False, B_trans=False):
    M = 100
    N = 110
    K = 120

    alpha = np.float64(1.5)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % K) / K, (M, K), dtype=np.float64)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % N) / N, (K, N), dtype=np.float64)
    C_res = np.fromfunction(
        lambda i, j: ((i * j + 1) % M) / M, (M, N), dtype=np.float64
    )
    C_ref = np.fromfunction(
        lambda i, j: ((i * j + 1) % M) / M, (M, N), dtype=np.float64
    )

    matmul(M, N, K, alpha, A, B, C_res)
    C_ref[:] = alpha * A @ B + C_ref

    np.testing.assert_allclose(C_res, C_ref, rtol=1e-5, atol=1e-8)


def test_naive():
    @native
    def naive(M, N, K, alpha, A, B, C):
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    C[i, j] = alpha * A[i, k] * B[k, j] + C[i, j]

    _check(naive)
