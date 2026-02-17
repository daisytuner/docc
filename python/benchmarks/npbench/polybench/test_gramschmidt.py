import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 70, "N": 60},
    "M": {"M": 220, "N": 180},
    "L": {"M": 600, "N": 500},
    "paper": {"M": 240, "N": 200},
}


def initialize(M, N, datatype=np.float64):
    from numpy.random import default_rng

    rng = default_rng(42)

    A = rng.random((M, N), dtype=datatype)
    while np.linalg.matrix_rank(A) < N:
        A = rng.random((M, N), dtype=datatype)

    return A


def kernel(A):
    Q = np.zeros_like(A)
    R = np.zeros((A.shape[1], A.shape[1]), dtype=A.dtype)

    for k in range(A.shape[1]):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    return Q, R


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Instrumentation not supported on macOS"
)
@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_gramschmidt(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "MAP": 9,
                "CMath": 1,
                "SEQUENTIAL": 9,
                "FOR": 12,
                "Memset": 2,
                "Malloc": 5,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 2,
                "GEMM": 1,
                "MAP": 6,
                "CMath": 1,
                "DOT": 0,
                "SEQUENTIAL": 4,
                "FOR": 9,
                "Memset": 2,
                "Malloc": 2,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 2,
                "GEMM": 1,
                "MAP": 6,
                "CPU_PARALLEL": 3,
                "CMath": 1,
                "DOT": 0,
                "SEQUENTIAL": 1,
                "FOR": 9,
                "Memset": 2,
                "Malloc": 2,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "CUDA": 5,
                "MAP": 6,
                "CUDAOffloading": 14,
                "CMath": 1,
                "DOT": 0,
                "SEQUENTIAL": 1,
                "FOR": 9,
                "Memset": 2,
                "Malloc": 2,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gramschmidt")
