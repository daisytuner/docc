import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 40, "N": 50},
    "M": {"M": 120, "N": 150},
    "L": {"M": 350, "N": 550},
    "paper": {"M": 1000, "N": 1200},
}


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i + j) % 100) / M, (M, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((N + i - j) % 100) / M, (M, N), dtype=datatype)
    A = np.empty((M, M), dtype=datatype)
    for i in range(M):
        A[i, : i + 1] = np.fromfunction(
            lambda j: ((i + j) % 100) / M, (i + 1,), dtype=datatype
        )
        A[i, i + 1 :] = -999

    return alpha, beta, C, A, B


def kernel(alpha, beta, C, A, B):

    temp2 = np.empty((C.shape[1],), dtype=C.dtype)
    C *= beta
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_symm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"GEMM": 1, "SEQUENTIAL": 7, "FOR": 9, "MAP": 7, "Malloc": 3}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "HIGHWAY": 4,
                "SEQUENTIAL": 3,
                "FOR": 9,
                "MAP": 7,
                "Malloc": 3,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "HIGHWAY": 4,
                "CPU_PARALLEL": 2,
                "SEQUENTIAL": 1,
                "FOR": 9,
                "MAP": 7,
                "Malloc": 3,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "FOR": 7,
                "MAP": 5,
                "SEQUENTIAL": 0,
                "CUDA": 5,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "symm")
