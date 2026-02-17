import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 65, "N": 80},
    "M": {"M": 200, "N": 250},
    "L": {"M": 600, "N": 700},
    "paper": {"M": 1000, "N": 1200},
}


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    A = np.fromfunction(lambda i, j: ((i * j) % M) / M, (M, M), dtype=datatype)
    for i in range(M):
        A[i, i] = 1.0
    B = np.fromfunction(lambda i, j: ((N + i - j) % N) / N, (M, N), dtype=datatype)

    return alpha, A, B


def kernel(alpha, A, B):

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1 :, i], B[i + 1 :, j])
    B *= alpha


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_trmm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 4, "Malloc": 1, "GEMM": 1, "SEQUENTIAL": 4, "FOR": 6}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 4,
                "MAP": 2,
                "SEQUENTIAL": 1,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 1,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "FOR": 4,
                "MAP": 2,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 1,
                "HIGHWAY": 1,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "FOR": 4,
                "MAP": 2,
                "SEQUENTIAL": 0,
                "CUDA": 2,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "trmm")
