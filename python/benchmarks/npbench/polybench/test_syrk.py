import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 50, "N": 70},
    "M": {"M": 150, "N": 200},
    "L": {"M": 500, "N": 600},
    "paper": {"M": 1000, "N": 1200},
}


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: (i * j + 2) % N / M, (N, N), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j + 1) % N / N, (N, M), dtype=datatype)
    return alpha, beta, C, A


def kernel(alpha, beta, C, A):
    for i in range(A.shape[0]):
        C[i, : i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, : i + 1] += alpha * A[i, k] * A[: i + 1, k]


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_syrk(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 5, "Malloc": 3, "SEQUENTIAL": 5, "FOR": 7}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 5,
                "MAP": 4,
                "SEQUENTIAL": 3,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 1,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "FOR": 5,
                "MAP": 4,
                "SEQUENTIAL": 1,
                "CUDA": 0,
                "CPU_PARALLEL": 2,
                "HIGHWAY": 1,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "FOR": 5,
                "MAP": 4,
                "SEQUENTIAL": 2,
                "CUDA": 2,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "syrk")
