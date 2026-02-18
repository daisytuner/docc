import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 35, "N": 50},
    "M": {"M": 110, "N": 140},
    "L": {"M": 350, "N": 400},
    "paper": {"M": 1000, "N": 1200},
}


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: (i * j + 3) % N / M, (N, N), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j + 1) % N / N, (N, M), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * j + 2) % M / M, (N, M), dtype=datatype)
    return alpha, beta, C, A, B


def kernel(alpha, beta, C, A, B):
    for i in range(A.shape[0]):
        C[i, : i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, : i + 1] += (
                A[: i + 1, k] * alpha * B[i, k] + B[: i + 1, k] * alpha * A[i, k]
            )


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_syr2k(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"Malloc": 3, "MAP": 5, "SEQUENTIAL": 5, "FOR": 7}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "Malloc": 3,
                "HIGHWAY": 3,
                "SEQUENTIAL": 3,
                "FOR": 8,
                "MAP": 6,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "Malloc": 3,
                "HIGHWAY": 3,
                "CPU_PARALLEL": 1,
                "SEQUENTIAL": 2,
                "FOR": 8,
                "MAP": 6,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "Malloc": 3,
                "CUDA": 1,
                "SEQUENTIAL": 5,
                "FOR": 8,
                "MAP": 6,
                "CUDAOffloading": 2,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "syr2k")
