import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 2000},
    "M": {"N": 4000},
    "L": {"N": 14000},
    "paper": {"N": 11200},
}


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % N) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: ((i * j + 2) % N) / N, (N, N), dtype=datatype)
    x = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)

    return alpha, beta, A, B, x


def kernel(alpha, beta, A, B, x):
    return alpha * A @ x + beta * B @ x


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_gesummv(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "Free": 4,
                "GEMM": 2,
                "SEQUENTIAL": 5,
                "FOR": 5,
                "MAP": 5,
                "Malloc": 4,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "Free": 4,
                "GEMM": 2,
                "VECTORIZE": 3,
                "SEQUENTIAL": 2,
                "FOR": 5,
                "MAP": 5,
                "Malloc": 4,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "Free": 4,
                "GEMM": 2,
                "CPU_PARALLEL": 3,
                "FOR": 3,
                "MAP": 3,
                "Malloc": 4,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "Free": 4,
                "GEMM": 2,
                "CUDA": 5,
                "FOR": 5,
                "MAP": 5,
                "CUDAOffloading": 14,
                "Malloc": 4,
            }
        )
    else:  # rocm
        verifier = SDFGVerification(
            verification={
                "Free": 4,
                "GEMM": 2,
                "ROCM": 5,
                "FOR": 5,
                "MAP": 5,
                "ROCMOffloading": 14,
                "Malloc": 4,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gesummv")
