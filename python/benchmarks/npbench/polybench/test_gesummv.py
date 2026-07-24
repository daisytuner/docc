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
    i = np.arange(N).reshape(-1, 1)
    j = np.arange(N)
    A = ((i * j + 1) % N) / N
    B = ((i * j + 2) % N) / N
    x = (np.arange(N) % N) / N

    return alpha, beta, A, B, x


def kernel(alpha, beta, A, B, x):
    return alpha * A @ x + beta * B @ x


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_gesummv(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
                "SEQUENTIAL": 5,
                "MAP": 5,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
                "VECTORIZE": 3,
                "SEQUENTIAL": 2,
                "MAP": 5,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
                "CPU_PARALLEL": 3,
                "MAP": 3,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 6,
                "SEQUENTIAL": 2,
                "REDUCE": 2,
                "MAP": 6,
                "CUDAOffloading": 6,
            },
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "ROCM": 6,
                "SEQUENTIAL": 2,
                "REDUCE": 2,
                "MAP": 6,
                "ROCMOffloading": 6,
            },
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gesummv")
