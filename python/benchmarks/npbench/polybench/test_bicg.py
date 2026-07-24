import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 4000, "N": 5000},
    "M": {"M": 10000, "N": 12500},
    "L": {"M": 20000, "N": 25000},
    "paper": {"M": 18000, "N": 22000},
}


def initialize(M, N, datatype=np.float64):
    i = np.arange(N).reshape(-1, 1)
    j = np.arange(M)
    A = (i * (j + 1) % N) / N
    p = (np.arange(M) % M) / M
    r = (np.arange(N) % N) / N

    return A, p, r


def kernel(A, p, r):

    return r @ A, A @ p


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_bicg(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "SEQUENTIAL": 2,
                "GEMM": 2,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "VECTORIZE": 2,
                "GEMM": 2,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "CPU_PARALLEL": 2,
                "GEMM": 2,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 4,
                "REDUCE": 2,
                "MAP": 4,
                "CUDAOffloading": 4,
            },
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "ROCM": 4,
                "REDUCE": 2,
                "MAP": 4,
                "ROCMOffloading": 4,
            },
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "bicg")
