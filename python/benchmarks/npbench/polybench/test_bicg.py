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
    A = np.fromfunction(lambda i, j: (i * (j + 1) % N) / N, (N, M), dtype=datatype)
    p = np.fromfunction(lambda i: (i % M) / M, (M,), dtype=datatype)
    r = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)

    return A, p, r


def kernel(A, p, r):

    return r @ A, A @ p


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_bicg(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 2,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 2,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 2,
                "HIGHWAY": 0,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "CUDA": 2,
                "FOR": 2,
                "MAP": 2,
                "CUDAOffloading": 8,
                "GEMM": 2,
                "Malloc": 2,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "bicg")
