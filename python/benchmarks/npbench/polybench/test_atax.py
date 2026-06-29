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
    fn = datatype(N)
    x = np.fromfunction(lambda i: 1 + (i / fn), (N,), dtype=datatype)
    A = np.fromfunction(lambda i, j: ((i + j) % N) / (5 * M), (M, N), dtype=datatype)

    return A, x


def kernel(A, x):

    return (A @ x) @ A


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_atax(target):
    verifier = None
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "MAP": 1,
                "SEQUENTIAL": 1,
                "GEMM": 2,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "MAP": 1,
                "VECTORIZE": 1,
                "GEMM": 2,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "MAP": 1,
                "CPU_PARALLEL": 1,
                "GEMM": 2,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 2,
                "MAP": 2,
                "REDUCE": 2,
                "CUDAOffloading": 4,
            },
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "ROCM": 2,
                "MAP": 2,
                "REDUCE": 2,
                "ROCMOffloading": 4,
            },
        )
    run_pytest(initialize, kernel, PARAMETERS, target=target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "atax")
