import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 1000},
    "M": {"N": 3000},
    "L": {"N": 10000},
    "paper": {"N": 8000},
}


def initialize(N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    fn = datatype(N)
    i = np.arange(N).reshape(-1, 1)
    j = np.arange(N)
    A = (i * j % N) / N
    i = np.arange(N, dtype=datatype)
    u1 = np.arange(N, dtype=datatype)
    u2 = ((i + 1) / fn) / 2.0
    v1 = ((i + 1) / fn) / 4.0
    v2 = ((i + 1) / fn) / 6.0
    w = np.zeros((N,), dtype=datatype)
    x = np.zeros((N,), dtype=datatype)
    y = ((i + 1) / fn) / 8.0
    z = ((i + 1) / fn) / 9.0

    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z


def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_gemver(target):
    if target == "none":
        verifier = SDFGVerification(verification={"SEQUENTIAL": 3, "MAP": 3, "GEMM": 4})
    elif target == "sequential":
        verifier = SDFGVerification(verification={"VECTORIZE": 3, "MAP": 3, "GEMM": 4})
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 3,
                "MAP": 3,
                "GEMM": 4,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 8,
                "SEQUENTIAL": 4,
                "REDUCE": 4,
                "MAP": 8,
                "CUDAOffloading": 6,
            },
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "ROCM": 8,
                "SEQUENTIAL": 4,
                "REDUCE": 4,
                "MAP": 8,
                "ROCMOffloading": 6,
            },
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gemver")
