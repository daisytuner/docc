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
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)
    u1 = np.fromfunction(lambda i: i, (N,), dtype=datatype)
    u2 = np.fromfunction(lambda i: ((i + 1) / fn) / 2.0, (N,), dtype=datatype)
    v1 = np.fromfunction(lambda i: ((i + 1) / fn) / 4.0, (N,), dtype=datatype)
    v2 = np.fromfunction(lambda i: ((i + 1) / fn) / 6.0, (N,), dtype=datatype)
    w = np.zeros((N,), dtype=datatype)
    x = np.zeros((N,), dtype=datatype)
    y = np.fromfunction(lambda i: ((i + 1) / fn) / 8.0, (N,), dtype=datatype)
    z = np.fromfunction(lambda i: ((i + 1) / fn) / 9.0, (N,), dtype=datatype)

    return alpha, beta, A, u1, v1, u2, v2, w, x, y, z


def kernel(alpha, beta, A, u1, v1, u2, v2, w, x, y, z):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_gemver(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"SEQUENTIAL": 2, "FOR": 2, "MAP": 2, "Malloc": 2, "GEMM": 4}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={"HIGHWAY": 2, "FOR": 2, "MAP": 2, "Malloc": 2, "GEMM": 4}
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"CPU_PARALLEL": 2, "FOR": 2, "MAP": 2, "Malloc": 2, "GEMM": 4}
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "CUDA": 2,
                "FOR": 2,
                "MAP": 2,
                "CUDAOffloading": 10,
                "Malloc": 2,
                "GEMM": 4,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gemver")
