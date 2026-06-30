import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 5500},
    "M": {"N": 11000},
    "L": {"N": 22000},
    "paper": {"N": 16000},
}


def initialize(N, datatype=np.float64):
    x1 = np.fromfunction(lambda i: (i % N) / N, (N,), dtype=datatype)
    x2 = np.fromfunction(lambda i: ((i + 1) % N) / N, (N,), dtype=datatype)
    y_1 = np.fromfunction(lambda i: ((i + 3) % N) / N, (N,), dtype=datatype)
    y_2 = np.fromfunction(lambda i: ((i + 4) % N) / N, (N,), dtype=datatype)
    A = np.fromfunction(lambda i, j: (i * j % N) / N, (N, N), dtype=datatype)

    return x1, x2, y_1, y_2, A


def kernel(x1, x2, y_1, y_2, A):

    x1 += A @ y_1
    x2 += y_2 @ A


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_mvt(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={"SEQUENTIAL": 2, "REDUCE": 2, "CUDA": 1, "MAP": 1},
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={"SEQUENTIAL": 2, "REDUCE": 2, "ROCM": 1, "MAP": 1}
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "mvt")
