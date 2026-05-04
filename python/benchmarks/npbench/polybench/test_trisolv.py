import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 2000},
    "M": {"N": 5000},
    "L": {"N": 14000},
    "paper": {"N": 16000},
}


def initialize(N, datatype=np.float64):
    L = np.fromfunction(lambda i, j: (i + N - j + 1) * 2 / N, (N, N), dtype=datatype)
    x = np.full((N,), -999, dtype=datatype)
    b = np.fromfunction(lambda i: i, (N,), dtype=datatype)

    return L, x, b


def kernel(L, x, b):

    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_trisolv(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    else:  # rocm
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "ROCM": 0,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "trisolv")
