import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"NI": 1000, "NJ": 1100, "NK": 1200},
    "M": {"NI": 2500, "NJ": 2750, "NK": 3000},
    "L": {"NI": 7000, "NJ": 7500, "NK": 8000},
    "paper": {"NI": 2000, "NJ": 2300, "NK": 2600},
}


def initialize(NI, NJ, NK, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    C = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NJ), dtype=datatype)
    A = np.fromfunction(lambda i, k: (i * (k + 1) % NK) / NK, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda k, j: (k * (j + 2) % NJ) / NJ, (NK, NJ), dtype=datatype)

    return alpha, beta, C, A, B


def kernel(alpha, beta, C, A, B):
    C[:] = alpha * A @ B + beta * C


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_gemm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "FOR": 0,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 0,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "FOR": 0,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "FOR": 0,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 1,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "gemm")
