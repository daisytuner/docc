import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"NI": 800, "NJ": 850, "NK": 900, "NL": 950},
    "M": {"NI": 2000, "NJ": 2250, "NK": 2500, "NL": 2750},
    "L": {"NI": 6000, "NJ": 6500, "NK": 7000, "NL": 7500},
    "paper": {"NI": 3200, "NJ": 3600, "NK": 4400, "NL": 4800},
}


def initialize(NI, NJ, NK, NL, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    A = np.fromfunction(lambda i, j: ((i * j + 1) % NI) / NI, (NI, NK), dtype=datatype)
    B = np.fromfunction(lambda i, j: (i * (j + 1) % NJ) / NJ, (NK, NJ), dtype=datatype)
    C = np.fromfunction(
        lambda i, j: ((i * (j + 3) + 1) % NL) / NL, (NJ, NL), dtype=datatype
    )
    D = np.fromfunction(lambda i, j: (i * (j + 2) % NK) / NK, (NI, NL), dtype=datatype)

    return alpha, beta, A, B, C, D


def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_k2mm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 2,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 1,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 1,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
                "CPU_PARALLEL": 1,
                "FOR": 1,
                "MAP": 1,
                "Malloc": 2,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 0,
                "CUDA": 2,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    else:  # rocm
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 0,
                "ROCM": 2,
                "CPU_PARALLEL": 0,
                "VECTORIZE": 0,
                "GEMM": 2,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "k2mm")
