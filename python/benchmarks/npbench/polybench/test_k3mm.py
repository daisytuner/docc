import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"NI": 800, "NJ": 850, "NK": 900, "NL": 950, "NM": 1000},
    "M": {"NI": 2000, "NJ": 2200, "NK": 2400, "NL": 2600, "NM": 2800},
    "L": {"NI": 5500, "NJ": 6000, "NK": 6500, "NL": 7000, "NM": 7500},
    "paper": {"NI": 3200, "NJ": 3600, "NK": 4000, "NL": 4400, "NM": 4800},
}


def initialize(NI, NJ, NK, NL, NM, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j: ((i * j + 1) % NI) / (5 * NI), (NI, NK), dtype=datatype
    )
    B = np.fromfunction(
        lambda i, j: ((i * (j + 1) + 2) % NJ) / (5 * NJ), (NK, NJ), dtype=datatype
    )
    C = np.fromfunction(
        lambda i, j: (i * (j + 3) % NL) / (5 * NL), (NJ, NM), dtype=datatype
    )
    D = np.fromfunction(
        lambda i, j: ((i * (j + 2) + 2) % NK) / (5 * NK), (NM, NL), dtype=datatype
    )

    return A, B, C, D


def kernel(A, B, C, D):

    return A @ B @ C @ D


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_k3mm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "MAP": 2,
                "SEQUENTIAL": 2,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 3,
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
                "HIGHWAY": 1,
                "GEMM": 3,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "Free": 3,
                "CPU_PARALLEL": 1,
                "FOR": 1,
                "MAP": 1,
                "GEMM": 3,
                "Malloc": 3,
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
                "HIGHWAY": 0,
                "GEMM": 3,
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
                "HIGHWAY": 0,
                "GEMM": 3,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "k3mm")
