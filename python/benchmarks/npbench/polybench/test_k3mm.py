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
    i = np.arange(NI).reshape(-1, 1)
    j = np.arange(NK)
    A = ((i * j + 1) % NI) / (5 * NI)
    i = np.arange(NK).reshape(-1, 1)
    j = np.arange(NJ)
    B = ((i * (j + 1) + 2) % NJ) / (5 * NJ)
    i = np.arange(NJ).reshape(-1, 1)
    j = np.arange(NM)
    C = (i * (j + 3) % NL) / (5 * NL)
    i = np.arange(NM).reshape(-1, 1)
    j = np.arange(NL)
    D = ((i * (j + 2) + 2) % NK) / (5 * NK)

    return A, B, C, D


def kernel(A, B, C, D):

    return A @ B @ C @ D


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_k3mm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "SEQUENTIAL": 2,
                "GEMM": 3,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "SEQUENTIAL": 1,
                "VECTORIZE": 1,
                "GEMM": 3,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 1,
                "MAP": 1,
                "GEMM": 3,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "CUDA": 2,
                "CUDAOffloading": 4,
                "GEMM": 3,
            },
            device_resident=True,
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "ROCM": 2,
                "ROCMOffloading": 4,
                "GEMM": 3,
            },
            device_resident=True,
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "k3mm")
