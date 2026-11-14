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
    i = np.arange(NI).reshape(-1, 1)
    j = np.arange(NK)
    A = ((i * j + 1) % NI) / NI
    i = np.arange(NK).reshape(-1, 1)
    j = np.arange(NJ)
    B = (i * (j + 1) % NJ) / NJ
    i = np.arange(NJ).reshape(-1, 1)
    j = np.arange(NL)
    C = ((i * (j + 3) + 1) % NL) / NL
    i = np.arange(NI).reshape(-1, 1)
    j = np.arange(NL)
    D = (i * (j + 2) % NK) / NK

    return alpha, beta, A, B, C, D


def kernel(alpha, beta, A, B, C, D):
    D[:] = alpha * A @ B @ C + beta * D


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_k2mm(target):
    # Runtime scaling remains elementwise to preserve NumPy's zero/NaN semantics.
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "MAP": 6,
                "SEQUENTIAL": 6,
                "GEMM": 2,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "MAP": 6,
                "SEQUENTIAL": 3,
                "VECTORIZE": 3,
                "GEMM": 2,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 2,
                "CPU_PARALLEL": 3,
                "MAP": 3,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "MAP": 6,
                "CUDA_Offload": 6,
                "CUDAOffloading": 4,
                "GEMM": 2,
            },
            device_resident=True,
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "MAP": 6,
                "ROCM_Offload": 6,
                "ROCMOffloading": 4,
                "GEMM": 2,
            },
            device_resident=True,
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "k2mm")
