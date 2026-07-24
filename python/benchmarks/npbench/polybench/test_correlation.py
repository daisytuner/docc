import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 500, "N": 600},
    "M": {"M": 1400, "N": 1800},
    "L": {"M": 3200, "N": 4000},
    "paper": {"M": 1200, "N": 1400},
}


def initialize(M, N, datatype=np.float64):
    float_n = datatype(N)
    i = np.arange(N, dtype=datatype).reshape(-1, 1)
    j = np.arange(M, dtype=datatype)
    data = (i * j) / M + i

    return M, float_n, data


def kernel(M, float_n, data):

    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1 : M, i] = corr[i, i + 1 : M] = data[:, i] @ data[:, i + 1 : M]

    return corr


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda",
        # "rocm"
    ],
)
def test_correlation(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "CMath": 2,
                "REDUCE": 3,
                "MAP": 25,
                "FOR": 1,
                "SEQUENTIAL": 29,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "CMath": 2,
                "VECTORIZE": 16,
                "REDUCE": 3,
                "MAP": 22,
                "FOR": 1,
                "SEQUENTIAL": 10,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "VECTORIZE": 5,
                "REDUCE": 3,
                "CMath": 2,
                "CPU_PARALLEL": 14,
                "MAP": 16,
                "SEQUENTIAL": 1,
                "FOR": 1,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "CMath": 2,
                "CUDA": 21,
                "SEQUENTIAL": 2,
                "Memset": 1,
                "FOR": 27,
                "MAP": 23,
                "CUDAOffloading": 52,
                "Malloc": 7,
            }
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "CMath": 2,
                "ROCM": 21,
                "SEQUENTIAL": 2,
                "Memset": 1,
                "FOR": 27,
                "MAP": 23,
                "ROCMOffloading": 52,
                "Malloc": 7,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "correlation")
