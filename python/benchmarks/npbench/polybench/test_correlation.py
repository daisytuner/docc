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
    data = np.fromfunction(lambda i, j: (i * j) / M + i, (N, M), dtype=datatype)

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


@pytest.mark.skip(reason="sdfg does not validate")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_correlation(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "CMath": 2,
                "Memset": 1,
                "SEQUENTIAL": 23,
                "FOR": 27,
                "MAP": 23,
                "Malloc": 7,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "CMath": 2,
                "HIGHWAY": 12,
                "Memset": 1,
                "SEQUENTIAL": 11,
                "FOR": 27,
                "MAP": 23,
                "Malloc": 7,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "HIGHWAY": 3,
                "CMath": 2,
                "CPU_PARALLEL": 17,
                "Memset": 1,
                "SEQUENTIAL": 3,
                "FOR": 27,
                "MAP": 23,
                "Malloc": 7,
            }
        )
    else:  # cuda
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
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "correlation")
