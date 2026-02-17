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
    data = np.fromfunction(lambda i, j: (i * j) / M, (N, M), dtype=datatype)

    return M, float_n, data


def kernel(M, float_n, data):

    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


@pytest.mark.skip(reason="sdfg does not validate")
@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_covariance(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "Memset": 1,
                "SEQUENTIAL": 10,
                "FOR": 12,
                "MAP": 10,
                "Malloc": 4,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "HIGHWAY": 5,
                "Memset": 1,
                "SEQUENTIAL": 5,
                "FOR": 12,
                "MAP": 10,
                "Malloc": 4,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 1,
                "GEMM": 1,
                "CPU_PARALLEL": 8,
                "Memset": 1,
                "SEQUENTIAL": 1,
                "FOR": 12,
                "MAP": 10,
                "Malloc": 4,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(verification={})
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "covariance")
