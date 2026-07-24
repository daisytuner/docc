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
    data = (i * j) / M

    return M, float_n, data


def kernel(M, float_n, data):

    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    return cov


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
def test_covariance(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"GEMM": 1, "REDUCE": 1, "MAP": 11, "FOR": 1, "SEQUENTIAL": 13}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "VECTORIZE": 7,
                "REDUCE": 1,
                "MAP": 10,
                "FOR": 1,
                "SEQUENTIAL": 5,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "VECTORIZE": 3,
                "REDUCE": 1,
                "CPU_PARALLEL": 5,
                "MAP": 7,
                "SEQUENTIAL": 1,
                "FOR": 1,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 9,
                "CUDAOffloading": 12,
                "SEQUENTIAL": 4,
                "MAP": 13,
                "REDUCE": 1,
                "FOR": 16,
            }
        )
    elif target == "rocm":
        verifier = SDFGVerification(verification={})
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "covariance")
