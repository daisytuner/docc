import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 65, "N": 80},
    "M": {"M": 200, "N": 250},
    "L": {"M": 600, "N": 700},
    "paper": {"M": 1000, "N": 1200},
}


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    i = np.arange(M).reshape(-1, 1)
    j = np.arange(M)
    A = ((i * j) % M) / M
    for i in range(M):
        A[i, i] = 1.0
    i = np.arange(M).reshape(-1, 1)
    j = np.arange(N)
    B = ((N + i - j) % N) / N

    return alpha, A, B


def kernel(alpha, A, B):

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i, j] += np.dot(A[i + 1 :, i], B[i + 1 :, j])
    B *= alpha


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        "cuda",
        # "rocm"
    ],
)
def test_trmm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 4, "GEMM": 1, "SEQUENTIAL": 6, "FOR": 2}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "VECTORIZE": 2,
                "MAP": 4,
                "GEMM": 1,
                "SEQUENTIAL": 4,
                "FOR": 2,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 2,
                "MAP": 2,
                "GEMM": 1,
                "SEQUENTIAL": 2,
                "FOR": 2,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "SEQUENTIAL": 1,
                "REDUCE": 1,
                "CUDA": 4,
                "MAP": 4,
                "CUDAOffloading": 2,
            },
            rtol=5 - 1,
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "SEQUENTIAL": 1,
                "REDUCE": 1,
                "CUDA": 4,
                "MAP": 4,
                "CUDAOffloading": 2,
            },
            rtol=5 - 1,
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "trmm")
