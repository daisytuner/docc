import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 40, "N": 50},
    "M": {"M": 120, "N": 150},
    "L": {"M": 350, "N": 550},
    "paper": {"M": 1000, "N": 1200},
}


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    i = np.arange(M).reshape(-1, 1)
    j = np.arange(N)
    C = ((i + j) % 100) / M
    B = ((N + i - j) % 100) / M
    A = np.empty((M, M), dtype=datatype)
    for i in range(M):
        A[i, : i + 1] = ((i + np.arange(i + 1)) % 100) / M
        A[i, i + 1 :] = -999

    return alpha, beta, C, A, B


def kernel(alpha, beta, C, A, B):

    temp2 = np.empty((C.shape[1],), dtype=C.dtype)
    C *= beta
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[:i, j] += alpha * B[i, j] * A[i, :i]
            temp2[j] = B[:i, j] @ A[i, :i]
        C[i, :] += alpha * B[i, :] * A[i, i] + alpha * temp2


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
def test_symm(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"GEMM": 1, "FOR": 2, "SEQUENTIAL": 11, "MAP": 9},
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "VECTORIZE": 7,
                "FOR": 2,
                "SEQUENTIAL": 4,
                "MAP": 9,
            },
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "GEMM": 1,
                "VECTORIZE": 5,
                "SEQUENTIAL": 2,
                "FOR": 2,
                "CPU_PARALLEL": 2,
                "MAP": 7,
            },
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "REDUCE": 1,
                "SEQUENTIAL": 8,
                "FOR": 2,
                "CUDA": 4,
                "MAP": 9,
                "CUDAOffloading": 4,
            }
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "REDUCE": 1,
                "SEQUENTIAL": 8,
                "FOR": 2,
                "ROCM": 4,
                "MAP": 9,
                "ROCMOffloading": 4,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "symm")
