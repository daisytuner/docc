import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 35, "N": 50},
    "M": {"M": 110, "N": 140},
    "L": {"M": 350, "N": 400},
    "paper": {"M": 1000, "N": 1200},
}


def initialize(M, N, datatype=np.float64):
    alpha = datatype(1.5)
    beta = datatype(1.2)
    i = np.arange(N).reshape(-1, 1)
    j = np.arange(N)
    C = (i * j + 3) % N / M
    j = np.arange(M)
    A = (i * j + 1) % N / N
    B = (i * j + 2) % M / M
    return alpha, beta, C, A, B


def kernel(alpha, beta, C, A, B):
    for i in range(A.shape[0]):
        C[i, : i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, : i + 1] += (
                A[: i + 1, k] * alpha * B[i, k] + B[: i + 1, k] * alpha * A[i, k]
            )


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_syr2k(target):
    verifier = None
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 6, "SEQUENTIAL": 8, "FOR": 2},
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={"VECTORIZE": 6, "MAP": 6, "SEQUENTIAL": 2, "FOR": 2},
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"VECTORIZE": 6, "MAP": 6, "SEQUENTIAL": 2, "FOR": 2},
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={"MAP": 6, "SEQUENTIAL": 8, "FOR": 2},
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={"MAP": 6, "SEQUENTIAL": 8, "FOR": 2},
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "syr2k")
