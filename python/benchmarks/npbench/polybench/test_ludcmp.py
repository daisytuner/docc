import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {"S": {"N": 60}, "M": {"N": 220}, "L": {"N": 650}, "paper": {"N": 2000}}


def initialize(N, datatype=np.float64):
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, : i + 1] = (-np.arange(i + 1, dtype=datatype) % N) / N + 1
        A[i, i + 1 :] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)
    fn = datatype(N)
    b = (np.arange(N, dtype=datatype) + 1) / fn / 2.0 + 4.0

    return A, b


def kernel(A, b):

    x = np.zeros_like(b)
    y = np.zeros_like(b)

    for i in range(A.shape[0]):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, A.shape[0]):
            A[i, j] -= A[i, :i] @ A[:i, j]
    for i in range(A.shape[0]):
        y[i] = b[i] - A[i, :i] @ y[:i]
    for i in range(A.shape[0] - 1, -1, -1):
        x[i] = (y[i] - A[i, i + 1 :] @ x[i + 1 :]) / A[i, i]

    return x, y


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_ludcmp(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 2, "REDUCE": 4, "SEQUENTIAL": 10, "FOR": 4}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "MAP": 2,
                "VECTORIZE": 5,
                "REDUCE": 4,
                "SEQUENTIAL": 5,
                "FOR": 4,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 1,
                "MAP": 2,
                "VECTORIZE": 4,
                "REDUCE": 4,
                "SEQUENTIAL": 5,
                "FOR": 4,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 1,
                "REDUCE": 4,
                "SEQUENTIAL": 9,
                "FOR": 4,
                "MAP": 2,
                "CUDAOffloading": 12,
            }
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "ROCM": 1,
                "REDUCE": 4,
                "SEQUENTIAL": 9,
                "FOR": 4,
                "MAP": 2,
                "ROCMOffloading": 12,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "ludcmp")
