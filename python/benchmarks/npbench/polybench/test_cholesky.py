import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {"S": {"N": 100}, "M": {"N": 300}, "L": {"N": 900}, "paper": {"N": 2000}}


def initialize(N, datatype=np.float64):
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, : i + 1] = (-np.arange(i + 1, dtype=datatype) % N) / N + 1
        A[i, i + 1 :] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)

    return A


def kernel(A):

    A[0, 0] = np.sqrt(A[0, 0])
    for i in range(1, A.shape[0]):
        for j in range(i):
            A[i, j] -= np.dot(A[i, :j], A[j, :j])
            A[i, j] /= A[j, j]
        A[i, i] -= np.dot(A[i, :i], A[i, :i])
        A[i, i] = np.sqrt(A[i, i])


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Instrumentation not supported on macOS"
)
@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        "cuda",
        # "rocm",
    ],
)
def test_cholesky(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "REDUCE": 2,
                "SEQUENTIAL": 4,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={"VECTORIZE": 2, "REDUCE": 2, "SEQUENTIAL": 2, "FOR": 2}
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "VECTORIZE": 2,
                "REDUCE": 2,
                "SEQUENTIAL": 2,
                "FOR": 2,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "REDUCE": 2,
                "SEQUENTIAL": 4,
            }
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "FOR": 2,
                "REDUCE": 2,
                "SEQUENTIAL": 4,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "cholesky")
