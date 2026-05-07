import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {"S": {"N": 60}, "M": {"N": 220}, "L": {"N": 700}, "paper": {"N": 2000}}


def initialize(N, datatype=np.float64):
    A = np.empty((N, N), dtype=datatype)
    for i in range(N):
        A[i, : i + 1] = np.fromfunction(
            lambda j: (-j % N) / N + 1, (i + 1,), dtype=datatype
        )
        A[i, i + 1 :] = 0.0
        A[i, i] = 1.0
    A[:] = A @ np.transpose(A)

    return A


def kernel(A):

    for i in range(A.shape[0]):
        for j in range(i):
            A[i, j] -= A[i, :j] @ A[:j, j]
            A[i, j] /= A[j, j]
        for j in range(i, A.shape[0]):
            A[i, j] -= A[i, :i] @ A[:i, j]


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault")
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
def test_lu(target):
    if target == "none":
        verifier = SDFGVerification(verification={"MAP": 1, "SEQUENTIAL": 1, "FOR": 5})
    elif target == "sequential":
        verifier = SDFGVerification(verification={"MAP": 1, "SEQUENTIAL": 1, "FOR": 5})
    elif target == "openmp":
        verifier = SDFGVerification(verification={"MAP": 1, "SEQUENTIAL": 1, "FOR": 5})
    elif target == "cuda":
        verifier = SDFGVerification(verification={"MAP": 1, "SEQUENTIAL": 1, "FOR": 5})
    else:  # rocm
        verifier = SDFGVerification(verification={"MAP": 1, "SEQUENTIAL": 1, "FOR": 5})
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "lu")
