import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TSTEPS": 8, "N": 50},
    "M": {"TSTEPS": 15, "N": 100},
    "L": {"TSTEPS": 40, "N": 200},
    "paper": {"TSTEPS": 100, "N": 400},
}


def initialize(TSTEPS, N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: (i * (j + 2) + 2) / N, (N, N), dtype=datatype)

    return TSTEPS, N, A


def kernel(TSTEPS, N, A):

    for t in range(0, TSTEPS - 1):
        for i in range(1, N - 1):
            A[i, 1:-1] += (
                A[i - 1, :-2]
                + A[i - 1, 1:-1]
                + A[i - 1, 2:]
                + A[i, 2:]
                + A[i + 1, :-2]
                + A[i + 1, 1:-1]
                + A[i + 1, 2:]
            )
            for j in range(1, N - 1):
                A[i, j] += A[i, j - 1]
                A[i, j] /= 9.0


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_seidel_2d(target):
    if target == "none":
        verifier = SDFGVerification(verification={"MAP": 1, "SEQUENTIAL": 1, "FOR": 4})
    elif target == "sequential":
        verifier = SDFGVerification(verification={"HIGHWAY": 1, "MAP": 1, "FOR": 4})
    elif target == "openmp":
        verifier = SDFGVerification(verification={"HIGHWAY": 1, "MAP": 1, "FOR": 4})
    else:  # cuda
        verifier = SDFGVerification(verification={"MAP": 1, "SEQUENTIAL": 1, "FOR": 4})
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "seidel_2d")
