import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TSTEPS": 50, "N": 150},
    "M": {"TSTEPS": 80, "N": 350},
    "L": {"TSTEPS": 200, "N": 700},
    "paper": {"TSTEPS": 1000, "N": 2800},
}


def initialize(TSTEPS, N, datatype=np.float64):
    A = np.fromfunction(lambda i, j: i * (j + 2) / N, (N, N), dtype=datatype)
    B = np.fromfunction(lambda i, j: i * (j + 3) / N, (N, N), dtype=datatype)

    return TSTEPS, A, B


def kernel(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1, 1:-1] = 0.2 * (
            A[1:-1, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:] + A[2:, 1:-1] + A[:-2, 1:-1]
        )
        A[1:-1, 1:-1] = 0.2 * (
            B[1:-1, 1:-1] + B[1:-1, :-2] + B[1:-1, 2:] + B[2:, 1:-1] + B[:-2, 1:-1]
        )


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_jacobi_2d(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 24, "Malloc": 10, "SEQUENTIAL": 24, "FOR": 25}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 12,
                "MAP": 24,
                "Malloc": 10,
                "SEQUENTIAL": 12,
                "FOR": 25,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 12,
                "MAP": 24,
                "Malloc": 10,
                "SEQUENTIAL": 12,
                "FOR": 25,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={"MAP": 24, "Malloc": 10, "SEQUENTIAL": 24, "FOR": 25}
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "jacobi_2d")
