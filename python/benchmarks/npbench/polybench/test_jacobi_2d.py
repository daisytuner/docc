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
    i = np.arange(N, dtype=datatype).reshape(-1, 1)
    j = np.arange(N, dtype=datatype)
    A = i * (j + 2) / N
    B = i * (j + 3) / N

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
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_jacobi_2d(target):
    if target == "none":
        verifier = SDFGVerification(verification={"MAP": 4, "SEQUENTIAL": 5, "FOR": 1})
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={"VECTORIZE": 2, "MAP": 4, "SEQUENTIAL": 3, "FOR": 1}
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"CPU_PARALLEL": 2, "SEQUENTIAL": 1, "MAP": 2, "FOR": 1}
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={"CUDA": 4, "MAP": 4, "FOR": 1, "SEQUENTIAL": 1},
            device_resident=True,
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={"ROCM": 4, "MAP": 4, "FOR": 1, "SEQUENTIAL": 1},
            device_resident=True,
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "jacobi_2d")
