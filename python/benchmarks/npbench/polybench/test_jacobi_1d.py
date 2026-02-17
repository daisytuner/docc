import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TSTEPS": 800, "N": 3200},
    "M": {"TSTEPS": 3000, "N": 12000},
    "L": {"TSTEPS": 8500, "N": 34000},
    "paper": {"TSTEPS": 4000, "N": 32000},
}


def initialize(TSTEPS, N, datatype=np.float64):
    A = np.fromfunction(lambda i: (i + 2) / N, (N,), dtype=datatype)
    B = np.fromfunction(lambda i: (i + 3) / N, (N,), dtype=datatype)

    return TSTEPS, A, B


def kernel(TSTEPS, A, B):

    for t in range(1, TSTEPS):
        B[1:-1] = 0.33333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.33333 * (B[:-2] + B[1:-1] + B[2:])


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_jacobi_1d(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 8, "Malloc": 6, "SEQUENTIAL": 8, "FOR": 9}
        )
    elif target == "sequential":
        verifier = SDFGVerification(verification={"HIGHWAY": 2, "MAP": 2, "FOR": 3})
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"CPU_PARALLEL": 2, "MAP": 2, "FOR": 3}
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={"CUDA": 2, "MAP": 2, "CUDAOffloading": 8, "FOR": 3}
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "jacobi_1d")
