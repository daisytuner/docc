import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TSTEPS": 25, "N": 25},
    "M": {"TSTEPS": 50, "N": 40},
    "L": {"TSTEPS": 100, "N": 70},
    "paper": {"TSTEPS": 500, "N": 120},
}


def initialize(TSTEPS, N, datatype=np.float64):
    A = np.fromfunction(
        lambda i, j, k: (i + j + (N - k)) * 10 / N, (N, N, N), dtype=datatype
    )
    B = np.copy(A)

    return TSTEPS, A, B


def kernel(TSTEPS, A, B):
    for t in range(1, TSTEPS):
        B[1:-1, 1:-1, 1:-1] = (
            0.125 * (A[2:, 1:-1, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[:-2, 1:-1, 1:-1])
            + 0.125
            * (A[1:-1, 2:, 1:-1] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, :-2, 1:-1])
            + 0.125
            * (A[1:-1, 1:-1, 2:] - 2.0 * A[1:-1, 1:-1, 1:-1] + A[1:-1, 1:-1, 0:-2])
            + A[1:-1, 1:-1, 1:-1]
        )
        A[1:-1, 1:-1, 1:-1] = (
            0.125 * (B[2:, 1:-1, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[:-2, 1:-1, 1:-1])
            + 0.125
            * (B[1:-1, 2:, 1:-1] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, :-2, 1:-1])
            + 0.125
            * (B[1:-1, 1:-1, 2:] - 2.0 * B[1:-1, 1:-1, 1:-1] + B[1:-1, 1:-1, 0:-2])
            + B[1:-1, 1:-1, 1:-1]
        )


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_heat_3d(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 15, "SEQUENTIAL": 15, "FOR": 16, "Malloc": 3}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "VECTORIZE": 5,
                "MAP": 15,
                "SEQUENTIAL": 10,
                "FOR": 16,
                "Malloc": 3,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"CPU_PARALLEL": 5, "MAP": 5, "FOR": 6, "Malloc": 3}
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 20,
                "CUDAOffloading": 32,
                "MAP": 62,
                "SEQUENTIAL": 42,
                "FOR": 67,
                "Malloc": 22,
            }
        )
    else:  # rocm
        verifier = SDFGVerification(
            verification={
                "ROCM": 20,
                "ROCMOffloading": 32,
                "MAP": 62,
                "SEQUENTIAL": 42,
                "FOR": 67,
                "Malloc": 22,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "heat_3d")
