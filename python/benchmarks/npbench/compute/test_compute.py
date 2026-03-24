# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"M": 2000, "N": 2000},
    "M": {"M": 5000, "N": 5000},
    "L": {"M": 16000, "N": 16000},
    "paper": {"M": 12500, "N": 12500},
}


def initialize(M, N):
    from numpy.random import default_rng

    rng = default_rng(42)
    array_1 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    array_2 = rng.uniform(0, 1000, size=(M, N)).astype(np.int64)
    a = np.int64(4)
    b = np.int64(3)
    c = np.int64(9)
    return array_1, array_2, a, b, c


# https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html
def kernel(array_1, array_2, a, b, c):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c


@pytest.mark.parametrize(
    "target",
    ["none", "sequential", "openmp", "cuda"],
)
def test_compute(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"SEQUENTIAL": 2, "FOR": 2, "MAP": 2, "Malloc": 0}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 1,
                "SEQUENTIAL": 1,
                "FOR": 2,
                "MAP": 2,
                "Malloc": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 1,
                "SEQUENTIAL": 0,
                "FOR": 1,
                "MAP": 1,
                "Malloc": 0,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 2,
                "SEQUENTIAL": 0,
                "FOR": 2,
                "MAP": 2,
                "CUDAOffloading": 6,
                "Malloc": 0,
            }
        )
    else:  # rocm
        verifier = SDFGVerification(
            verification={
                "ROCM": 2,
                "FOR": 2,
                "MAP": 2,
                "ROCMOffloading": 6,
                "Malloc": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "compute")
