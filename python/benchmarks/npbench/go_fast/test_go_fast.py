# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 2000},
    "M": {"N": 6000},
    "L": {"N": 20000},
    "paper": {"N": 12500},
}


def initialize(N):
    from numpy.random import default_rng

    rng = default_rng(42)
    a = rng.random((N, N), dtype=np.float64)
    return a


def kernel(a):
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    return a + trace


@pytest.mark.parametrize(
    "target",
    ["none", "sequential", "openmp", "cuda"],
)
def test_go_fast(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 6, "Malloc": 2, "CMath": 1, "SEQUENTIAL": 6, "FOR": 7}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 3,
                "MAP": 6,
                "Malloc": 2,
                "CMath": 1,
                "SEQUENTIAL": 3,
                "FOR": 7,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 3,
                "MAP": 6,
                "Malloc": 2,
                "CPU_PARALLEL": 3,
                "CMath": 1,
                "FOR": 7,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "CUDA": 6,
                "MAP": 6,
                "CUDAOffloading": 8,
                "Malloc": 2,
                "CMath": 1,
                "FOR": 7,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "go_fast")
