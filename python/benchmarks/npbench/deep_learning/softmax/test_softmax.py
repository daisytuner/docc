# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 16, "H": 16, "SM": 128},
    "M": {"N": 32, "H": 8, "SM": 256},
    "L": {"N": 64, "H": 16, "SM": 448},
    "paper": {"N": 64, "H": 16, "SM": 512},
}


def initialize(N, H, SM):
    from numpy.random import default_rng

    rng = default_rng(42)
    x = rng.random((N, H, SM, SM), dtype=np.float32)
    return x


# Numerically-stable version of softmax
def kernel(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda",
        # "rocm",
    ],
)
def test_softmax(target):
    verifier = None
    if target == "none":
        verifier = SDFGVerification(
            verification={"REDUCE": 2, "SEQUENTIAL": 13, "MAP": 11}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "VECTORIZE": 4,
                "REDUCE": 2,
                "SEQUENTIAL": 12,
                "MAP": 14,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "VECTORIZE": 2,
                "REDUCE": 2,
                "CPU_PARALLEL": 4,
                "MAP": 4,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "SEQUENTIAL": 6,
                "MAP": 12,
                "REDUCE": 2,
                "CUDA": 6,
                "CUDAOffloading": 7,
            }
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "SEQUENTIAL": 6,
                "MAP": 12,
                "REDUCE": 2,
                "ROCM": 6,
                "ROCMOffloading": 7,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "softmax")
