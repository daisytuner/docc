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
        # "cuda"
    ],
)
def test_softmax(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "CMath": 2,
                "SEQUENTIAL": 27,
                "FOR": 32,
                "MAP": 27,
                "Malloc": 5,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "CMath": 2,
                "HIGHWAY": 6,
                "SEQUENTIAL": 21,
                "FOR": 32,
                "MAP": 27,
                "Malloc": 5,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 6,
                "CMath": 2,
                "CPU_PARALLEL": 7,
                "SEQUENTIAL": 14,
                "FOR": 32,
                "MAP": 27,
                "Malloc": 5,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "CMath": 2,
                "SEQUENTIAL": 35,
                "FOR": 40,
                "MAP": 35,
                "Malloc": 7,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "softmax")
