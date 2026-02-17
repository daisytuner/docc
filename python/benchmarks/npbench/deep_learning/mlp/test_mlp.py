# Copyright 2021 ETH Zurich and the NPBench authors. All rights reserved.

import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"C_in": 3, "N": 8, "S0": 30000, "S1": 2000, "S2": 2000},
    "M": {"C_in": 3, "N": 8, "S0": 30000, "S1": 10000, "S2": 10000},
    "L": {"C_in": 3, "N": 8, "S0": 30000, "S1": 30000, "S2": 30000},
    "paper": {"C_in": 3, "N": 8, "S0": 30000, "S1": 10000, "S2": 1000},
}


def initialize(C_in, N, S0, S1, S2):
    from numpy.random import default_rng

    rng = default_rng(42)

    mlp_sizes = [S0, S1, S2]  # [300, 100, 10]
    # Inputs
    input = np.random.rand(N, C_in).astype(np.float32)
    # Weights
    w1 = rng.random((C_in, mlp_sizes[0]), dtype=np.float32)
    b1 = rng.random((mlp_sizes[0],), dtype=np.float32)
    w2 = rng.random((mlp_sizes[0], mlp_sizes[1]), dtype=np.float32)
    b2 = rng.random((mlp_sizes[1],), dtype=np.float32)
    w3 = rng.random((mlp_sizes[1], mlp_sizes[2]), dtype=np.float32)
    b3 = rng.random((mlp_sizes[2],), dtype=np.float32)

    return input, w1, b1, w2, b2, w3, b3


def relu(x):
    return np.maximum(x, 0)


# Numerically-stable version of softmax
def softmax(x):
    tmp_max = np.max(x, axis=-1, keepdims=True)
    tmp_out = np.exp(x - tmp_max)
    tmp_sum = np.sum(tmp_out, axis=-1, keepdims=True)
    return tmp_out / tmp_sum


# 3-layer MLP
def kernel(input, w1, b1, w2, b2, w3, b3):
    x = relu(input @ w1 + b1)
    x = relu(x @ w2 + b2)
    x = softmax(x @ w3 + b3)  # Softmax call can be omitted if necessary
    return x


@pytest.mark.parametrize(
    "target",
    ["none", "sequential", "openmp", "cuda"],
)
def test_mlp(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "CMath": 4,
                "SEQUENTIAL": 34,
                "FOR": 36,
                "MAP": 34,
                "GEMM": 3,
                "Malloc": 18,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "CMath": 4,
                "HIGHWAY": 13,
                "SEQUENTIAL": 21,
                "FOR": 36,
                "MAP": 34,
                "GEMM": 3,
                "Malloc": 18,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 13,
                "CMath": 4,
                "CPU_PARALLEL": 18,
                "SEQUENTIAL": 3,
                "FOR": 36,
                "MAP": 34,
                "GEMM": 3,
                "Malloc": 18,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "CMath": 4,
                "CUDA": 34,
                "FOR": 36,
                "MAP": 34,
                "CUDAOffloading": 62,
                "GEMM": 3,
                "Malloc": 18,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "mlp")
