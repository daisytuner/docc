import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 1000},
    "M": {"N": 6000},
    "L": {"N": 20000},
    "paper": {"N": 16000},
}


def initialize(N, datatype=np.float64):
    r = np.fromfunction(lambda i: N + 1 - i, (N,), dtype=datatype)
    return r


def kernel(r):
    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha

    return y


@pytest.mark.skip(reason="dot with tensor types not implemented")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_durbin(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 4, "SEQUENTIAL": 4, "FOR": 6, "Malloc": 3}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "FOR": 0,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "FOR": 0,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "FOR": 0,
                "MAP": 0,
                "SEQUENTIAL": 0,
                "CUDA": 0,
                "CPU_PARALLEL": 0,
                "HIGHWAY": 0,
                "GEMM": 0,
                "DOT": 0,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "durbin")
