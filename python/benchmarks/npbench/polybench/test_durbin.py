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


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda",
        # "rocm"
    ],
)
def test_durbin(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"REDUCE": 1, "MAP": 4, "SEQUENTIAL": 6, "FOR": 1}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "REDUCE": 1,
                "VECTORIZE": 5,
                "MAP": 4,
                "SEQUENTIAL": 1,
                "FOR": 1,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 1,
                "REDUCE": 1,
                "VECTORIZE": 4,
                "MAP": 4,
                "SEQUENTIAL": 1,
                "FOR": 1,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={
                "CUDA": 1,
                "REDUCE": 1,
                "CUDAOffloading": 4,
                "MAP": 4,
                "SEQUENTIAL": 5,
                "FOR": 1,
            }
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={
                "ROCM": 1,
                "REDUCE": 1,
                "ROCMOffloading": 4,
                "MAP": 4,
                "SEQUENTIAL": 5,
                "FOR": 1,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "durbin")
