import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"N": 2000},
    "M": {"N": 5000},
    "L": {"N": 14000},
    "paper": {"N": 16000},
}


def initialize(N, datatype=np.float64):
    i = np.arange(N, dtype=datatype).reshape(-1, 1)
    j = np.arange(N, dtype=datatype)
    L = (i + N - j + 1) * 2 / N
    x = np.full((N,), -999, dtype=datatype)
    b = np.arange(N, dtype=datatype)

    return L, x, b


def kernel(L, x, b):

    for i in range(x.shape[0]):
        x[i] = (b[i] - L[i, :i] @ x[:i]) / L[i, i]


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_trisolv(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"REDUCE": 1, "SEQUENTIAL": 2, "FOR": 1}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={"VECTORIZE": 1, "REDUCE": 1, "SEQUENTIAL": 1, "FOR": 1}
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"VECTORIZE": 1, "REDUCE": 1, "SEQUENTIAL": 1, "FOR": 1}
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={"REDUCE": 1, "SEQUENTIAL": 2, "FOR": 1}
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={"REDUCE": 1, "SEQUENTIAL": 2, "FOR": 1}
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "trisolv")
