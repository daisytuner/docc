import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"NR": 60, "NQ": 60, "NP": 128},
    "M": {"NR": 110, "NQ": 125, "NP": 256},
    "L": {"NR": 220, "NQ": 250, "NP": 512},
    "paper": {"NR": 220, "NQ": 250, "NP": 270},
}


def initialize(NR, NQ, NP, datatype=np.float64):
    i = np.arange(NR).reshape(-1, 1, 1)
    j = np.arange(NQ).reshape(-1, 1)
    k = np.arange(NP)
    A = ((i * j + k) % NP) / NP
    i = np.arange(NP).reshape(-1, 1)
    j = np.arange(NP)
    C4 = (i * j % NP) / NP

    return NR, NQ, NP, A, C4


def kernel(NR, NQ, NP, A, C4):
    A[:] = np.reshape(np.reshape(A, (NR, NQ, 1, NP)) @ C4, (NR, NQ, NP))


@pytest.mark.xfail()
@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        pytest.param("cuda", marks=pytest.mark.xfail(reason="nan mismatch")),
        "rocm",
    ],
)
def test_doitgen(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "MAP": 3,
                "GEMM": 1,
                "SEQUENTIAL": 5,
                "FOR": 2,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "VECTORIZE": 1,
                "MAP": 3,
                "GEMM": 1,
                "SEQUENTIAL": 4,
                "FOR": 2,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 1,
                "MAP": 1,
                "GEMM": 1,
                "SEQUENTIAL": 2,
                "FOR": 2,
            }
        )
    elif target == "cuda":
        verifier = SDFGVerification(
            verification={"REDUCE": 1, "SEQUENTIAL": 5, "MAP": 4}
        )
    elif target == "rocm":
        verifier = SDFGVerification(
            verification={"REDUCE": 1, "SEQUENTIAL": 5, "MAP": 4}
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "doitgen")
