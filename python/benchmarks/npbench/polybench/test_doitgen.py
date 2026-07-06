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
    A = np.fromfunction(
        lambda i, j, k: ((i * j + k) % NP) / NP, (NR, NQ, NP), dtype=datatype
    )
    C4 = np.fromfunction(lambda i, j: (i * j % NP) / NP, (NP, NP), dtype=datatype)

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
