import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TMAX": 20, "NX": 200, "NY": 220},
    "M": {"TMAX": 60, "NX": 400, "NY": 450},
    "L": {"TMAX": 150, "NX": 800, "NY": 900},
    "paper": {"TMAX": 500, "NX": 1000, "NY": 1200},
}


def initialize(TMAX, NX, NY, datatype=np.float64):
    ex = np.fromfunction(lambda i, j: (i * (j + 1)) / NX, (NX, NY), dtype=datatype)
    ey = np.fromfunction(lambda i, j: (i * (j + 2)) / NY, (NX, NY), dtype=datatype)
    hz = np.fromfunction(lambda i, j: (i * (j + 3)) / NX, (NX, NY), dtype=datatype)
    _fict_ = np.fromfunction(lambda i: i, (TMAX,), dtype=datatype)

    return TMAX, ex, ey, hz, _fict_


def kernel(TMAX, ex, ey, hz, _fict_):
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_fdtd_2d(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"Malloc": 11, "MAP": 29, "SEQUENTIAL": 29, "FOR": 30}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "Malloc": 11,
                "HIGHWAY": 15,
                "MAP": 29,
                "SEQUENTIAL": 14,
                "FOR": 30,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "Malloc": 11,
                "HIGHWAY": 15,
                "MAP": 29,
                "SEQUENTIAL": 14,
                "FOR": 30,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={"Malloc": 11, "MAP": 29, "SEQUENTIAL": 29, "FOR": 30}
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "fdtd_2d")
