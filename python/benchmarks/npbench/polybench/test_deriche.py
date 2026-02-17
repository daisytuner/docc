import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"W": 400, "H": 200},
    "M": {"W": 1500, "H": 1000},
    "L": {"W": 6000, "H": 3000},
    "paper": {"W": 7680, "H": 4320},
}


def initialize(W, H, datatype=np.float64):
    alpha = datatype(0.25)
    imgIn = np.fromfunction(
        lambda i, j: ((313 * i + 991 * j) % 65536) / 65535.0, (W, H), dtype=datatype
    )

    return alpha, imgIn


def kernel(alpha, imgIn):
    k = (
        (1.0 - np.exp(-alpha))
        * (1.0 - np.exp(-alpha))
        / (1.0 + alpha * np.exp(-alpha) - np.exp(2.0 * alpha))
    )
    a1 = a5 = k
    a2 = a6 = k * np.exp(-alpha) * (alpha - 1.0)
    a3 = a7 = k * np.exp(-alpha) * (alpha + 1.0)
    a4 = a8 = -k * np.exp(-2.0 * alpha)
    b1 = 2.0 ** (-alpha)
    b2 = -np.exp(-2.0 * alpha)
    c1 = c2 = 1

    y1 = np.empty_like(imgIn)
    y1[:, 0] = a1 * imgIn[:, 0]
    y1[:, 1] = a1 * imgIn[:, 1] + a2 * imgIn[:, 0] + b1 * y1[:, 0]
    for j in range(2, imgIn.shape[1]):
        y1[:, j] = (
            a1 * imgIn[:, j]
            + a2 * imgIn[:, j - 1]
            + b1 * y1[:, j - 1]
            + b2 * y1[:, j - 2]
        )

    y2 = np.empty_like(imgIn)
    y2[:, -1] = 0.0
    y2[:, -2] = a3 * imgIn[:, -1]
    for j in range(imgIn.shape[1] - 3, -1, -1):
        y2[:, j] = (
            a3 * imgIn[:, j + 1]
            + a4 * imgIn[:, j + 2]
            + b1 * y2[:, j + 1]
            + b2 * y2[:, j + 2]
        )

    imgOut = c1 * (y1 + y2)

    y1[0, :] = a5 * imgOut[0, :]
    y1[1, :] = a5 * imgOut[1, :] + a6 * imgOut[0, :] + b1 * y1[0, :]
    for i in range(2, imgIn.shape[0]):
        y1[i, :] = (
            a5 * imgOut[i, :]
            + a6 * imgOut[i - 1, :]
            + b1 * y1[i - 1, :]
            + b2 * y1[i - 2, :]
        )

    y2[-1, :] = 0.0
    y2[-2, :] = a7 * imgOut[-1, :]
    for i in range(imgIn.shape[0] - 3, -1, -1):
        y2[i, :] = (
            a7 * imgOut[i + 1, :]
            + a8 * imgOut[i + 2, :]
            + b1 * y2[i + 1, :]
            + b2 * y2[i + 2, :]
        )

    imgOut[:] = c2 * (y1 + y2)

    return imgOut


@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_deriche(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={
                "SEQUENTIAL": 62,
                "FOR": 70,
                "MAP": 62,
                "CMath": 9,
                "Malloc": 48,
            }
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 9,
                "MAP": 14,
                "SEQUENTIAL": 5,
                "FOR": 24,
                "CMath": 9,
                "Malloc": 4,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 4,
                "MAP": 14,
                "FOR": 24,
                "CPU_PARALLEL": 10,
                "CMath": 9,
                "Malloc": 4,
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
    run_benchmark(initialize, kernel, PARAMETERS, "deriche")
