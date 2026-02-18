import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"TSTEPS": 5, "N": 100},
    "M": {"TSTEPS": 20, "N": 200},
    "L": {"TSTEPS": 50, "N": 500},
    "paper": {"TSTEPS": 100, "N": 200},
}


def initialize(TSTEPS, N, datatype=np.float64):
    u = np.fromfunction(lambda i, j: (i + N - j) / N, (N, N), dtype=datatype)

    return TSTEPS, N, u


def kernel(TSTEPS, N, u):

    v = np.empty(u.shape, dtype=u.dtype)
    p = np.empty(u.shape, dtype=u.dtype)
    q = np.empty(u.shape, dtype=u.dtype)

    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)

    a = -mul1 / 2.0
    b = 1.0 + mul2
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d

    for t in range(1, TSTEPS + 1):
        v[0, 1 : N - 1] = 1.0
        p[1 : N - 1, 0] = 0.0
        q[1 : N - 1, 0] = v[0, 1 : N - 1]
        for j in range(1, N - 1):
            p[1 : N - 1, j] = -c / (a * p[1 : N - 1, j - 1] + b)
            q[1 : N - 1, j] = (
                -d * u[j, 0 : N - 2]
                + (1.0 + 2.0 * d) * u[j, 1 : N - 1]
                - f * u[j, 2:N]
                - a * q[1 : N - 1, j - 1]
            ) / (a * p[1 : N - 1, j - 1] + b)
        v[N - 1, 1 : N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            v[j, 1 : N - 1] = p[1 : N - 1, j] * v[j + 1, 1 : N - 1] + q[1 : N - 1, j]

        u[1 : N - 1, 0] = 1.0
        p[1 : N - 1, 0] = 0.0
        q[1 : N - 1, 0] = u[1 : N - 1, 0]
        for j in range(1, N - 1):
            p[1 : N - 1, j] = -f / (d * p[1 : N - 1, j - 1] + e)
            q[1 : N - 1, j] = (
                -a * v[0 : N - 2, j]
                + (1.0 + 2.0 * a) * v[1 : N - 1, j]
                - c * v[2:N, j]
                - d * q[1 : N - 1, j - 1]
            ) / (d * p[1 : N - 1, j - 1] + e)
        u[1 : N - 1, N - 1] = 1.0
        for j in range(N - 2, 0, -1):
            u[1 : N - 1, j] = p[1 : N - 1, j] * u[1 : N - 1, j + 1] + q[1 : N - 1, j]

    return u


@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda"])
def test_adi(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 18, "SEQUENTIAL": 18, "FOR": 23, "Malloc": 5}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 4,
                "MAP": 18,
                "SEQUENTIAL": 14,
                "FOR": 23,
                "Malloc": 5,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "CPU_PARALLEL": 1,
                "HIGHWAY": 4,
                "MAP": 18,
                "SEQUENTIAL": 13,
                "FOR": 23,
                "Malloc": 5,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "CUDA": 2,
                "CUDAOffloading": 4,
                "MAP": 18,
                "SEQUENTIAL": 16,
                "FOR": 23,
                "Malloc": 5,
            }
        )

    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "adi")
