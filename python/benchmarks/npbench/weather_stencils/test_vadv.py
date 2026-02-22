import sys
import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

# Sample constants
BET_M = 0.5
BET_P = 0.5

PARAMETERS = {
    "S": {"I": 60, "J": 60, "K": 40},
    "M": {"I": 112, "J": 112, "K": 80},
    "L": {"I": 180, "J": 180, "K": 160},
    "paper": {"I": 256, "J": 256, "K": 160},
}


def initialize(I, J, K, datatype=np.float64):
    from numpy.random import default_rng

    rng = default_rng(42)

    dtr_stage = datatype(3.0 / 20.0)

    # Define arrays
    utens_stage = rng.random((I, J, K)).astype(datatype)
    u_stage = rng.random((I, J, K)).astype(datatype)
    wcon = rng.random((I + 1, J, K)).astype(datatype)
    u_pos = rng.random((I, J, K)).astype(datatype)
    utens = rng.random((I, J, K)).astype(datatype)

    return utens_stage, u_stage, wcon, u_pos, utens, dtr_stage


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L111
# Note: Using lowercase ni, nj, nk to avoid SymEngine interpreting I/J as imaginary units
def kernel(utens_stage, u_stage, wcon, u_pos, utens, dtr_stage):
    ni, nj, nk = utens_stage.shape[0], utens_stage.shape[1], utens_stage.shape[2]
    ccol = np.ndarray((ni, nj, nk), dtype=utens_stage.dtype)
    dcol = np.ndarray((ni, nj, nk), dtype=utens_stage.dtype)
    data_col = np.ndarray((ni, nj), dtype=utens_stage.dtype)

    for k in range(1):
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M

        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - ccol[:, :, k]

        # update the d column
        correction_term = -cs * (u_stage[:, :, k + 1] - u_stage[:, :, k])
        dcol[:, :, k] = (
            dtr_stage * u_pos[:, :, k]
            + utens[:, :, k]
            + utens_stage[:, :, k]
            + correction_term
        )

        # Thomas forward
        divided = 1.0 / bcol
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = dcol[:, :, k] * divided

    for k in range(1, nk - 1):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        gcv = 0.25 * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])

        as_ = gav * BET_M
        cs = gcv * BET_M

        acol = gav * BET_P
        ccol[:, :, k] = gcv * BET_P
        bcol = dtr_stage - acol - ccol[:, :, k]

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k]) - cs * (
            u_stage[:, :, k + 1] - u_stage[:, :, k]
        )
        dcol[:, :, k] = (
            dtr_stage * u_pos[:, :, k]
            + utens[:, :, k]
            + utens_stage[:, :, k]
            + correction_term
        )

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        ccol[:, :, k] = ccol[:, :, k] * divided
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(nk - 1, nk):
        gav = -0.25 * (wcon[1:, :, k] + wcon[:-1, :, k])
        as_ = gav * BET_M
        acol = gav * BET_P
        bcol = dtr_stage - acol

        # update the d column
        correction_term = -as_ * (u_stage[:, :, k - 1] - u_stage[:, :, k])
        dcol[:, :, k] = (
            dtr_stage * u_pos[:, :, k]
            + utens[:, :, k]
            + utens_stage[:, :, k]
            + correction_term
        )

        # Thomas forward
        divided = 1.0 / (bcol - ccol[:, :, k - 1] * acol)
        dcol[:, :, k] = (dcol[:, :, k] - (dcol[:, :, k - 1]) * acol) * divided

    for k in range(nk - 1, nk - 2, -1):
        datacol = dcol[:, :, k]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])

    for k in range(nk - 2, -1, -1):
        datacol = dcol[:, :, k] - ccol[:, :, k] * data_col[:, :]
        data_col[:] = datacol
        utens_stage[:, :, k] = dtr_stage * (datacol - u_pos[:, :, k])


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        "openmp",
        # "cuda"
    ],
)
def test_vadv(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"MAP": 141, "SEQUENTIAL": 141, "FOR": 165, "Malloc": 69}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 35,
                "MAP": 141,
                "SEQUENTIAL": 106,
                "FOR": 165,
                "Malloc": 69,
            }
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 35,
                "MAP": 141,
                "SEQUENTIAL": 106,
                "FOR": 165,
                "Malloc": 69,
            }
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={
                "HIGHWAY": 35,
                "MAP": 141,
                "SEQUENTIAL": 106,
                "FOR": 165,
                "Malloc": 69,
            }
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "vadv")
