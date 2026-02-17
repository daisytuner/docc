import pytest
import numpy as np
from benchmarks.npbench.harness import SDFGVerification, run_benchmark, run_pytest

PARAMETERS = {
    "S": {"I": 64, "J": 64, "K": 60},
    "M": {"I": 128, "J": 128, "K": 160},
    "L": {"I": 384, "J": 384, "K": 160},
    "paper": {"I": 256, "J": 256, "K": 160},
}


def initialize(I, J, K, datatype=np.float64):
    from numpy.random import default_rng

    rng = default_rng(42)

    # Define arrays
    in_field = rng.random((I + 4, J + 4, K)).astype(datatype)
    out_field = rng.random((I, J, K)).astype(datatype)
    coeff = rng.random((I, J, K)).astype(datatype)

    return in_field, out_field, coeff


# Adapted from https://github.com/GridTools/gt4py/blob/1caca893034a18d5df1522ed251486659f846589/tests/test_integration/stencil_definitions.py#L194
def kernel(in_field, out_field, coeff):
    I, J, K = out_field.shape[0], out_field.shape[1], out_field.shape[2]
    lap_field = 4.0 * in_field[1 : I + 3, 1 : J + 3, :] - (
        in_field[2 : I + 4, 1 : J + 3, :]
        + in_field[0 : I + 2, 1 : J + 3, :]
        + in_field[1 : I + 3, 2 : J + 4, :]
        + in_field[1 : I + 3, 0 : J + 2, :]
    )

    res = lap_field[1:, 1 : J + 1, :] - lap_field[:-1, 1 : J + 1, :]
    flx_field = np.where(
        (res * (in_field[2 : I + 3, 2 : J + 2, :] - in_field[1 : I + 2, 2 : J + 2, :]))
        > 0,
        0,
        res,
    )

    res = lap_field[1 : I + 1, 1:, :] - lap_field[1 : I + 1, :-1, :]
    fly_field = np.where(
        (res * (in_field[2 : I + 2, 2 : J + 3, :] - in_field[2 : I + 2, 1 : J + 2, :]))
        > 0,
        0,
        res,
    )

    out_field[:, :, :] = in_field[2 : I + 2, 2 : J + 2, :] - coeff[:, :, :] * (
        flx_field[1:, :, :]
        - flx_field[:-1, :, :]
        + fly_field[:, 1:, :]
        - fly_field[:, :-1, :]
    )


@pytest.mark.parametrize(
    "target",
    ["none", "sequential", "openmp", "cuda"],
)
def test_hdiff(target):
    if target == "none":
        verifier = SDFGVerification(
            verification={"SEQUENTIAL": 57, "FOR": 63, "MAP": 57, "Malloc": 20}
        )
    elif target == "sequential":
        verifier = SDFGVerification(
            verification={"MAP": 53, "SEQUENTIAL": 53, "FOR": 87, "Malloc": 28}
        )
    elif target == "openmp":
        verifier = SDFGVerification(
            verification={"MAP": 53, "SEQUENTIAL": 53, "FOR": 87, "Malloc": 28}
        )
    else:  # cuda
        verifier = SDFGVerification(
            verification={"MAP": 53, "SEQUENTIAL": 53, "FOR": 87, "Malloc": 28}
        )
    run_pytest(initialize, kernel, PARAMETERS, target, verifier=verifier)


if __name__ == "__main__":
    run_benchmark(initialize, kernel, PARAMETERS, "hdiff")
