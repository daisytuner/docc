"""Test + benchmark for LULESH ``calc_elem_volume_derivative``."""

import sys

import numpy as np
import pytest

import lulesh
from harness import SDFGVerification, check_flat_kernel, run_flat_benchmark


def _coords(ne, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random((ne, 8)), rng.random((ne, 8)), rng.random((ne, 8))


@pytest.mark.skip(reason="Slow in CI; run manually for benchmarking")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_elem_volume_derivative(target):
    check_flat_kernel(lulesh.calc_elem_volume_derivative, target, _coords(27))


if __name__ == "__main__":
    run_flat_benchmark(
        lulesh.calc_elem_volume_derivative,
        "calc_elem_volume_derivative",
        lambda nx: _coords(nx**3),
    )
