"""Test + benchmark for LULESH ``calc_elem_volume``."""

import sys

import numpy as np
import pytest

import lulesh
from harness import SDFGVerification, check_flat_kernel, run_flat_benchmark


def _coords(ne, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((ne, 8)), rng.random((ne, 8)), rng.random((ne, 8))


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_elem_volume(target):
    check_flat_kernel(lulesh.calc_elem_volume, target, _coords(27))


if __name__ == "__main__":
    run_flat_benchmark(
        lulesh.calc_elem_volume, "calc_elem_volume", lambda nx: _coords(nx**3)
    )
