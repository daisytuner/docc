"""Test + benchmark for LULESH ``sum_elem_stresses_to_node_forces``."""

import sys

import numpy as np
import pytest

import lulesh
from harness import SDFGVerification, check_flat_kernel, run_flat_benchmark


def _inputs(ne, seed=4):
    rng = np.random.default_rng(seed)
    B = rng.random((ne, 3, 8))
    sigxx = rng.random(ne)
    sigyy = rng.random(ne)
    sigzz = rng.random(ne)
    return B, sigxx, sigyy, sigzz


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_sum_elem_stresses_to_node_forces(target):
    check_flat_kernel(lulesh.sum_elem_stresses_to_node_forces, target, _inputs(27))


if __name__ == "__main__":
    run_flat_benchmark(
        lulesh.sum_elem_stresses_to_node_forces,
        "sum_elem_stresses_to_node_forces",
        lambda nx: _inputs(nx**3),
    )
