"""Test + benchmark for LULESH ``calc_elem_node_normals``.

``pf`` (shape (ne, 3, 8)) is mutated in place: zeroed then accumulated.
"""

import sys

import numpy as np
import pytest

import lulesh
from harness import SDFGVerification, check_flat_kernel, run_flat_benchmark


def _inputs(ne, seed=3):
    rng = np.random.default_rng(seed)
    pf = rng.random((ne, 3, 8))
    x = rng.random((ne, 8))
    y = rng.random((ne, 8))
    z = rng.random((ne, 8))
    return pf, x, y, z


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_elem_node_normals(target):
    check_flat_kernel(lulesh.calc_elem_node_normals, target, _inputs(27))


if __name__ == "__main__":
    run_flat_benchmark(
        lulesh.calc_elem_node_normals,
        "calc_elem_node_normals",
        lambda nx: _inputs(nx**3),
    )
