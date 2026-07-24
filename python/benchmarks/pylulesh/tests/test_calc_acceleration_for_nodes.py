"""Test + benchmark for LULESH ``calc_acceleration_for_nodes``.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_calc_acceleration_for_nodes.py --nx 20
"""

import sys

import pytest

import lulesh
from harness import SDFGVerification, check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("fx", "fy", "fz")
_COMPARE = ("xdd", "ydd", "zdd")


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_acceleration_for_nodes(target):
    check_domain_kernel(
        lulesh.calc_acceleration_for_nodes,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_acceleration_for_nodes,
        "calc_acceleration_for_nodes",
        randomize=_RANDOMIZE,
    )
