"""Test + benchmark for LULESH ``calc_position_for_nodes``."""

import sys

import pytest

import lulesh
from harness import SDFGVerification, check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("xd", "yd", "zd")
_COMPARE = ("x", "y", "z")
_EXTRA = (1e-3,)  # dt


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_position_for_nodes(target):
    check_domain_kernel(
        lulesh.calc_position_for_nodes,
        target,
        randomize=_RANDOMIZE,
        extra_args=_EXTRA,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_position_for_nodes,
        "calc_position_for_nodes",
        randomize=_RANDOMIZE,
        extra_args=_EXTRA,
    )
