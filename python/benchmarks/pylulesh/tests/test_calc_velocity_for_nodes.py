"""Test + benchmark for LULESH ``calc_velocity_for_nodes``."""

import sys

import pytest

import lulesh
from harness import SDFGVerification, check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("xd", "yd", "zd", "xdd", "ydd", "zdd")
_COMPARE = ("xd", "yd", "zd")
_EXTRA = (1e-3, 1e-7)  # dt, u_cut


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_calc_velocity_for_nodes(target):
    check_domain_kernel(
        lulesh.calc_velocity_for_nodes,
        target,
        randomize=_RANDOMIZE,
        extra_args=_EXTRA,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.calc_velocity_for_nodes,
        "calc_velocity_for_nodes",
        randomize=_RANDOMIZE,
        extra_args=_EXTRA,
    )
