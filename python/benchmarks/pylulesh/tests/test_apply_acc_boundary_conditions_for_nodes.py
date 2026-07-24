"""Test + benchmark for LULESH ``apply_acc_boundary_conditions_for_nodes``."""

import sys

import pytest

import lulesh
from harness import SDFGVerification, check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("xdd", "ydd", "zdd")
_COMPARE = ("xdd", "ydd", "zdd")


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_apply_acc_boundary_conditions_for_nodes(target):
    check_domain_kernel(
        lulesh.apply_acc_boundary_conditions_for_nodes,
        target,
        randomize=_RANDOMIZE,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.apply_acc_boundary_conditions_for_nodes,
        "apply_acc_boundary_conditions_for_nodes",
        randomize=_RANDOMIZE,
    )
