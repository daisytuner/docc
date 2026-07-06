"""Test + benchmark for LULESH ``collect_domain_nodes_to_elem_nodes``."""

import sys

import pytest

import lulesh
from harness import SDFGVerification, check_domain_kernel, run_domain_benchmark

_RANDOMIZE = ("x", "y", "z")


def _extra(d):
    return (d.nodelist,)


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_collect_domain_nodes_to_elem_nodes(target):
    check_domain_kernel(
        lulesh.collect_domain_nodes_to_elem_nodes,
        target,
        randomize=_RANDOMIZE,
        extra_args=_extra,
        compare_fields=(),
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.collect_domain_nodes_to_elem_nodes,
        "collect_domain_nodes_to_elem_nodes",
        randomize=_RANDOMIZE,
        extra_args=_extra,
    )
