"""Test + benchmark for LULESH ``integrate_stress_for_elems``.

Accumulates element stress contributions into the global nodal force arrays
(``fx/fy/fz``) via a scatter-add over the element node list, and returns the
per-element volume ``determ``. ``fx/fy/fz`` must start zeroed.
"""

import sys

import numpy as np
import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

_ZEROS = ("fx", "fy", "fz")
_COMPARE = ("fx", "fy", "fz")


def _sig(ne, seed=7):
    rng = np.random.default_rng(seed)
    return rng.random(ne), rng.random(ne), rng.random(ne)


def _extra(d):
    return _sig(d.numelem)


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize(
    "target",
    [
        "none",
        "sequential",
        # "openmp",
        # "cuda",
        # "rocm"
    ],
)
def test_integrate_stress_for_elems(target):
    check_domain_kernel(
        lulesh.integrate_stress_for_elems,
        target,
        zeros=_ZEROS,
        extra_args=_extra,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.integrate_stress_for_elems,
        "integrate_stress_for_elems",
        zeros=_ZEROS,
        extra_args=_extra,
    )
