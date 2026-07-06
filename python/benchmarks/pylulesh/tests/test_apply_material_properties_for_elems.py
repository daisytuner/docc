"""Test + benchmark for LULESH ``apply_material_properties_for_elems``.

Runs the equation-of-state update (pressure/energy/artificial-viscosity/sound
speed) over the material regions.

Run as a benchmark with::

    python benchmarks/pylulesh/tests/test_apply_material_properties_for_elems.py --nx 20
"""

import sys

import pytest

import lulesh
from harness import check_domain_kernel, run_domain_benchmark

# EOS drivers: relative volume and the energy/pressure state. The remaining
# state fields are zeroed to keep the EOS inputs well-conditioned.
_RANDOMIZE = ("vnew", "e")
_ZEROS = ("p", "q", "qq", "ql", "delv")
_COMPARE = ("p", "e", "q", "ss")


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
@pytest.mark.parametrize("target", ["none", "sequential", "openmp", "cuda", "rocm"])
def test_apply_material_properties_for_elems(target):
    check_domain_kernel(
        lulesh.apply_material_properties_for_elems,
        target,
        randomize=_RANDOMIZE,
        zeros=_ZEROS,
        compare_fields=_COMPARE,
    )


if __name__ == "__main__":
    run_domain_benchmark(
        lulesh.apply_material_properties_for_elems,
        "apply_material_properties_for_elems",
        randomize=_RANDOMIZE,
        zeros=_ZEROS,
    )
