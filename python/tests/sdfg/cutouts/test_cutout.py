"""Tests for cutout utility bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    AnalysisManager,
    cutout,
    Scalar,
    PrimitiveType,
)


def test_cutout():
    """Test cutting out the outer loop of nested loops."""
    # Create an SDFG with nested loops
    builder = StructuredSDFGBuilder("source_sdfg")
    builder.add_container("i", Scalar(PrimitiveType.Int32))
    builder.add_container("j", Scalar(PrimitiveType.Int32))

    builder.begin_for("i", "0", "10", "1")
    builder.begin_for("j", "0", "20", "1")
    builder.add_block()
    builder.end_for()
    builder.end_for()

    # Finalize the SDFG
    sdfg = builder.move()

    # Get the outer loop
    analysis = AnalysisManager(sdfg)
    loop_analysis = analysis.loop_analysis()
    j_loop = loop_analysis.find_loop_by_indvar("j")
    assert j_loop is not None

    # Create the cutout of the outer loop (includes inner loop)
    cutout_sdfg = cutout(sdfg, analysis, j_loop)

    # Verify the cutout
    assert cutout_sdfg is not None

    # The cutout should contain the nested loop structure
    cutout_analysis = AnalysisManager(cutout_sdfg)
    cutout_loop_analysis = cutout_analysis.loop_analysis()
    cutout_loops = cutout_loop_analysis.loops()
    assert len(cutout_loops) == 1
    cutout_loop = cutout_loops[0]
    assert cutout_loop.indvar == "j"
