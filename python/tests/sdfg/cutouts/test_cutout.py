"""Tests for cutout utility bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    AnalysisManager,
    LoopTiling,
    apply_cutout,
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


def _two_nests():
    builder = StructuredSDFGBuilder("two_nests")
    builder.add_container("N", Scalar(PrimitiveType.Int64), True)
    for name in ("i", "j", "k"):
        builder.add_container(name, Scalar(PrimitiveType.Int64))

    builder.begin_for("i", "0", "N", "1")
    builder.begin_for("j", "0", "N", "1")
    builder.add_block()
    builder.end_for()
    builder.end_for()

    builder.begin_for("k", "0", "N", "1")
    builder.add_block()
    builder.end_for()

    return builder.move()


def _outermost(sdfg, indvar):
    loops = AnalysisManager(sdfg).loop_analysis().outermost_loops()
    return next(loop for loop in loops if loop.indvar == indvar)


def test_apply_cutout_identity_preserves_signature():
    sdfg = _two_nests()
    arguments_before = list(sdfg.arguments)
    node = _outermost(sdfg, "i")

    piece = cutout(sdfg, AnalysisManager(sdfg), node)
    apply_cutout(sdfg, int(node.element_id), piece)

    assert list(sdfg.arguments) == arguments_before
    outermost = AnalysisManager(sdfg).loop_analysis().outermost_loops()
    assert len(outermost) == 2
    assert all(int(loop.element_id) != int(node.element_id) for loop in outermost)


def test_apply_cutout_merges_transformed_cutout():
    sdfg = _two_nests()
    node = _outermost(sdfg, "k")
    piece = cutout(sdfg, AnalysisManager(sdfg), node)

    analysis = AnalysisManager(piece)
    target = analysis.loop_analysis().outermost_loops()[0]
    piece_builder = StructuredSDFGBuilder(piece)
    tiling = LoopTiling(target, 32)
    assert tiling.can_be_applied(piece_builder, analysis)
    tiling.apply(piece_builder, analysis)

    loops_before = len(AnalysisManager(sdfg).loop_analysis().loops())
    apply_cutout(sdfg, int(node.element_id), piece)

    assert len(AnalysisManager(sdfg).loop_analysis().loops()) == loops_before + 1


def test_apply_cutout_unknown_element_id_raises():
    sdfg = _two_nests()
    piece = cutout(sdfg, AnalysisManager(sdfg), _outermost(sdfg, "i"))

    with pytest.raises(RuntimeError):
        apply_cutout(sdfg, 999_999_999, piece)
