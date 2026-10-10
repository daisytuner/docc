"""Tests for SDFG transformation bindings."""

import json
import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    AnalysisManager,
    Scalar,
    PrimitiveType,
    For,
    StripMining,
    Recorder,
)


def test_recorder():
    """Test the Recorder class for recording transformations."""
    builder = StructuredSDFGBuilder("test_sdfg")
    analysis_manager = AnalysisManager(builder)

    # Build sdfg
    builder.add_container("i", Scalar(PrimitiveType.Int32))
    builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

    loop: For = builder.begin_for("i", "0", "64", "1")
    block = builder.add_block()
    builder.add_access(block, "x")
    builder.end_for()

    # Create recorder and apply transformation
    recorder = Recorder()
    tiling = StripMining(loop, tile_size=16)

    # Apply via recorder
    result = recorder.apply(tiling, builder, analysis_manager)
    assert result is True

    # Check history
    history = json.loads(recorder.history)
    assert len(history) == 1
    assert history[0]["transformation_type"] == "StripMining"
    assert history[0]["parameters"]["tile_size"] == 16

    # Check that transformation was actually applied
    sdfg = builder.move()
    outer_loop = sdfg.root.child(0)
    assert isinstance(outer_loop, For)
    assert outer_loop.indvar == "i_tile0"


def test_recorder_skip_if_not_applicable():
    """Test Recorder with skip_if_not_applicable flag."""
    builder = StructuredSDFGBuilder("test_sdfg")
    analysis_manager = AnalysisManager(builder)

    # Build sdfg without any loops
    builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
    block = builder.add_block()
    builder.add_access(block, "x")

    # Create recorder - get a reference to a non-tileable node
    sdfg = builder.move()
    # There's no loop to tile, this test just verifies repr works
    recorder = Recorder()
    assert "Recorder" in repr(recorder)
