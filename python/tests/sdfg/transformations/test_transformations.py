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
)


def test_loop_tiling():
    builder = StructuredSDFGBuilder("test_sdfg")
    analysis_manager = AnalysisManager(builder)

    # Build sdfg
    builder.add_container("i", Scalar(PrimitiveType.Int32))
    builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

    loop: For = builder.begin_for("i", "0", "64", "1")
    block = builder.add_block()
    builder.add_access(block, "x")
    builder.end_for()

    # Apply transformation
    tiling = StripMining(loop, tile_size=16)
    assert tiling.can_be_applied(builder, analysis_manager)
    tiling.apply(builder, analysis_manager)

    # Check result
    sdfg = builder.move()
    outer_loop = sdfg.root.child(0)
    assert isinstance(outer_loop, For)
    assert outer_loop.indvar == "i_tile0"

    inner_loop = outer_loop.body.child(0)
    assert isinstance(inner_loop, For)
    assert inner_loop.indvar == "i"
