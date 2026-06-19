"""Tests for DataFlowGraph bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    AnalysisManager,
    Scalar,
    PrimitiveType,
    TaskletCode,
    SymbolPromotion,
)


class TestPasses:
    """Test suite for SDFG passes."""

    def test_symbol_promotion(self):
        """Test that the SymbolPromotion pass can be created and run."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        block_ptr = builder.add_block()
        x = builder.add_access(block_ptr, "x")
        zero = builder.add_constant(block_ptr, "0", Scalar(PrimitiveType.Int32))
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, zero, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", x, "")

        # Run the SymbolPromotion pass
        analysis_manager = AnalysisManager(builder)
        sp_pass = SymbolPromotion()
        assert sp_pass.run(builder, analysis_manager) == True

        assert len(block_ptr.dataflow.nodes) == 0
