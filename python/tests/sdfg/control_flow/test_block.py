"""Tests for Block bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    DebugInfo,
    TaskletCode,
    Block,
)


class TestBlock:
    """Test suite for Block properties."""

    def test_block_parent(self):
        """Test that block has the correct parent"""
        builder = StructuredSDFGBuilder("test_sdfg")

        block_ptr = builder.add_block()

        sdfg = builder.move()
        block = sdfg.root.child(0)

        assert sdfg.root.parent is None

        parent = block.parent
        assert parent is sdfg.root

    def test_dataflow_property(self):
        """Test that dataflow property returns the DataFlowGraph."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        builder.add_access(block_ptr, "x")

        sdfg = builder.move()
        block = sdfg.root.child(0)

        # Block should have a dataflow property
        dataflow = block.dataflow
        assert dataflow is not None

        parent = block.parent
        assert parent is sdfg.root

        # The dataflow graph should contain the access node we added
        nodes = list(dataflow.nodes)
        assert len(nodes) >= 1

    def test_block_is_control_flow_node(self):
        """Test that Block inherits from ControlFlowNode."""
        builder = StructuredSDFGBuilder("test_sdfg")
        block_ptr = builder.add_block()

        sdfg = builder.move()
        block = sdfg.root.child(0)

        # Block should have element_id from ControlFlowNode
        assert block.element_id >= 0

        # Block should have debug_info from ControlFlowNode
        assert block.debug_info is not None

    def test_block_isinstance(self):
        """Test that block is an instance of Block class."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        block = sdfg.root.child(0)

        assert isinstance(block, Block)

    def test_block_with_computation(self):
        """Test a block with computational nodes."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)
        builder.add_container("y", Scalar(PrimitiveType.Float), is_argument=True)

        block_ptr = builder.add_block()
        x_ptr = builder.add_access(block_ptr, "x")
        y_ptr = builder.add_access(block_ptr, "y")
        tasklet_ptr = builder.add_tasklet(
            block_ptr, TaskletCode.assign, ["_in"], ["_out"]
        )
        builder.add_memlet(block_ptr, x_ptr, "", tasklet_ptr, "_in")
        builder.add_memlet(block_ptr, tasklet_ptr, "_out", y_ptr, "")

        sdfg = builder.move()
        block = sdfg.root.child(0)

        # Check dataflow has expected structure
        assert len(list(block.dataflow.nodes)) == 3
        assert len(list(block.dataflow.edges)) == 2
        assert len(list(block.dataflow.tasklets)) == 1

    def test_block_repr(self):
        """Test Block string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        block = sdfg.root.child(0)

        repr_str = repr(block)
        assert "Block" in repr_str
        assert "id=" in repr_str

    def test_multiple_blocks(self):
        """Test SDFG with multiple blocks."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Float), is_argument=True)

        # Add first block
        block1_ptr = builder.add_block()
        builder.add_access(block1_ptr, "x")

        # Add second block
        block2_ptr = builder.add_block()
        builder.add_access(block2_ptr, "x")

        sdfg = builder.move()

        # Root sequence should have 2 children
        assert sdfg.root.size == 2

        block1 = sdfg.root.child(0)
        block2 = sdfg.root.child(1)

        assert isinstance(block1, Block)
        assert isinstance(block2, Block)
        assert block1.element_id != block2.element_id

    def test_empty_block(self):
        """Test a block with no nodes."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()

        sdfg = builder.move()
        block = sdfg.root.child(0)

        dataflow = block.dataflow
        nodes = list(dataflow.nodes)
        edges = list(dataflow.edges)

        assert len(nodes) == 0
        assert len(edges) == 0
