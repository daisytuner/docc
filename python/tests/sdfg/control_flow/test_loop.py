"""Tests for loop bindings (For, Map, While, StructuredLoop)."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    Scalar,
    PrimitiveType,
    DebugInfo,
    For,
    While,
    Sequence,
    StructuredLoop,
    ScheduleTypeCategory,
)


class TestStructuredLoop:
    """Test suite for StructuredLoop base class properties."""

    def test_indvar_property(self):
        """Test indvar property returns the induction variable."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        indvar = for_node.indvar
        assert isinstance(indvar, str)
        assert "i" in indvar

    def test_init_property(self):
        """Test init property returns the initialization expression."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        init = for_node.init
        assert isinstance(init, str)
        assert "0" in init

    def test_update_property(self):
        """Test update property returns the update expression."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        update = for_node.update
        assert isinstance(update, str)
        # Update should contain i + 1
        assert "i" in update

    def test_condition_property(self):
        """Test condition property returns the loop condition."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        condition = for_node.condition
        assert isinstance(condition, str)
        # Condition should contain i < 10
        assert "i" in condition
        assert "10" in condition

    def test_body_property(self):
        """Test body property returns the loop body sequence."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        body = for_node.body
        assert isinstance(body, Sequence)
        assert body.size == 1

    def test_loop_parent(self):
        """Test parent references of structured loop"""
        builder = StructuredSDFGBuilder("test_sdfg")

        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        assert for_node.parent is sdfg.root
        assert for_node.body.parent is for_node
        assert for_node.body.child(0).parent is for_node.body


class TestFor:
    """Test suite for For loop properties."""

    def test_for_isinstance(self):
        """Test that for loop is an instance of For."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        assert isinstance(for_node, For)

    def test_for_inherits_structured_loop(self):
        """Test that For inherits from StructuredLoop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        assert isinstance(for_node, StructuredLoop)

    def test_for_repr(self):
        """Test For string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        repr_str = repr(for_node)
        assert "For" in repr_str
        assert "indvar=" in repr_str

    def test_for_with_step(self):
        """Test for loop with different step values."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "100", "5")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        update = for_node.update
        # Update should contain i + 5
        assert "5" in update

    def test_for_with_symbolic_bounds(self):
        """Test for loop with symbolic bounds."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("n", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_for("i", "0", "n", "1")
        builder.add_block()
        builder.end_for()

        sdfg = builder.move()
        for_node = sdfg.root.child(0)

        condition = for_node.condition
        assert "n" in condition

    def test_nested_for_loops(self):
        """Test nested for loops."""
        builder = StructuredSDFGBuilder("test_sdfg")

        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()

        sdfg = builder.move()
        outer_for = sdfg.root.child(0)

        assert isinstance(outer_for, For)
        assert outer_for.indvar == "i"

        # Get nested for loop
        inner_for = outer_for.body.child(0)
        assert isinstance(inner_for, For)
        assert inner_for.indvar == "j"


class TestWhile:
    """Test suite for While loop properties."""

    def test_while_isinstance(self):
        """Test that while loop is an instance of While."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_while()
        builder.begin_if("x > 0")
        builder.add_block()
        builder.begin_else()
        builder.add_break()
        builder.end_if()
        builder.end_while()

        sdfg = builder.move()
        while_node = sdfg.root.child(0)

        assert isinstance(while_node, While)

    def test_while_body_property(self):
        """Test body property returns the loop body sequence."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_while()
        builder.begin_if("x > 0")
        builder.add_block()
        builder.begin_else()
        builder.add_break()
        builder.end_if()
        builder.end_while()

        sdfg = builder.move()
        while_node = sdfg.root.child(0)

        body = while_node.body
        assert isinstance(body, Sequence)

    def test_while_repr(self):
        """Test While string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_container("x", Scalar(PrimitiveType.Int32), is_argument=True)

        builder.begin_while()
        builder.add_break()
        builder.end_while()

        sdfg = builder.move()
        while_node = sdfg.root.child(0)

        repr_str = repr(while_node)
        assert "While" in repr_str


class TestBreakContinue:
    """Test suite for Break and Continue properties."""

    def test_break_in_while(self):
        """Test break statement in while loop."""
        builder = StructuredSDFGBuilder("test_sdfg")

        builder.begin_while()
        builder.add_break()
        builder.end_while()

        sdfg = builder.move()
        while_node = sdfg.root.child(0)
        break_node = while_node.body.child(0)

        from docc.sdfg import Break

        assert isinstance(break_node, Break)

    def test_continue_in_while(self):
        """Test continue statement in while loop."""
        builder = StructuredSDFGBuilder("test_sdfg")

        builder.begin_while()
        builder.add_continue()
        builder.add_break()  # Need break to avoid infinite loop
        builder.end_while()

        sdfg = builder.move()
        while_node = sdfg.root.child(0)
        continue_node = while_node.body.child(0)

        from docc.sdfg import Continue

        assert isinstance(continue_node, Continue)

    def test_break_repr(self):
        """Test Break string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")

        builder.begin_while()
        builder.add_break()
        builder.end_while()

        sdfg = builder.move()
        while_node = sdfg.root.child(0)
        break_node = while_node.body.child(0)

        repr_str = repr(break_node)
        assert "Break" in repr_str

    def test_continue_repr(self):
        """Test Continue string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")

        builder.begin_while()
        builder.add_continue()
        builder.add_break()
        builder.end_while()

        sdfg = builder.move()
        while_node = sdfg.root.child(0)
        continue_node = while_node.body.child(0)

        repr_str = repr(continue_node)
        assert "Continue" in repr_str


class TestScheduleType:
    """Test suite for ScheduleType and ScheduleTypeCategory."""

    def test_schedule_type_category_values(self):
        """Test ScheduleTypeCategory enum values are accessible."""
        assert ScheduleTypeCategory.Offloader is not None
        assert ScheduleTypeCategory.Parallelizer is not None
        assert ScheduleTypeCategory.Vectorizer is not None
        assert ScheduleTypeCategory.None_ is not None

    def test_schedule_type_category_comparison(self):
        """Test ScheduleTypeCategory comparison."""
        assert ScheduleTypeCategory.Offloader == ScheduleTypeCategory.Offloader
        assert ScheduleTypeCategory.Offloader != ScheduleTypeCategory.Parallelizer
