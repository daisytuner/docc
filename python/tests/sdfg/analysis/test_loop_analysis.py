"""Tests for LoopAnalysis bindings."""

import pytest
from docc.sdfg import (
    StructuredSDFGBuilder,
    AnalysisManager,
    LoopInfo,
    For,
    StructuredLoop,
    ControlFlowNode,
)


class TestLoopAnalysisBasic:
    """Basic tests for LoopAnalysis functionality."""

    def test_loops_empty_sdfg(self):
        """Test loops() returns empty list for SDFG with no loops."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.add_block()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        assert isinstance(loops, list)
        assert len(loops) == 0

    def test_loops_single_for(self):
        """Test loops() returns single loop for SDFG with one for loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        assert len(loops) == 1
        assert isinstance(loops[0], For)

    def test_loops_nested_fors(self):
        """Test loops() returns all nested loops in DFS order."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        assert len(loops) == 2
        # DFS order: outer loop first, then inner loop
        assert "i" in loops[0].indvar
        assert "j" in loops[1].indvar


class TestLoopInfo:
    """Tests for LoopInfo struct."""

    def test_loop_info_single_loop(self):
        """Test loop_info() returns correct info for a single loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        info = loop_analysis.loop_info(loops[0])

        assert isinstance(info, LoopInfo)
        assert info.num_loops >= 1
        assert info.num_fors >= 1
        assert info.num_maps == 0
        assert info.num_whiles == 0
        assert info.max_depth >= 1

    def test_loop_info_nested_loops(self):
        """Test loop_info() returns correct depth for nested loops."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.begin_for("k", "0", "30", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        # The outermost loop should have max_depth covering all nested loops
        outer_info = loop_analysis.loop_info(loops[0])
        assert outer_info.max_depth >= 3

    def test_loop_info_repr(self):
        """Test LoopInfo string representation."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        info = loop_analysis.loop_info(loops[0])

        repr_str = repr(info)
        assert "LoopInfo" in repr_str
        assert "element_id=" in repr_str
        assert "num_loops=" in repr_str
        assert "max_depth=" in repr_str

    def test_loop_info_all_properties(self):
        """Test all LoopInfo properties are accessible."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        info = loop_analysis.loop_info(loops[0])

        # Check all properties are accessible and have expected types
        assert isinstance(info.loopnest_index, int)
        assert isinstance(info.element_id, int)
        assert isinstance(info.num_loops, int)
        assert isinstance(info.num_maps, int)
        assert isinstance(info.num_fors, int)
        assert isinstance(info.num_whiles, int)
        assert isinstance(info.max_depth, int)
        assert isinstance(info.is_perfectly_nested, bool)
        assert isinstance(info.is_perfectly_parallel, bool)
        assert isinstance(info.is_elementwise, bool)
        assert isinstance(info.has_side_effects, bool)


class TestFindLoopByIndvar:
    """Tests for find_loop_by_indvar()."""

    def test_find_existing_indvar(self):
        """Test finding a loop by existing induction variable."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop = loop_analysis.find_loop_by_indvar("i")
        assert loop is not None
        assert isinstance(loop, For)
        assert "i" in loop.indvar

    def test_find_nonexistent_indvar(self):
        """Test finding a loop by nonexistent induction variable returns None."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop = loop_analysis.find_loop_by_indvar("nonexistent")
        assert loop is None

    def test_find_indvar_nested_loops(self):
        """Test finding loops by indvar in nested loop structure."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop_i = loop_analysis.find_loop_by_indvar("i")
        loop_j = loop_analysis.find_loop_by_indvar("j")

        assert loop_i is not None
        assert loop_j is not None
        assert loop_i is not loop_j
        assert "i" in loop_i.indvar
        assert "j" in loop_j.indvar


class TestParentLoop:
    """Tests for parent_loop()."""

    def test_parent_loop_outermost(self):
        """Test parent_loop() returns None for outermost loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        parent = loop_analysis.parent_loop(loops[0])
        assert parent is None

    def test_parent_loop_nested(self):
        """Test parent_loop() returns correct parent for nested loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop_i = loop_analysis.find_loop_by_indvar("i")
        loop_j = loop_analysis.find_loop_by_indvar("j")

        parent_of_j = loop_analysis.parent_loop(loop_j)
        assert parent_of_j is not None
        assert parent_of_j is loop_i


class TestOutermostLoops:
    """Tests for outermost_loops() and is_outermost_loop()."""

    def test_outermost_loops_single(self):
        """Test outermost_loops() with single outermost loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        outermost = loop_analysis.outermost_loops()
        assert len(outermost) == 1
        assert "i" in outermost[0].indvar

    def test_outermost_loops_multiple(self):
        """Test outermost_loops() with multiple sequential loops."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        outermost = loop_analysis.outermost_loops()
        assert len(outermost) == 2

    def test_outermost_loops_nested_only_returns_outer(self):
        """Test outermost_loops() only returns outer loop for nested structure."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        outermost = loop_analysis.outermost_loops()
        assert len(outermost) == 1
        assert "i" in outermost[0].indvar

    def test_is_outermost_loop(self):
        """Test is_outermost_loop() correctly identifies outermost loops."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop_i = loop_analysis.find_loop_by_indvar("i")
        loop_j = loop_analysis.find_loop_by_indvar("j")

        assert loop_analysis.is_outermost_loop(loop_i) is True
        assert loop_analysis.is_outermost_loop(loop_j) is False


class TestChildrenAndDescendants:
    """Tests for children() and descendants()."""

    def test_children_empty(self):
        """Test children() returns empty list for innermost loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        children = loop_analysis.children(loops[0])
        assert isinstance(children, list)
        assert len(children) == 0

    def test_children_nested(self):
        """Test children() returns immediate child loops."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop_i = loop_analysis.find_loop_by_indvar("i")
        loop_j = loop_analysis.find_loop_by_indvar("j")

        children_i = loop_analysis.children(loop_i)
        assert len(children_i) == 1
        assert children_i[0] is loop_j

        children_j = loop_analysis.children(loop_j)
        assert len(children_j) == 0

    def test_descendants_empty(self):
        """Test descendants() returns empty for innermost loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        descendants = loop_analysis.descendants(loops[0])
        assert isinstance(descendants, list)
        assert len(descendants) == 0

    def test_descendants_nested(self):
        """Test descendants() returns all descendant loops."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.begin_for("k", "0", "30", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop_i = loop_analysis.find_loop_by_indvar("i")
        loop_j = loop_analysis.find_loop_by_indvar("j")
        loop_k = loop_analysis.find_loop_by_indvar("k")

        # Descendants of i should include j and k
        descendants_i = loop_analysis.descendants(loop_i)
        assert len(descendants_i) == 2
        assert loop_j in descendants_i
        assert loop_k in descendants_i

        # Descendants of j should include only k
        descendants_j = loop_analysis.descendants(loop_j)
        assert len(descendants_j) == 1
        assert loop_k in descendants_j


class TestLoopTreePaths:
    """Tests for loop_tree_paths()."""

    def test_loop_tree_paths_single_loop(self):
        """Test loop_tree_paths() for single loop returns path with just that loop."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loops = loop_analysis.loops()
        paths = loop_analysis.loop_tree_paths(loops[0])

        assert isinstance(paths, list)
        assert len(paths) >= 1
        # Each path should be a list of ControlFlowNode
        for path in paths:
            assert isinstance(path, list)

    def test_loop_tree_paths_nested(self):
        """Test loop_tree_paths() returns correct paths for nested loops."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.begin_for("j", "0", "20", "1")
        builder.add_block()
        builder.end_for()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        loop_i = loop_analysis.find_loop_by_indvar("i")
        paths = loop_analysis.loop_tree_paths(loop_i)

        # Should have at least one path from i to the leaf
        assert len(paths) >= 1


class TestLoopAnalysisRepr:
    """Tests for LoopAnalysis string representation."""

    def test_loop_analysis_repr(self):
        """Test LoopAnalysis __repr__."""
        builder = StructuredSDFGBuilder("test_sdfg")
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        assert str(loop_analysis) == "<LoopAnalysis>"


class TestOutermostMaps:
    """Tests for outermost_maps() - testing with For loops since Maps require special construction."""

    def test_outermost_maps_empty_when_no_maps(self):
        """Test outermost_maps() returns empty when only For loops present."""
        builder = StructuredSDFGBuilder("test_sdfg")
        builder.begin_for("i", "0", "10", "1")
        builder.add_block()
        builder.end_for()
        sdfg = builder.move()

        analysis = AnalysisManager(sdfg)
        loop_analysis = analysis.loop_analysis()

        maps = loop_analysis.outermost_maps()
        assert isinstance(maps, list)
        # No maps in this SDFG, only For loops
        assert len(maps) == 0
