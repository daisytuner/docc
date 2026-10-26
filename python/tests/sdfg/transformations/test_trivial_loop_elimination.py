import json

from docc.sdfg import (
    AnalysisManager,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
    TrivialLoopElimination,
)


def _count_maps(node):
    if isinstance(node, dict):
        return int(node.get("type") == "map") + sum(
            _count_maps(v) for v in node.values()
        )
    if isinstance(node, list):
        return sum(_count_maps(v) for v in node)
    return 0


def _build_map(name, bound):
    """Single sequential map ``B[i] = A[i]`` for ``i in [0, bound)``."""
    builder = StructuredSDFGBuilder(name)
    f = Scalar(PrimitiveType.Float)
    u = Scalar(PrimitiveType.Int64)

    builder.add_container("A", Pointer(f), is_argument=True)
    builder.add_container("B", Pointer(f), is_argument=True)
    builder.add_container("i", u)

    m = builder.begin_map("i", "0", bound, "1")
    builder.add_assignment("B(i)", "A(i)")
    builder.end_map()
    return builder, m


def test_trivial_loop_elimination_applies():
    builder, m = _build_map("trivial_map", "1")
    am = AnalysisManager(builder)

    t = TrivialLoopElimination(m)
    assert "TrivialLoopElimination" in repr(t)
    assert t.name == "TrivialLoopElimination"
    assert t.try_apply(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    assert _count_maps(json.loads(sdfg.to_json())) == 0


def test_trivial_loop_elimination_not_applicable():
    builder, m = _build_map("non_trivial_map", "8")
    am = AnalysisManager(builder)

    t = TrivialLoopElimination(m)
    assert not t.can_be_applied(builder, am)
    assert not t.try_apply(builder, am)
    assert _count_maps(json.loads(builder.move().to_json())) == 1
