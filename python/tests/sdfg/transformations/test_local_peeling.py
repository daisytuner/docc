from docc.sdfg import (
    AnalysisManager,
    LoopPeeling,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
)


def _build_elementwise_map(name="cuda_copy"):
    """Single parallel map ``B[i] = A[i]`` over contiguous pointer arguments."""
    builder = StructuredSDFGBuilder(name)
    f = Scalar(PrimitiveType.Float)
    u = Scalar(PrimitiveType.UInt64)

    builder.add_container("N", u, is_argument=True)
    builder.add_container("A", Pointer(f), is_argument=True)
    builder.add_container("B", Pointer(f), is_argument=True)
    builder.add_container("i", u)

    m = builder.begin_map("i", "0", "N", "1")
    builder.add_assignment("B(i)", "A(i)")
    builder.end_map()
    return builder, m


def test_loop_peeling_binding_smoke():
    builder, m = _build_elementwise_map("peel_copy")
    am = AnalysisManager(builder)

    t = LoopPeeling(m)
    assert "LoopPeeling" in repr(t)
    # A simple (non-compound-condition) loop is not peelable; the point here is
    # that the binding evaluates applicability without throwing.
    assert isinstance(t.can_be_applied(builder, am), bool)
