from docc.sdfg import (
    AnalysisManager,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
    VectorizeTransform,
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


def test_vectorize_transform():
    builder, m = _build_elementwise_map("vec_copy")
    am = AnalysisManager(builder)

    t = VectorizeTransform(m)
    assert "VectorizeTransform" in repr(t)
    assert t.can_be_applied(builder, am)
    t.apply(builder, am)

    assert m.schedule_type.value != "SEQUENTIAL"
