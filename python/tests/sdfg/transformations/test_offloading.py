from docc.sdfg import (
    AnalysisManager,
    CUDAParallelizeNestedMap,
    CUDATransform,
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


def _build_nested_maps(name="cuda_nest"):
    """Two nested parallel maps ``B[i*M + j] = A[i*M + j]``."""
    builder = StructuredSDFGBuilder(name)
    f = Scalar(PrimitiveType.Float)
    u = Scalar(PrimitiveType.UInt64)

    builder.add_container("N", u, is_argument=True)
    builder.add_container("M", u, is_argument=True)
    builder.add_container("A", Pointer(f), is_argument=True)
    builder.add_container("B", Pointer(f), is_argument=True)
    builder.add_container("i", u)
    builder.add_container("j", u)

    outer = builder.begin_map("i", "0", "N", "1")
    inner = builder.begin_map("j", "0", "M", "1")
    builder.add_assignment("B(i*M + j)", "A(i*M + j)")
    builder.end_map()
    builder.end_map()
    return builder, outer, inner


# ---------------------------------------------------------------------------
# CUDATransform
# ---------------------------------------------------------------------------


def test_cuda_transform_offloads_map():
    builder, m = _build_elementwise_map()
    am = AnalysisManager(builder)

    assert m.schedule_type.value == "SEQUENTIAL"

    t = CUDATransform(m, block_size=32)
    assert "CUDATransform" in repr(t)
    assert t.can_be_applied(builder, am)
    t.apply(builder, am)

    assert m.schedule_type.value == "CUDA"
    # CUDATransform assigns the X grid dimension.
    assert m.schedule_type.properties.get("dimension") == "0"


def test_cuda_transform_default_block_size():
    builder, m = _build_elementwise_map("cuda_copy_default")
    am = AnalysisManager(builder)

    t = CUDATransform(m)  # default block_size = 32
    assert t.can_be_applied(builder, am)
    t.apply(builder, am)
    assert m.schedule_type.value == "CUDA"


# ---------------------------------------------------------------------------
# CUDAParallelizeNestedMap
# ---------------------------------------------------------------------------


def test_cuda_parallelize_nested_map():
    builder, outer, inner = _build_nested_maps()

    # Offload the outer map to X, then parallelize the nested map to Y.
    am = AnalysisManager(builder)
    outer_t = CUDATransform(outer, block_size=32)
    assert outer_t.can_be_applied(builder, am)
    outer_t.apply(builder, am)
    assert outer.schedule_type.value == "CUDA"

    am = AnalysisManager(builder)
    nested_t = CUDAParallelizeNestedMap(inner, block_size=1)
    assert "CUDAParallelizeNestedMap" in repr(nested_t)
    assert nested_t.can_be_applied(builder, am)
    nested_t.apply(builder, am)

    assert inner.schedule_type.value == "CUDA"
    # Parent is X (0), so the child takes the Y grid dimension (1).
    assert inner.schedule_type.properties.get("dimension") == "1"
