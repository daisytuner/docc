import pytest

from docc.sdfg import (
    InLocalStorage,
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


def test_in_local_storage_rejects_unknown_storage():
    builder, m = _build_elementwise_map("ils_reject")
    block = builder.add_block()
    access = builder.add_access(block, "A")

    with pytest.raises((ValueError, RuntimeError)):
        InLocalStorage(m, access, "BOGUS")


def test_in_local_storage_accepts_known_storage():
    builder, m = _build_elementwise_map("ils_accept")
    block = builder.add_block()
    access = builder.add_access(block, "A")

    for storage_type in ("CPU_Stack", "NV_Shared"):
        t = InLocalStorage(m, access, storage_type)
        assert "InLocalStorage" in repr(t)

    # Default storage type (CPU_Stack) also constructs.
    assert "InLocalStorage" in repr(InLocalStorage(m, access))
