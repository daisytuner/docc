"""CUDA execution test for the ``LoopPeeling`` transformation.

Builds an already-offloaded per-thread reduction ``C[i] = sum_k A[i, k]`` (``i``
an ``X_BLOCK`` thread dim), tiles the ``k`` reduction loop so the inner loop gets
a compound condition ``k < K && k < k_tile + TILE``, then applies
``LoopPeeling`` to over-approximate the inner loop to its constant-trip
bound and guard the body with ``k < K``.

The transformed inner loop has a compile-time-constant trip count, so the nest
unrolls while the ``k < K`` predicate preserves correctness for ragged ``K`` (not
a multiple of the tile). Sizes cover exact-fit and ragged ``N`` / ``K``.
"""

import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    BufferLifecycle,
    DataTransferDirection,
    LoopTiling,
    Pointer,
    LoopPeeling,
    PrimitiveType,
    Scalar,
    ScheduleType,
    StorageType,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

pytestmark = pytest.mark.cuda()

FLOAT_BYTES = 4


def _build(N, K, block):
    """Offloaded kernel ``C[i] = sum_k A[i*K + k]`` (per-thread reduction over k)."""
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("predicate_boundary_cuda")
    b.add_container("A", host, is_argument=True)
    b.add_container("C", host, is_argument=True)
    b.add_container("__daisy_dev_A", dev, is_argument=False)
    b.add_container("__daisy_dev_C", dev, is_argument=False)
    b.add_container("i", i32)
    b.add_container("k", i32)

    def off(hc, dc, dirn, life, size):
        b.add_cuda_offloading_block(hc, dc, dirn, life, dev, size)

    off(
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{N * K} * {FLOAT_BYTES}",
    )
    off(
        "__daisy_dev_C",
        "__daisy_dev_C",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{N} * {FLOAT_BYTES}",
    )
    off(
        "A",
        "__daisy_dev_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        f"{N * K} * {FLOAT_BYTES}",
    )
    off(
        "C",
        "__daisy_dev_C",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        f"{N} * {FLOAT_BYTES}",
    )

    b.begin_map(
        "i", "0", str(N), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, block)
    )
    inner = b.begin_for("k", "0", str(K), "1")
    blk = b.add_block()
    c_in = b.add_access(blk, "__daisy_dev_C")
    a = b.add_access(blk, "__daisy_dev_A")
    c_out = b.add_access(blk, "__daisy_dev_C")
    t = b.add_tasklet(blk, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    b.add_memlet(blk, c_in, "", t, "_in1", "i", dev)
    b.add_memlet(blk, a, "", t, "_in2", f"i*{K} + k", dev)
    b.add_memlet(blk, t, "_out", c_out, "", "i", dev)
    b.end_for()
    b.end_map()

    off(
        "C",
        "__daisy_dev_C",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        f"{N} * {FLOAT_BYTES}",
    )
    off(
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.FREE,
        "0",
    )
    off(
        "__daisy_dev_C",
        "__daisy_dev_C",
        DataTransferDirection.NONE,
        BufferLifecycle.FREE,
        "0",
    )

    return b, inner


@pytest.mark.parametrize(
    "N,K,block,tile",
    [
        (64, 16, 32, 8),  # exact fit
        (64, 20, 32, 8),  # ragged K (20 % 8 != 0)
        (100, 20, 32, 8),  # ragged N and K
        (33, 10, 16, 8),  # ragged N and K, small block
    ],
    ids=["64x16", "64x20", "100x20", "33x10"],
)
@pytest.mark.parametrize("predicate", [False, True], ids=["hoist", "predicate"])
def test_loop_peeling_ragged_reduction(N, K, block, tile, predicate, tmp_path):
    builder, inner = _build(N, K, block)
    am = AnalysisManager(builder)

    # Tiling the reduction loop yields the compound inner condition
    # `k < K && k < k_tile + tile` that LoopPeeling targets.
    tiling = LoopTiling(inner, tile)
    assert tiling.can_be_applied(builder, am)
    tiling.apply(builder, am)
    tiled_inner = tiling.inner_loop

    pb = LoopPeeling(tiled_inner, predicate=predicate)
    assert pb.can_be_applied(
        builder, am
    ), "tiled inner loop should have a predicable compound boundary"
    pb.apply(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"pb_{N}x{K}x{block}x{tile}_{int(predicate)}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    # Both modes emit a boundary check (hoisted then/else guard, or inner predicate).
    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    assert "if" in generated, "boundary check not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, K)).astype(np.float32)
    C = np.zeros((N,), dtype=np.float32)

    compiled(A.reshape(-1), C)

    np.testing.assert_allclose(C, A.sum(axis=1), rtol=1e-4, atol=1e-4)
