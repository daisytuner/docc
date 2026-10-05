"""CUDA execution test for the ``SoftwarePipelining`` transformation.

Builds an offloaded cooperative reduction ``out[i] += A[k]`` (``i`` an
``X_BLOCK`` thread dim, ``A[k]`` cooperative across the block), localizes ``A``
into shared memory with ``LocalStorage`` (+ ``SyncConditionPropagation`` for the
staging barrier guard), then software-pipelines the panel loop.

``SoftwarePipelining`` double-buffers the shared tile and prefetches the next
panel via ``cp.async`` (``__pipeline_memcpy_async`` / ``__pipeline_commit`` /
``__pipeline_wait_prior``), overlapping the next panel's global load with the
current panel's compute.  The generated kernel must emit those primitives and
still produce the correct row-reduction on the GPU.
"""

from pathlib import Path

import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    BufferLifecycle,
    DataTransferDirection,
    LocalStorage,
    LoopTiling,
    Pointer,
    PrimitiveType,
    Scalar,
    ScheduleType,
    SoftwarePipelining,
    StorageType,
    StructuredSDFGBuilder,
    SyncConditionPropagation,
    TargetLevel,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

pytestmark = pytest.mark.cuda()

FLOAT_BYTES = 4


def _build(N, K, block):
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("sp_coop_cuda")
    b.add_container("A", host, is_argument=True)
    b.add_container("out", host, is_argument=True)
    b.add_container("__daisy_dev_A", dev, is_argument=False)
    b.add_container("__daisy_dev_out", dev, is_argument=False)
    b.add_container("i", i32)
    b.add_container("k", i32)

    def off(hc, dc, dirn, life, size):
        b.add_cuda_offloading_block(hc, dc, dirn, life, dev, size)

    off(
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{K} * {FLOAT_BYTES}",
    )
    off(
        "__daisy_dev_out",
        "__daisy_dev_out",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{N} * {FLOAT_BYTES}",
    )
    off(
        "A",
        "__daisy_dev_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        f"{K} * {FLOAT_BYTES}",
    )
    off(
        "out",
        "__daisy_dev_out",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        f"{N} * {FLOAT_BYTES}",
    )

    b.begin_map(
        "i", "0", str(N), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, block)
    )
    inner = b.begin_for("k", "0", str(K), "1")
    blk = b.add_block()
    o_in = b.add_access(blk, "__daisy_dev_out")
    a = b.add_access(blk, "__daisy_dev_A")
    o_out = b.add_access(blk, "__daisy_dev_out")
    t = b.add_tasklet(blk, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    b.add_memlet(blk, o_in, "", t, "_in1", "i", dev)
    b.add_memlet(blk, a, "", t, "_in2", "k", dev)
    b.add_memlet(blk, t, "_out", o_out, "", "i", dev)
    b.end_for()
    b.end_map()

    off(
        "out",
        "__daisy_dev_out",
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
        "__daisy_dev_out",
        "__daisy_dev_out",
        DataTransferDirection.NONE,
        BufferLifecycle.FREE,
        "0",
    )

    return b, inner, a


@pytest.mark.parametrize(
    "N,K,block,tile",
    [(64, 24, 32, 8), (128, 32, 32, 8), (64, 12, 16, 4), (128, 64, 32, 8)],
    ids=["64x24x32", "128x32x32", "64x12x16", "128x64x32"],
)
@pytest.mark.parametrize("stages", [2, 3, 4], ids=["stages2", "stages3", "stages4"])
def test_software_pipelining_cooperative(N, K, block, tile, stages, tmp_path):
    builder, inner, a = _build(N, K, block)
    am = AnalysisManager(builder)

    # Tile the reduction loop so each outer (panel) iteration stages a *new*
    # A[kt:kt+tile] shared tile — the structure software pipelining overlaps.
    tiling = LoopTiling(inner, tile)
    assert tiling.can_be_applied(builder, am)
    tiling.apply(builder, am)
    panel = tiling.outer_loop

    # Cooperatively stage each panel's A tile into shared memory.
    ls = LocalStorage(tiling.inner_loop, a)
    assert ls.can_be_applied(builder, am), "cooperative shared-memory tile should apply"
    ls.apply(builder, am)
    SyncConditionPropagation().run(builder, am)

    # Software-pipeline the (now shared-staging) panel loop.
    sp = SoftwarePipelining(panel, stages=stages)
    if K // tile < stages:
        assert not sp.can_be_applied(builder, am), "fewer panels than stages"
        return
    assert sp.can_be_applied(
        builder, am
    ), "shared-staging GPU panel loop should be pipelineable"
    sp.apply(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"sp_{N}x{K}x{block}_s{stages}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    assert "__shared__" in generated, "shared buffer not emitted"
    assert "__pipeline_memcpy_async" in generated, "cp.async prefetch not emitted"
    assert "__pipeline_commit" in generated, "pipeline commit not emitted"
    assert "__pipeline_wait_prior" in generated, "pipeline wait not emitted"
    assert (
        f"__pipeline_wait_prior({stages - 1})" in generated
    ), "steady-state wait must keep stages-1 groups in flight"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((K,)).astype(np.float32)
    out = np.zeros((N,), dtype=np.float32)

    compiled(A, out)

    np.testing.assert_allclose(out, np.full(N, A.sum()), rtol=1e-4, atol=1e-4)
