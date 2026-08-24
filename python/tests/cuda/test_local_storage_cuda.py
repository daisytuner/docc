"""CUDA execution test for the cooperative (shared-memory) ``LocalStorage`` path.

Builds an already-offloaded kernel ``out[i] += A[k]`` (``i`` is an ``X_BLOCK``
thread dim, ``A[k]`` a read tile independent of ``i`` — i.e. cooperative across
the block), localizes ``A`` into shared memory with ``LocalStorage``, finalizes
the staging barrier's divergence guard with ``SyncConditionPropagation``,
compiles for CUDA and checks the result against NumPy on the GPU.

The generated kernel must cooperatively load ``A[0:K]`` into ``__shared__``,
``__syncthreads()`` (unconditionally, so ragged block sizes don't deadlock),
then have every thread read the shared tile.  Sizes cover exact-fit and ragged
(N not a multiple of the block) cases.
"""

from pathlib import Path

import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    BufferLifecycle,
    DataTransferDirection,
    LocalStorage,
    Pointer,
    PrimitiveType,
    Scalar,
    ScheduleType,
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

    b = StructuredSDFGBuilder("ls_coop_cuda")
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
    "N,K,block",
    [(64, 16, 32), (100, 16, 32), (48, 10, 32), (33, 8, 16)],
    ids=["64x16x32", "100x16x32", "48x10x32", "33x8x16"],
)
def test_local_storage_cooperative_shared(N, K, block, tmp_path):
    builder, inner, a = _build(N, K, block)

    am = AnalysisManager(builder)
    xform = LocalStorage(inner, a)
    assert xform.can_be_applied(
        builder, am
    ), "cooperative shared-memory tile should apply"
    xform.apply(builder, am)
    # Hoist the boundary guard off the staging barrier so ragged blocks don't deadlock.
    SyncConditionPropagation().run(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"coop_{N}x{K}x{block}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    # The shared tile must be cooperatively staged via the offload coverage loop.
    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    assert "__shared__" in generated, "shared buffer not emitted"
    assert "__syncthreads" in generated, "staging barrier not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((K,)).astype(np.float32)
    out = np.zeros((N,), dtype=np.float32)

    compiled(A, out)

    np.testing.assert_allclose(out, np.full(N, A.sum()), rtol=1e-4, atol=1e-4)


def _build_mixed(N, M, K, bm, bn):
    """``C[i,j] += A[i,k]`` over k — A is per-thread in i, cooperative in j.

    Localizing A gives a shared A_sh[BM][K]: each i-thread owns a row slot, the
    j-threads cooperatively load it. Result: C[i,j] = sum_k A[i,k] (row sum,
    broadcast across columns).
    """
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("ls_coop_mixed_cuda")
    b.add_container("A", host, is_argument=True)
    b.add_container("C", host, is_argument=True)
    b.add_container("__daisy_dev_A", dev, is_argument=False)
    b.add_container("__daisy_dev_C", dev, is_argument=False)
    b.add_container("i", i32)
    b.add_container("j", i32)
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
        f"{N * M} * {FLOAT_BYTES}",
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
        f"{N * M} * {FLOAT_BYTES}",
    )

    b.begin_map(
        "i", "0", str(N), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, bm)
    )
    b.begin_map(
        "j", "0", str(M), "1", ScheduleType.cuda_offload(TargetLevel.Y_BLOCK, bn)
    )
    inner = b.begin_for("k", "0", str(K), "1")
    blk = b.add_block()
    c_in = b.add_access(blk, "__daisy_dev_C")
    a = b.add_access(blk, "__daisy_dev_A")
    c_out = b.add_access(blk, "__daisy_dev_C")
    t = b.add_tasklet(blk, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    b.add_memlet(blk, c_in, "", t, "_in1", f"i*{M} + j", dev)
    b.add_memlet(blk, a, "", t, "_in2", f"i*{K} + k", dev)
    b.add_memlet(blk, t, "_out", c_out, "", f"i*{M} + j", dev)
    b.end_for()
    b.end_map()
    b.end_map()

    off(
        "C",
        "__daisy_dev_C",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        f"{N * M} * {FLOAT_BYTES}",
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

    return b, inner, a


@pytest.mark.parametrize(
    "N,M,K,bm,bn",
    [(8, 4, 16, 8, 4), (16, 8, 16, 8, 4), (10, 6, 12, 8, 4), (33, 5, 8, 16, 4)],
    ids=["8x4x16", "16x8x16", "10x6x12", "33x5x8"],
)
def test_local_storage_cooperative_mixed(N, M, K, bm, bn, tmp_path):
    builder, inner, a = _build_mixed(N, M, K, bm, bn)

    am = AnalysisManager(builder)
    xform = LocalStorage(inner, a)
    assert xform.can_be_applied(
        builder, am
    ), "mixed per-thread + cooperative tile should apply"
    xform.apply(builder, am)
    SyncConditionPropagation().run(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"mixed_{N}x{M}x{K}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    assert "__shared__" in generated, "shared buffer not emitted"
    # Mixed staging emits a leading + trailing barrier around the copy.
    assert (
        generated.count("__syncthreads") >= 2
    ), "expected leading + trailing staging barriers"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, K)).astype(np.float32)
    C = np.zeros((N, M), dtype=np.float32)

    compiled(A.reshape(-1), C.reshape(-1))

    # Each row i of C is the row sum of A[i, :], broadcast across the M columns.
    expected = np.broadcast_to(A.sum(axis=1)[:, None], (N, M))
    np.testing.assert_allclose(C, expected, rtol=1e-4, atol=1e-4)
