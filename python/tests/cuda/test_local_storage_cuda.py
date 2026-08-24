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
    CMathFunction,
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


def _build_enclosing_reuse(R, N, block):
    """Two sibling block loops over columns both read row X[row,:].

    Pass 1: Y1[row,j] = X[row,j]. Pass 2: Y2[row,j] = X[row, N-1-j] (reversed).
    Localizing X at the enclosing grid row-loop stages X[row,:] once into a
    per-block shared row that both passes reuse; the reversed read proves the
    whole row is resident and threads read one another's staged elements.
    """
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("ls_enclosing_cuda")
    for nm in ("X", "Y1", "Y2"):
        b.add_container(nm, host, is_argument=True)
    for nm in ("__daisy_dev_X", "__daisy_dev_Y1", "__daisy_dev_Y2"):
        b.add_container(nm, dev, is_argument=False)
    b.add_container("row", i32)
    b.add_container("j1", i32)
    b.add_container("j2", i32)

    def off(hc, dc, dirn, life, size):
        b.add_cuda_offloading_block(hc, dc, dirn, life, dev, size)

    nb = f"{R} * {N} * {FLOAT_BYTES}"
    for nm in ("__daisy_dev_X", "__daisy_dev_Y1", "__daisy_dev_Y2"):
        off(nm, nm, DataTransferDirection.NONE, BufferLifecycle.ALLOC, nb)
    off("X", "__daisy_dev_X", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nb)

    row_map = b.begin_map(
        "row", "0", str(R), "1", ScheduleType.cuda_offload(TargetLevel.X_GRID, 1)
    )
    b.begin_map(
        "j1", "0", str(N), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, block)
    )
    blk1 = b.add_block()
    x1 = b.add_access(blk1, "__daisy_dev_X")
    y1 = b.add_access(blk1, "__daisy_dev_Y1")
    t1 = b.add_tasklet(blk1, TaskletCode.assign, ["_in"], ["_out"])
    b.add_memlet(blk1, x1, "", t1, "_in", f"row*{N} + j1", dev)
    b.add_memlet(blk1, t1, "_out", y1, "", f"row*{N} + j1", dev)
    b.end_map()
    b.begin_map(
        "j2", "0", str(N), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, block)
    )
    blk2 = b.add_block()
    x2 = b.add_access(blk2, "__daisy_dev_X")
    y2 = b.add_access(blk2, "__daisy_dev_Y2")
    t2 = b.add_tasklet(blk2, TaskletCode.assign, ["_in"], ["_out"])
    b.add_memlet(blk2, x2, "", t2, "_in", f"row*{N} + ({N} - 1 - j2)", dev)
    b.add_memlet(blk2, t2, "_out", y2, "", f"row*{N} + j2", dev)
    b.end_map()
    b.end_map()

    for host_nm, dev_nm in (("Y1", "__daisy_dev_Y1"), ("Y2", "__daisy_dev_Y2")):
        off(host_nm, dev_nm, DataTransferDirection.D2H, BufferLifecycle.NO_CHANGE, nb)
    for nm in ("__daisy_dev_X", "__daisy_dev_Y1", "__daisy_dev_Y2"):
        off(nm, nm, DataTransferDirection.NONE, BufferLifecycle.FREE, "0")

    return b, row_map, x1


@pytest.mark.parametrize(
    "R,N,block",
    [(4, 32, 32), (3, 50, 32), (5, 17, 16)],
    ids=["4x32x32", "3x50x32", "5x17x16"],
)
def test_local_storage_enclosing_reuse(R, N, block, tmp_path):
    builder, row_map, x1 = _build_enclosing_reuse(R, N, block)

    am = AnalysisManager(builder)
    xform = LocalStorage(row_map, x1)
    assert xform.can_be_applied(
        builder, am
    ), "enclosing-scope cooperative staging should apply"
    xform.apply(builder, am)
    # Hoist the staging barrier's boundary guard so ragged blocks don't deadlock.
    SyncConditionPropagation().run(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"enc_{R}x{N}x{block}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    assert "__shared__" in generated, "shared row buffer not emitted"
    assert "__syncthreads" in generated, "staging barrier not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((R, N)).astype(np.float32)
    Y1 = np.zeros((R, N), dtype=np.float32)
    Y2 = np.zeros((R, N), dtype=np.float32)

    compiled(X.reshape(-1), Y1.reshape(-1), Y2.reshape(-1))

    np.testing.assert_allclose(Y1, X, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(Y2, X[:, ::-1], rtol=1e-6, atol=1e-6)


def _build_fused_reduce(R, N, block):
    """Fused per-row normalize sharing one staged row across two reductions.

    map row: stage X[row,:] -> buf; reduce-max -> m; barrier; reduce-sum -> s;
    barrier; Y[row,j] = (buf[j] - m[row]) / s[row]. LocalStorage stages X so the
    two reduces and the normalize all read the one shared row (X loaded once).
    Reference: Y = (X - rowmax) / rowsum.
    """
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("ls_fused_cuda")
    for nm in ("X", "Y", "m", "s"):
        b.add_container(nm, host, is_argument=True)
    for nm in ("__daisy_dev_X", "__daisy_dev_Y", "__daisy_dev_m", "__daisy_dev_s"):
        b.add_container(nm, dev, is_argument=False)
    b.add_container("tmp", dev, is_argument=False)  # per-thread scratch
    for nm in ("row", "jr", "js", "jn"):
        b.add_container(nm, i32)

    def off(hc, dc, dirn, life, size):
        b.add_cuda_offloading_block(hc, dc, dirn, life, dev, size)

    nbX = f"{R} * {N} * {FLOAT_BYTES}"
    nbm = f"{R} * {FLOAT_BYTES}"
    for dc, sz in (
        ("__daisy_dev_X", nbX),
        ("__daisy_dev_Y", nbX),
        ("__daisy_dev_m", nbm),
        ("__daisy_dev_s", nbm),
        ("tmp", str(FLOAT_BYTES)),
    ):
        off(dc, dc, DataTransferDirection.NONE, BufferLifecycle.ALLOC, sz)
    off("X", "__daisy_dev_X", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbX)
    off("m", "__daisy_dev_m", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbm)
    off("s", "__daisy_dev_s", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbm)

    row_map = b.begin_map(
        "row", "0", str(R), "1", ScheduleType.cuda_offload(TargetLevel.X_GRID, 1)
    )

    b.begin_reduce(
        "jr",
        "0",
        str(N),
        "1",
        [("max", "__daisy_dev_m")],
        ScheduleType.cuda_offload(TargetLevel.X_BLOCK, block),
    )
    rb = b.add_block()
    mi = b.add_access(rb, "__daisy_dev_m")
    xr = b.add_access(rb, "__daisy_dev_X")
    mo = b.add_access(rb, "__daisy_dev_m")
    mx = b.add_cmath(rb, CMathFunction.fmax, PrimitiveType.Float)
    b.add_memlet(rb, mi, "", mx, "_in1", "row", dev)
    b.add_memlet(rb, xr, "", mx, "_in2", f"row*{N} + jr", dev)
    b.add_memlet(rb, mx, "_out", mo, "", "row", dev)
    b.end_reduce()

    b.add_barrier_local_block()

    b.begin_reduce(
        "js",
        "0",
        str(N),
        "1",
        [("add", "__daisy_dev_s")],
        ScheduleType.cuda_offload(TargetLevel.X_BLOCK, block),
    )
    sb = b.add_block()
    si = b.add_access(sb, "__daisy_dev_s")
    xs = b.add_access(sb, "__daisy_dev_X")
    so = b.add_access(sb, "__daisy_dev_s")
    at = b.add_tasklet(sb, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    b.add_memlet(sb, si, "", at, "_in1", "row", dev)
    b.add_memlet(sb, xs, "", at, "_in2", f"row*{N} + js", dev)
    b.add_memlet(sb, at, "_out", so, "", "row", dev)
    b.end_reduce()

    b.add_barrier_local_block()

    b.begin_map(
        "jn", "0", str(N), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, block)
    )
    nb = b.add_block()
    xn = b.add_access(nb, "__daisy_dev_X")
    mn = b.add_access(nb, "__daisy_dev_m")
    tmp_o = b.add_access(nb, "tmp")
    sub = b.add_tasklet(nb, TaskletCode.fp_sub, ["_in1", "_in2"], ["_out"])
    b.add_memlet(nb, xn, "", sub, "_in1", f"row*{N} + jn", dev)
    b.add_memlet(nb, mn, "", sub, "_in2", "row", dev)
    b.add_memlet(nb, sub, "_out", tmp_o, "", "0", dev)
    tmp_i = b.add_access(nb, "tmp")
    sn = b.add_access(nb, "__daisy_dev_s")
    yn = b.add_access(nb, "__daisy_dev_Y")
    div = b.add_tasklet(nb, TaskletCode.fp_div, ["_in1", "_in2"], ["_out"])
    b.add_memlet(nb, tmp_i, "", div, "_in1", "0", dev)
    b.add_memlet(nb, sn, "", div, "_in2", "row", dev)
    b.add_memlet(nb, div, "_out", yn, "", f"row*{N} + jn", dev)
    b.end_map()

    b.end_map()

    off("Y", "__daisy_dev_Y", DataTransferDirection.D2H, BufferLifecycle.NO_CHANGE, nbX)
    for dc in (
        "__daisy_dev_X",
        "__daisy_dev_Y",
        "__daisy_dev_m",
        "__daisy_dev_s",
        "tmp",
    ):
        off(dc, dc, DataTransferDirection.NONE, BufferLifecycle.FREE, "0")

    return b, row_map, xr


@pytest.mark.parametrize(
    "R,N,block",
    [(4, 32, 32), (3, 50, 32), (5, 17, 16)],
    ids=["4x32x32", "3x50x32", "5x17x16"],
)
def test_local_storage_fused_reduce(R, N, block, tmp_path):
    builder, row_map, xr = _build_fused_reduce(R, N, block)

    am = AnalysisManager(builder)
    xform = LocalStorage(row_map, xr)
    assert xform.can_be_applied(builder, am), "fused reduce staging should apply"
    xform.apply(builder, am)
    SyncConditionPropagation().run(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"fused_{R}x{N}x{block}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    # One staged row buffer, reused by both reduces and the normalize.
    assert "__daisy_local_storage" in generated, "staged row buffer not emitted"
    assert "__syncthreads" in generated, "barriers not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    X = rng.standard_normal((R, N)).astype(np.float32)
    Y = np.zeros((R, N), dtype=np.float32)
    m = np.full(R, -np.inf, dtype=np.float32)
    s = np.zeros(R, dtype=np.float32)

    compiled(X.reshape(-1), Y.reshape(-1), m, s)

    expected = (X - X.max(axis=1, keepdims=True)) / X.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(Y, expected, rtol=1e-4, atol=1e-4)


def _build_gemm_regtile(M, N, K, TY, TX, CY, CX):
    """Thread-coarsened GEMM ``C = A@B`` where each thread owns a CY*CX tile of C.

    Loop nest: jO(X_BLOCK) iO(Y_BLOCK) over the thread-tile grid, then k, then the
    per-thread micro-tile loops iI,jI:
        C[(iO*CY+iI), (jO*CX+jI)] += A[(iO*CY+iI), k] * B[k, (jO*CX+jI)]
    Localizing C at the k loop turns the CY*CX accumulator into a per-thread
    register tile (C_reg), accumulated across k and written back once.
    """
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("ls_gemm_regtile")
    for nm in ("A", "B", "C"):
        b.add_container(nm, host, is_argument=True)
    for nm in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        b.add_container(nm, dev, is_argument=False)
    for nm in ("jO", "iO", "k", "iI", "jI"):
        b.add_container(nm, i32)

    def off(hc, dc, dirn, life, size):
        b.add_cuda_offloading_block(hc, dc, dirn, life, dev, size)

    nbA = f"{M * K} * {FLOAT_BYTES}"
    nbB = f"{K * N} * {FLOAT_BYTES}"
    nbC = f"{M * N} * {FLOAT_BYTES}"
    for dc, sz in (
        ("__daisy_dev_A", nbA),
        ("__daisy_dev_B", nbB),
        ("__daisy_dev_C", nbC),
    ):
        off(dc, dc, DataTransferDirection.NONE, BufferLifecycle.ALLOC, sz)
    off("A", "__daisy_dev_A", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbA)
    off("B", "__daisy_dev_B", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbB)
    off("C", "__daisy_dev_C", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbC)

    b.begin_map(
        "jO", "0", str(N // CX), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, TX)
    )
    b.begin_map(
        "iO", "0", str(M // CY), "1", ScheduleType.cuda_offload(TargetLevel.Y_BLOCK, TY)
    )
    k_loop = b.begin_for("k", "0", str(K), "1")
    b.begin_for("iI", "0", str(CY), "1")
    b.begin_for("jI", "0", str(CX), "1")
    blk = b.add_block()
    a = b.add_access(blk, "__daisy_dev_A")
    bb = b.add_access(blk, "__daisy_dev_B")
    c_in = b.add_access(blk, "__daisy_dev_C")
    c_out = b.add_access(blk, "__daisy_dev_C")
    t = b.add_tasklet(blk, TaskletCode.fp_fma, ["_in1", "_in2", "_in3"], ["_out"])
    b.add_memlet(blk, a, "", t, "_in1", f"(iO*{CY} + iI)*{K} + k", dev)
    b.add_memlet(blk, bb, "", t, "_in2", f"k*{N} + jO*{CX} + jI", dev)
    b.add_memlet(blk, c_in, "", t, "_in3", f"(iO*{CY} + iI)*{N} + jO*{CX} + jI", dev)
    b.add_memlet(blk, t, "_out", c_out, "", f"(iO*{CY} + iI)*{N} + jO*{CX} + jI", dev)
    b.end_for()
    b.end_for()
    b.end_for()
    b.end_map()
    b.end_map()

    off("C", "__daisy_dev_C", DataTransferDirection.D2H, BufferLifecycle.NO_CHANGE, nbC)
    for dc in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        off(dc, dc, DataTransferDirection.NONE, BufferLifecycle.FREE, "0")

    return b, k_loop, c_out


@pytest.mark.parametrize(
    "M,N,K,TY,TX,CY,CX",
    [(8, 8, 4, 2, 2, 2, 2), (16, 16, 8, 4, 4, 2, 2), (8, 8, 3, 2, 2, 2, 2)],
    ids=["8x8x4", "16x16x8", "8x8x3"],
)
def test_local_storage_register_tile_gemm(M, N, K, TY, TX, CY, CX, tmp_path):
    builder, k_loop, c_out = _build_gemm_regtile(M, N, K, TY, TX, CY, CX)

    am = AnalysisManager(builder)
    xform = LocalStorage(k_loop, c_out)
    assert xform.can_be_applied(builder, am), "C register tile should apply"
    xform.apply(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"gemm_{M}x{N}x{K}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    assert "__daisy_local_storage" in generated, "C register tile not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))

    np.testing.assert_allclose(C, A @ B, rtol=1e-3, atol=1e-3)


def _build_gemm_regtile_reduce(M, N, K, TY, TX, CY, CX):
    """As ``_build_gemm_regtile`` but the k loop is a sequential ``Reduce`` node
    (as the frontend emits for ``C[i,j] += ...``). Localizing C at the Reduce
    privatizes the CY*CX accumulator into a per-thread register tile and retargets
    the Reduce descriptor to it — the per-thread (non-cooperative) reduction path.
    """
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("ls_gemm_regtile_reduce")
    for nm in ("A", "B", "C"):
        b.add_container(nm, host, is_argument=True)
    for nm in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        b.add_container(nm, dev, is_argument=False)
    for nm in ("jO", "iO", "k", "iI", "jI"):
        b.add_container(nm, i32)

    def off(hc, dc, dirn, life, size):
        b.add_cuda_offloading_block(hc, dc, dirn, life, dev, size)

    nbA = f"{M * K} * {FLOAT_BYTES}"
    nbB = f"{K * N} * {FLOAT_BYTES}"
    nbC = f"{M * N} * {FLOAT_BYTES}"
    for dc, sz in (
        ("__daisy_dev_A", nbA),
        ("__daisy_dev_B", nbB),
        ("__daisy_dev_C", nbC),
    ):
        off(dc, dc, DataTransferDirection.NONE, BufferLifecycle.ALLOC, sz)
    off("A", "__daisy_dev_A", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbA)
    off("B", "__daisy_dev_B", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbB)
    off("C", "__daisy_dev_C", DataTransferDirection.H2D, BufferLifecycle.NO_CHANGE, nbC)

    b.begin_map(
        "jO", "0", str(N // CX), "1", ScheduleType.cuda_offload(TargetLevel.X_BLOCK, TX)
    )
    b.begin_map(
        "iO", "0", str(M // CY), "1", ScheduleType.cuda_offload(TargetLevel.Y_BLOCK, TY)
    )
    # Sequential per-thread reduction over k into the C accumulator.
    k_loop = b.begin_reduce("k", "0", str(K), "1", [("add", "__daisy_dev_C")])
    b.begin_for("iI", "0", str(CY), "1")
    b.begin_for("jI", "0", str(CX), "1")
    blk = b.add_block()
    a = b.add_access(blk, "__daisy_dev_A")
    bb = b.add_access(blk, "__daisy_dev_B")
    c_in = b.add_access(blk, "__daisy_dev_C")
    c_out = b.add_access(blk, "__daisy_dev_C")
    t = b.add_tasklet(blk, TaskletCode.fp_fma, ["_in1", "_in2", "_in3"], ["_out"])
    b.add_memlet(blk, a, "", t, "_in1", f"(iO*{CY} + iI)*{K} + k", dev)
    b.add_memlet(blk, bb, "", t, "_in2", f"k*{N} + jO*{CX} + jI", dev)
    b.add_memlet(blk, c_in, "", t, "_in3", f"(iO*{CY} + iI)*{N} + jO*{CX} + jI", dev)
    b.add_memlet(blk, t, "_out", c_out, "", f"(iO*{CY} + iI)*{N} + jO*{CX} + jI", dev)
    b.end_for()
    b.end_for()
    b.end_reduce()
    b.end_map()
    b.end_map()

    off("C", "__daisy_dev_C", DataTransferDirection.D2H, BufferLifecycle.NO_CHANGE, nbC)
    for dc in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        off(dc, dc, DataTransferDirection.NONE, BufferLifecycle.FREE, "0")

    return b, k_loop, c_out


@pytest.mark.parametrize(
    "M,N,K,TY,TX,CY,CX",
    [(8, 8, 4, 2, 2, 2, 2), (16, 16, 8, 4, 4, 2, 2), (8, 8, 3, 2, 2, 2, 2)],
    ids=["8x8x4", "16x16x8", "8x8x3"],
)
def test_local_storage_register_tile_gemm_reduce(M, N, K, TY, TX, CY, CX, tmp_path):
    builder, k_loop, c_out = _build_gemm_regtile_reduce(M, N, K, TY, TX, CY, CX)

    am = AnalysisManager(builder)
    xform = LocalStorage(k_loop, c_out)
    assert xform.can_be_applied(
        builder, am
    ), "per-thread (sequential) reduction accumulator should localize"
    xform.apply(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    output_dir = tmp_path / f"gemm_reduce_{M}x{N}x{K}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    generated = "\n".join(p.read_text() for p in output_dir.rglob("*.cu"))
    assert "__daisy_local_storage" in generated, "C register tile not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))

    np.testing.assert_allclose(C, A @ B, rtol=1e-3, atol=1e-3)
