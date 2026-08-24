"""
Execution tests for the standalone ``LocalStorage`` transformation on CPU.

Each test builds a loop nest with ``StructuredSDFGBuilder``, localizes a
container's tile with ``LocalStorage`` (copy direction and storage space are
*derived* from the dataflow + schedule), compiles the result for the sequential
(CPU) target and checks the computed result against a NumPy reference — so we
validate that the buffer allocation, copy-in/out and access rewriting preserve
semantics, not just that the transformation applies.

The scenarios mirror the C++ apply unit tests:

* read-only cache        -- stage a reused ``A`` tile, copy-in only;
* write-only             -- localize a written ``C`` tile, writeback only;
* read-write accumulator -- copy-in + writeback (column sums);
* scalar accumulator     -- extent-1 tile promoted to a register;
* per-row cache          -- a tile whose base moves with the outer loop, re-staged
                            each outer iteration.

Matrices are stored row-major as flat buffers, so element ``(i, j)`` of an
``R x C`` matrix lives at offset ``i * C + j``.
"""

import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    LocalStorage,
    Pointer,
    PrimitiveType,
    Scalar,
    ScheduleType,
    StructuredSDFGBuilder,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

F = Scalar(PrimitiveType.Float)
U = Scalar(PrimitiveType.UInt64)
PF = Pointer(F)


def _apply_and_compile(builder, xform, name, output_root):
    """Assert ``xform`` applies, then compile the SDFG for the CPU target."""
    analysis_manager = AnalysisManager(builder)
    assert xform.can_be_applied(
        builder, analysis_manager
    ), "LocalStorage should be applicable"
    xform.apply(builder, analysis_manager)
    assert xform.local_container, "apply produced no local buffer"

    sdfg = builder.move()
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "sequential")
    return CompiledSDFG(lib_path, sdfg)


# ---------------------------------------------------------------------------
# Read-only cache: stage a reused A tile (copy-in only).
#   out[r] = sum_{i=0}^{K-1} A[i]   for r in 0..R   (A reused across r)
# Localize A at the inner i loop -> A[0:K] is staged into a buffer inside each r.
# ---------------------------------------------------------------------------


def _build_read_only_cache(K):
    builder = StructuredSDFGBuilder("ls_in_cache")
    builder.add_container("R", U, is_argument=True)
    builder.add_container("A", PF, is_argument=True)
    builder.add_container("out", PF, is_argument=True)
    builder.add_container("r", U)
    builder.add_container("i", U)

    builder.begin_for("r", "0", "R", "1")
    inner = builder.begin_for("i", "0", str(K), "1")
    block = builder.add_block()
    out_in = builder.add_access(block, "out")
    a_in = builder.add_access(block, "A")
    out_out = builder.add_access(block, "out")
    t = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(block, out_in, "", t, "_in1", subset="r", type=PF)
    builder.add_memlet(block, a_in, "", t, "_in2", subset="i", type=PF)
    builder.add_memlet(block, t, "_out", out_out, "", subset="r", type=PF)
    builder.end_for()
    builder.end_for()

    return builder, inner, a_in


@pytest.mark.parametrize("R,K", [(4, 8), (1, 16), (5, 3)], ids=["4x8", "1x16", "5x3"])
def test_local_storage_read_only_cache(R, K, tmp_path):
    builder, inner, a_in = _build_read_only_cache(K)
    compiled = _apply_and_compile(
        builder, LocalStorage(inner, a_in), f"in_{R}x{K}", tmp_path
    )

    rng = np.random.default_rng(0)
    A = rng.standard_normal((K,)).astype(np.float32)
    out = np.zeros((R,), dtype=np.float32)

    compiled(R, A, out)

    np.testing.assert_allclose(out, np.full(R, A.sum()), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Write-only: localize a written C tile (writeback only, no copy-in).
#   C[j] = A[j]   for j in 0..W
# ---------------------------------------------------------------------------


def _build_write_only(W):
    builder = StructuredSDFGBuilder("ls_out_write_only")
    builder.add_container("A", PF, is_argument=True)
    builder.add_container("C", PF, is_argument=True)
    builder.add_container("j", U)

    loop = builder.begin_for("j", "0", str(W), "1")
    block = builder.add_block()
    a_in = builder.add_access(block, "A")
    c_out = builder.add_access(block, "C")
    t = builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
    builder.add_memlet(block, a_in, "", t, "_in", subset="j", type=PF)
    builder.add_memlet(block, t, "_out", c_out, "", subset="j", type=PF)
    builder.end_for()

    return builder, loop, c_out


@pytest.mark.parametrize("W", [4, 1, 9], ids=["4", "1", "9"])
def test_local_storage_write_only(W, tmp_path):
    builder, loop, c_out = _build_write_only(W)
    compiled = _apply_and_compile(
        builder, LocalStorage(loop, c_out), f"out_wo_{W}", tmp_path
    )

    rng = np.random.default_rng(1)
    A = rng.standard_normal((W,)).astype(np.float32)
    C = np.zeros((W,), dtype=np.float32)

    compiled(A, C)

    np.testing.assert_allclose(C, A, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Read-write accumulator: copy-in + writeback (column sums).
#   C[j] += A[i*W + j]   over i in 0..M, j in 0..W
# Localize C at the outer i loop -> C[0:W] staged once, accumulated, written back.
# ---------------------------------------------------------------------------


def _build_accumulator(W):
    builder = StructuredSDFGBuilder("ls_out_accum")
    builder.add_container("M", U, is_argument=True)
    builder.add_container("A", PF, is_argument=True)
    builder.add_container("C", PF, is_argument=True)
    builder.add_container("i", U)
    builder.add_container("j", U)

    outer = builder.begin_for("i", "0", "M", "1")
    builder.begin_for("j", "0", str(W), "1")
    block = builder.add_block()
    c_in = builder.add_access(block, "C")
    a_in = builder.add_access(block, "A")
    c_out = builder.add_access(block, "C")
    t = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(block, c_in, "", t, "_in1", subset="j", type=PF)
    builder.add_memlet(block, a_in, "", t, "_in2", subset=f"i*{W} + j", type=PF)
    builder.add_memlet(block, t, "_out", c_out, "", subset="j", type=PF)
    builder.end_for()
    builder.end_for()

    return builder, outer, c_in


@pytest.mark.parametrize("M,W", [(6, 4), (3, 1), (10, 8)], ids=["6x4", "3x1", "10x8"])
def test_local_storage_accumulator(M, W, tmp_path):
    builder, outer, c_in = _build_accumulator(W)
    compiled = _apply_and_compile(
        builder, LocalStorage(outer, c_in), f"accum_{M}x{W}", tmp_path
    )

    rng = np.random.default_rng(2)
    A = rng.standard_normal((M, W)).astype(np.float32)
    C = np.zeros((W,), dtype=np.float32)

    compiled(M, A.reshape(-1), C)

    np.testing.assert_allclose(C, A.sum(axis=0), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Scalar accumulator: an extent-1 tile promoted to a register.
#   C[0] += A[i]   over i in 0..K
# ---------------------------------------------------------------------------


def _build_scalar_accumulator(K):
    builder = StructuredSDFGBuilder("ls_scalar_accum")
    builder.add_container("A", PF, is_argument=True)
    builder.add_container("C", PF, is_argument=True)
    builder.add_container("i", U)

    loop = builder.begin_for("i", "0", str(K), "1")
    block = builder.add_block()
    c_in = builder.add_access(block, "C")
    a_in = builder.add_access(block, "A")
    c_out = builder.add_access(block, "C")
    t = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(block, c_in, "", t, "_in1", subset="0", type=PF)
    builder.add_memlet(block, a_in, "", t, "_in2", subset="i", type=PF)
    builder.add_memlet(block, t, "_out", c_out, "", subset="0", type=PF)
    builder.end_for()

    return builder, loop, c_in


@pytest.mark.parametrize("K", [8, 1, 17], ids=["8", "1", "17"])
def test_local_storage_scalar_accumulator(K, tmp_path):
    builder, loop, c_in = _build_scalar_accumulator(K)
    compiled = _apply_and_compile(
        builder, LocalStorage(loop, c_in), f"scalar_{K}", tmp_path
    )

    rng = np.random.default_rng(3)
    A = rng.standard_normal((K,)).astype(np.float32)
    C = np.zeros((1,), dtype=np.float32)

    compiled(A, C)

    np.testing.assert_allclose(C[0], A.sum(), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Per-row cache: a tile whose base moves with the outer loop.
#   B[i*W + j] = A[i*W + j]   over i in 0..R, j in 0..W
# Localize A at the inner j loop -> the A[i*W : i*W+W] row is re-staged each i.
# ---------------------------------------------------------------------------


def _build_row_cache(W):
    builder = StructuredSDFGBuilder("ls_row_cache")
    builder.add_container("R", U, is_argument=True)
    builder.add_container("A", PF, is_argument=True)
    builder.add_container("B", PF, is_argument=True)
    builder.add_container("i", U)
    builder.add_container("j", U)

    builder.begin_for("i", "0", "R", "1")
    inner = builder.begin_for("j", "0", str(W), "1")
    block = builder.add_block()
    a_in = builder.add_access(block, "A")
    b_out = builder.add_access(block, "B")
    t = builder.add_tasklet(block, TaskletCode.assign, ["_in"], ["_out"])
    builder.add_memlet(block, a_in, "", t, "_in", subset=f"i*{W} + j", type=PF)
    builder.add_memlet(block, t, "_out", b_out, "", subset=f"i*{W} + j", type=PF)
    builder.end_for()
    builder.end_for()

    return builder, inner, a_in


@pytest.mark.parametrize("R,W", [(5, 4), (1, 8), (7, 3)], ids=["5x4", "1x8", "7x3"])
def test_local_storage_row_cache(R, W, tmp_path):
    builder, inner, a_in = _build_row_cache(W)
    compiled = _apply_and_compile(
        builder, LocalStorage(inner, a_in), f"row_{R}x{W}", tmp_path
    )

    rng = np.random.default_rng(4)
    A = rng.standard_normal((R, W)).astype(np.float32)
    B = np.zeros((R, W), dtype=np.float32)

    compiled(R, A.reshape(-1), B.reshape(-1))

    np.testing.assert_allclose(B, A, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# OpenMP: per-thread tile under a CPU-parallel map.
#   out[i] = sum_{k=0}^{K-1} A[i*K + k]   for i in 0..N   (i is OMP-parallel)
# Localize the A row at the inner k loop: the tile base i*K uses the parallel
# indvar, so the tile is *per-thread* -> a private CPU_Stack buffer. The OMP map
# dispatcher must privatise the buffer (else threads race on a shared row cache),
# so a correct result across threads validates the derivation + codegen.
# ---------------------------------------------------------------------------


def _build_omp_row_reduce(K):
    builder = StructuredSDFGBuilder("ls_omp_row")
    builder.add_container("N", U, is_argument=True)
    builder.add_container("A", PF, is_argument=True)
    builder.add_container("out", PF, is_argument=True)
    builder.add_container("i", U)
    builder.add_container("k", U)

    builder.begin_map("i", "0", "N", "1", ScheduleType.omp())
    inner = builder.begin_for("k", "0", str(K), "1")
    block = builder.add_block()
    out_in = builder.add_access(block, "out")
    a_in = builder.add_access(block, "A")
    out_out = builder.add_access(block, "out")
    t = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(block, out_in, "", t, "_in1", subset="i", type=PF)
    builder.add_memlet(block, a_in, "", t, "_in2", subset=f"i*{K} + k", type=PF)
    builder.add_memlet(block, t, "_out", out_out, "", subset="i", type=PF)
    builder.end_for()
    builder.end_map()

    return builder, inner, a_in


@pytest.mark.parametrize(
    "N,K", [(64, 8), (33, 5), (128, 16)], ids=["64x8", "33x5", "128x16"]
)
def test_local_storage_omp_per_thread(N, K, tmp_path):
    builder, inner, a_in = _build_omp_row_reduce(K)
    analysis_manager = AnalysisManager(builder)
    xform = LocalStorage(inner, a_in)
    assert xform.can_be_applied(
        builder, analysis_manager
    ), "per-thread tile under OMP should be a private buffer"
    xform.apply(builder, analysis_manager)

    sdfg = builder.move()
    output_dir = tmp_path / f"omp_{N}x{K}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "openmp")
    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(5)
    A = rng.standard_normal((N, K)).astype(np.float32)
    out = np.zeros((N,), dtype=np.float32)

    compiled(N, A.reshape(-1), out)

    np.testing.assert_allclose(out, A.sum(axis=1), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# OpenMP: a tile that is cooperative across the parallel dim is rejected.
#   out[i] = sum_{k=0}^{K-1} B[k]   for i in 0..N   (i is OMP-parallel)
# Localizing B at the inner k loop would share one buffer across all threads
# (its base is independent of i) -> a private stack cannot serve it, so
# derive_storage rejects it.
# ---------------------------------------------------------------------------


def test_local_storage_omp_cooperative_rejected():
    K = 8
    builder = StructuredSDFGBuilder("ls_omp_coop")
    builder.add_container("N", U, is_argument=True)
    builder.add_container("B", PF, is_argument=True)
    builder.add_container("out", PF, is_argument=True)
    builder.add_container("i", U)
    builder.add_container("k", U)

    builder.begin_map("i", "0", "N", "1", ScheduleType.omp())
    inner = builder.begin_for("k", "0", str(K), "1")
    block = builder.add_block()
    out_in = builder.add_access(block, "out")
    b_in = builder.add_access(block, "B")
    out_out = builder.add_access(block, "out")
    t = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(block, out_in, "", t, "_in1", subset="i", type=PF)
    builder.add_memlet(block, b_in, "", t, "_in2", subset="k", type=PF)
    builder.add_memlet(block, t, "_out", out_out, "", subset="i", type=PF)
    builder.end_for()
    builder.end_map()

    analysis_manager = AnalysisManager(builder)
    xform = LocalStorage(inner, b_in)
    assert not xform.can_be_applied(
        builder, analysis_manager
    ), "cooperative tile across an OMP-parallel dim must be rejected"


# ---------------------------------------------------------------------------
# Reduction-accumulator privatization (gemv-style): a *sequential* Reduce over j
# accumulates y[iO*CY + iI] for a per-iO block of CY outputs. Localizing y at the
# Reduce loads the block once, accumulates in a private buffer across j, writes
# back once, and retargets the Reduce descriptor to the buffer. A correct result
# validates the copy-in/out + the reduction-container retarget end to end.
#   y[iO*CY + iI] = sum_{j=0}^{K-1} A[(iO*CY+iI)*K + j] * x[j]
# ---------------------------------------------------------------------------


def _build_reduce_accumulator(K, CY):
    builder = StructuredSDFGBuilder("ls_reduce_acc")
    builder.add_container("R", U, is_argument=True)
    builder.add_container("A", PF, is_argument=True)
    builder.add_container("x", PF, is_argument=True)
    builder.add_container("y", PF, is_argument=True)
    builder.add_container("iO", U)
    builder.add_container("j", U)
    builder.add_container("iI", U)

    builder.begin_for("iO", "0", "R", "1")
    reduce = builder.begin_reduce("j", "0", str(K), "1", [("add", "y")])
    builder.begin_for("iI", "0", str(CY), "1")
    block = builder.add_block()
    a_in = builder.add_access(block, "A")
    x_in = builder.add_access(block, "x")
    y_in = builder.add_access(block, "y")
    y_out = builder.add_access(block, "y")
    t = builder.add_tasklet(
        block, TaskletCode.fp_fma, ["_in1", "_in2", "_in3"], ["_out"]
    )
    m = f"(iO*{CY} + iI)"
    builder.add_memlet(block, a_in, "", t, "_in1", subset=f"{m}*{K} + j", type=PF)
    builder.add_memlet(block, x_in, "", t, "_in2", subset="j", type=PF)
    builder.add_memlet(block, y_in, "", t, "_in3", subset=m, type=PF)
    builder.add_memlet(block, t, "_out", y_out, "", subset=m, type=PF)
    builder.end_for()  # iI
    builder.end_reduce()  # j
    builder.end_for()  # iO

    return builder, reduce, y_out


@pytest.mark.parametrize(
    "R,K,CY", [(4, 8, 2), (1, 5, 4), (3, 3, 2)], ids=["4x8x2", "1x5x4", "3x3x2"]
)
def test_local_storage_reduction_accumulator(R, K, CY, tmp_path):
    builder, reduce, y_out = _build_reduce_accumulator(K, CY)
    analysis_manager = AnalysisManager(builder)
    xform = LocalStorage(reduce, y_out)
    assert xform.can_be_applied(
        builder, analysis_manager
    ), "sequential reduction accumulator should be localizable"
    xform.apply(builder, analysis_manager)

    sdfg = builder.move()
    output_dir = tmp_path / f"reduce_{R}x{K}x{CY}"
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "sequential")
    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(7)
    M = R * CY
    A = rng.standard_normal((M, K)).astype(np.float32)
    x = rng.standard_normal((K,)).astype(np.float32)
    y = np.zeros((M,), dtype=np.float32)

    compiled(R, A.reshape(-1), x, y)

    np.testing.assert_allclose(y, A @ x, rtol=1e-4, atol=1e-4)
