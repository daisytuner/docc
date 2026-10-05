"""
Execution tests for parallelogram time tiling with a tile-level wavefront on OpenMP.

Each test builds a stencil nest with ``StructuredSDFGBuilder``, applies the full
transformation chain (strip-mine, skew, interchange, tile, in-tile interchange,
wavefront skew, wavefront interchange, parallelization), compiles for OpenMP and
compares the result against a NumPy reference. Sizes are chosen so that bands
and tiles are partial at the domain boundaries.
"""

import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    LoopInterchange,
    LoopParallelization,
    LoopSkewing,
    LoopTiling,
    OMPTransform,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

D = Scalar(PrimitiveType.Double)
I = Scalar(PrimitiveType.Int64)


def _loop(builder, indvar):
    return AnalysisManager(builder).loop_analysis().find_loop_by_indvar(indvar)


def _apply(builder, transformation):
    analysis_manager = AnalysisManager(builder)
    assert transformation.can_be_applied(builder, analysis_manager), transformation
    transformation.apply(builder, analysis_manager)


def _compile(builder, name, output_root):
    sdfg = builder.move()
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    return CompiledSDFG(sdfg._compile(str(output_dir), "openmp"), sdfg)


def _add_sum3(builder, src, idx, dst, dst_idx, src_type, dst_type, tmps, scale):
    """dst[dst_idx] = scale * (src[idx[0]] + src[idx[1]] + src[idx[2]])"""
    block = builder.add_block()
    a = builder.add_access(block, src)
    t1 = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(block, a, "", t1, "_in1", subset=idx[0], type=src_type)
    builder.add_memlet(block, a, "", t1, "_in2", subset=idx[1], type=src_type)
    builder.add_memlet(
        block, t1, "_out", builder.add_access(block, tmps[0]), "", type=D
    )

    block = builder.add_block()
    t2 = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(
        block, builder.add_access(block, tmps[0]), "", t2, "_in1", type=D
    )
    builder.add_memlet(
        block,
        builder.add_access(block, src),
        "",
        t2,
        "_in2",
        subset=idx[2],
        type=src_type,
    )
    builder.add_memlet(
        block, t2, "_out", builder.add_access(block, tmps[1]), "", type=D
    )

    block = builder.add_block()
    t3 = builder.add_tasklet(block, TaskletCode.fp_mul, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(
        block, builder.add_constant(block, scale, D), "", t3, "_in1", type=D
    )
    builder.add_memlet(
        block, builder.add_access(block, tmps[1]), "", t3, "_in2", type=D
    )
    builder.add_memlet(
        block,
        t3,
        "_out",
        builder.add_access(block, dst),
        "",
        subset=dst_idx,
        type=dst_type,
    )


# ---------------------------------------------------------------------------
# Gauss-Seidel-like nest with dependences (0, 1) and (1, -1):
#   for i in [1, N): for j in [1, M-1): A[i][j] = A[i-1][j+1] + A[i][j-1]
# ---------------------------------------------------------------------------


def _build_gauss_seidel(M):
    builder = StructuredSDFGBuilder("gauss_seidel")
    # Row-major N x M matrix as a flat buffer: element (i, j) lives at i * M + j.
    # M is a constant (as for shape-specialized frontends) so the linearized accesses delinearize.
    row = Pointer(D)
    builder.add_container("N", I, is_argument=True)
    builder.add_container("A", row, is_argument=True)
    builder.add_container("i", I)
    builder.add_container("j", I)

    builder.begin_for("i", "1", "N", "1")
    builder.begin_for("j", "1", f"{M} - 1", "1")
    block = builder.add_block()
    a_in = builder.add_access(block, "A")
    a_out = builder.add_access(block, "A")
    t = builder.add_tasklet(block, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
    builder.add_memlet(
        block, a_in, "", t, "_in1", subset=f"(i - 1) * {M} + j + 1", type=row
    )
    builder.add_memlet(block, a_in, "", t, "_in2", subset=f"i * {M} + j - 1", type=row)
    builder.add_memlet(block, t, "_out", a_out, "", subset=f"i * {M} + j", type=row)
    builder.end_for()
    builder.end_for()
    return builder


def _gauss_seidel_reference(A):
    A = A.copy()
    N, M = A.shape
    for i in range(1, N):
        for j in range(1, M - 1):
            A[i, j] = A[i - 1, j + 1] + A[i, j - 1]
    return A


@pytest.mark.parametrize(
    "N,M", [(5, 7), (33, 34), (70, 101)], ids=["5x7", "33x34", "70x101"]
)
def test_parallelogram_tiling_gauss_seidel(N, M, tmp_path):
    builder = _build_gauss_seidel(M)

    # Parallelogram tiles (32x32): strip i, skew within the band, interchange, tile j, sweep tiles row-wise.
    _apply(builder, LoopTiling(_loop(builder, "i"), 32))
    _apply(builder, LoopSkewing(_loop(builder, "i"), _loop(builder, "j"), 1))
    _apply(builder, LoopInterchange(_loop(builder, "i"), _loop(builder, "j")))
    _apply(builder, LoopTiling(_loop(builder, "j"), 32))
    _apply(builder, LoopInterchange(_loop(builder, "j"), _loop(builder, "i")))
    # Tile distances (0,1), (1,0), (1,-1): wavefront 2*band + tile.
    _apply(
        builder, LoopSkewing(_loop(builder, "i_tile0"), _loop(builder, "j_tile0"), 2)
    )
    _apply(
        builder, LoopInterchange(_loop(builder, "i_tile0"), _loop(builder, "j_tile0"))
    )
    _apply(builder, LoopParallelization(_loop(builder, "i_tile0")))
    _apply(builder, OMPTransform(_loop(builder, "i_tile0")))

    compiled = _compile(builder, f"gauss_seidel_{N}x{M}", tmp_path)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, M))
    expected = _gauss_seidel_reference(A)
    flat = A.reshape(-1).copy()
    compiled(N, flat)
    np.testing.assert_allclose(flat.reshape(N, M), expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# Jacobi-1D with both sweeps fused and the A-update shifted by one:
#   for t in [0, T): for f in [0, N-1):
#     if f < N-2: B[f+1] = (A[f] + A[f+1] + A[f+2]) / 3
#     if f >= 1:  A[f]   = (B[f-1] + B[f] + B[f+1]) / 3
# ---------------------------------------------------------------------------


def _build_jacobi_1d():
    builder = StructuredSDFGBuilder("jacobi_1d")
    vec = Pointer(D)
    builder.add_container("T", I, is_argument=True)
    builder.add_container("N", I, is_argument=True)
    builder.add_container("A", vec, is_argument=True)
    builder.add_container("B", vec, is_argument=True)
    builder.add_container("t", I)
    builder.add_container("f", I)
    for tmp in ("s1", "s2", "s3", "s4"):
        builder.add_container(tmp, D)

    builder.begin_for("t", "0", "T", "1")
    builder.begin_for("f", "0", "N - 1", "1")
    builder.begin_if("f < N - 2")
    _add_sum3(
        builder,
        "A",
        ("f", "f + 1", "f + 2"),
        "B",
        "f + 1",
        vec,
        vec,
        ("s1", "s2"),
        "0.333",
    )
    builder.end_if()
    builder.begin_if("f >= 1")
    _add_sum3(
        builder, "B", ("f - 1", "f", "f + 1"), "A", "f", vec, vec, ("s3", "s4"), "0.333"
    )
    builder.end_if()
    builder.end_for()
    builder.end_for()
    return builder


def _jacobi_1d_reference(T, A, B):
    A, B = A.copy(), B.copy()
    for _ in range(T):
        B[1:-1] = 0.333 * (A[:-2] + A[1:-1] + A[2:])
        A[1:-1] = 0.333 * (B[:-2] + B[1:-1] + B[2:])
    return A, B


@pytest.mark.parametrize(
    "T,N", [(1, 3), (5, 33), (37, 150)], ids=["1x3", "5x33", "37x150"]
)
def test_parallelogram_tiling_jacobi_1d(T, N, tmp_path):
    builder = _build_jacobi_1d()

    # Parallelogram tiles (16x32): strip t, skew by the stencil slope, interchange, tile f, sweep tiles row-wise.
    _apply(builder, LoopTiling(_loop(builder, "t"), 16))
    _apply(builder, LoopSkewing(_loop(builder, "t"), _loop(builder, "f"), 2))
    _apply(builder, LoopInterchange(_loop(builder, "t"), _loop(builder, "f")))
    _apply(builder, LoopTiling(_loop(builder, "f"), 32))
    _apply(builder, LoopInterchange(_loop(builder, "f"), _loop(builder, "t")))
    # A band shifts by 2*16 = 32 = one tile: wavefront 2*band + tile, i.e. 4 per time step.
    _apply(
        builder, LoopSkewing(_loop(builder, "t_tile0"), _loop(builder, "f_tile0"), 4)
    )
    _apply(
        builder, LoopInterchange(_loop(builder, "t_tile0"), _loop(builder, "f_tile0"))
    )
    _apply(builder, LoopParallelization(_loop(builder, "t_tile0")))
    _apply(builder, OMPTransform(_loop(builder, "t_tile0")))

    compiled = _compile(builder, f"jacobi_1d_{T}x{N}", tmp_path)

    rng = np.random.default_rng(1)
    A = rng.standard_normal(N)
    B = rng.standard_normal(N)
    expected_A, expected_B = _jacobi_1d_reference(T, A, B)
    compiled(T, N, A, B)
    np.testing.assert_allclose(A, expected_A, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(B, expected_B, rtol=1e-12, atol=1e-12)
