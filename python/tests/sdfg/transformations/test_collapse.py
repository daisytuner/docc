"""
Integration tests for the ``MapCollapse`` transformation on CPU.

Each test builds a map nest with ``StructuredSDFGBuilder``, collapses the outer
map with ``MapCollapse``, compiles the result for the sequential (CPU) target
and verifies the computed result against a NumPy reference.

Three nest shapes are covered:

* ``Map(map)``       - a single perfectly nested inner map (one matrix-vector add).
* ``Map(map, map)``  - two sibling inner maps of different column counts, each
                       adding a vector to every row of its own matrix.
* ``Map(block, map)``- a per-row scalar block followed by an inner map; the block
                       is replicated across the flattened iteration space while the
                       inner map is collapsed into it.

The matrices are stored row-major as flat buffers, so element ``(i, j)`` of an
``R x C`` matrix lives at offset ``i * C + j``.
"""

import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    MapCollapse,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
)
from docc.compiler.compiled_sdfg import CompiledSDFG


def _collapse_to_compiled(builder, outer_map, name, output_root):
    """Collapse ``outer_map`` and compile the SDFG for the CPU (sequential) target."""
    analysis_manager = AnalysisManager(builder)

    collapse = MapCollapse(outer_map, 2)
    assert collapse.can_be_applied(
        builder, analysis_manager
    ), "MapCollapse should be applicable to the outer map"
    collapse.apply(builder, analysis_manager)
    assert collapse.collapsed_loop is not None, "collapse produced no map"

    sdfg = builder.move()

    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_path = sdfg._compile(str(output_dir), "sequential")
    return CompiledSDFG(lib_path, sdfg)


# ---------------------------------------------------------------------------
# Map(map): single inner map -- B[i, j] = A[i, j] + v[j]
# ---------------------------------------------------------------------------


def _build_single_map():
    builder = StructuredSDFGBuilder("collapse_single_map")
    f = Scalar(PrimitiveType.Float)
    u = Scalar(PrimitiveType.UInt64)

    builder.add_container("N", u, is_argument=True)
    builder.add_container("M", u, is_argument=True)
    builder.add_container("A", Pointer(f), is_argument=True)
    builder.add_container("v", Pointer(f), is_argument=True)
    builder.add_container("B", Pointer(f), is_argument=True)
    builder.add_container("i", u)
    builder.add_container("j", u)

    outer = builder.begin_map("i", "0", "N", "1")
    builder.begin_map("j", "0", "M", "1")
    builder.add_assignment("B(i*M + j)", "A(i*M + j) + v(j)")
    builder.end_map()
    builder.end_map()

    return builder, outer


@pytest.mark.parametrize(
    "N,M",
    [(8, 5), (1, 1), (16, 32), (33, 7)],
    ids=["8x5", "1x1", "16x32", "33x7"],
)
def test_collapse_single_map(N, M, tmp_path):
    builder, outer = _build_single_map()
    compiled = _collapse_to_compiled(builder, outer, f"single_{N}x{M}", tmp_path)

    rng = np.random.default_rng(0)
    A = rng.standard_normal((N, M)).astype(np.float32)
    v = rng.standard_normal((M,)).astype(np.float32)
    B = np.zeros((N, M), dtype=np.float32)

    compiled(N, M, A, v, B)

    np.testing.assert_allclose(B, A + v[None, :], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Map(map, map): two sibling inner maps with differing column counts.
#   B1[i, j] = A1[i, j] + v1[j]   (M columns)
#   B2[i, k] = A2[i, k] + v2[k]   (P columns)
# Both reuse the same row count N.
# ---------------------------------------------------------------------------


def _build_two_maps():
    builder = StructuredSDFGBuilder("collapse_two_maps")
    f = Scalar(PrimitiveType.Float)
    u = Scalar(PrimitiveType.UInt64)

    for name in ("N", "M", "P"):
        builder.add_container(name, u, is_argument=True)
    for name in ("A1", "v1", "B1", "A2", "v2", "B2"):
        builder.add_container(name, Pointer(f), is_argument=True)
    for name in ("i", "j", "k"):
        builder.add_container(name, u)

    outer = builder.begin_map("i", "0", "N", "1")
    builder.begin_map("j", "0", "M", "1")
    builder.add_assignment("B1(i*M + j)", "A1(i*M + j) + v1(j)")
    builder.end_map()
    builder.begin_map("k", "0", "P", "1")
    builder.add_assignment("B2(i*P + k)", "A2(i*P + k) + v2(k)")
    builder.end_map()
    builder.end_map()

    return builder, outer


@pytest.mark.parametrize(
    "N,M,P",
    [(8, 5, 7), (4, 16, 3), (1, 1, 1), (10, 8, 12)],
    ids=["8x5x7", "4x16x3", "1x1x1", "10x8x12"],
)
def test_collapse_two_maps(N, M, P, tmp_path):
    builder, outer = _build_two_maps()
    compiled = _collapse_to_compiled(builder, outer, f"two_{N}x{M}x{P}", tmp_path)

    rng = np.random.default_rng(1)
    A1 = rng.standard_normal((N, M)).astype(np.float32)
    v1 = rng.standard_normal((M,)).astype(np.float32)
    B1 = np.zeros((N, M), dtype=np.float32)
    A2 = rng.standard_normal((N, P)).astype(np.float32)
    v2 = rng.standard_normal((P,)).astype(np.float32)
    B2 = np.zeros((N, P), dtype=np.float32)

    compiled(N, M, P, A1, v1, B1, A2, v2, B2)

    np.testing.assert_allclose(B1, A1 + v1[None, :], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(B2, A2 + v2[None, :], rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Map(block, map): a per-row scalar block followed by an inner map.
#   rowsum[i] = bias[i] + 1.0      (replicated across the flattened space)
#   B[i, j]   = A[i, j] + v[j]     (collapsed inner map)
# ---------------------------------------------------------------------------


def _build_block_map():
    builder = StructuredSDFGBuilder("collapse_block_map")
    f = Scalar(PrimitiveType.Float)
    u = Scalar(PrimitiveType.UInt64)

    builder.add_container("N", u, is_argument=True)
    builder.add_container("M", u, is_argument=True)
    for name in ("A", "v", "B", "bias", "rowsum"):
        builder.add_container(name, Pointer(f), is_argument=True)
    builder.add_container("i", u)
    builder.add_container("j", u)

    outer = builder.begin_map("i", "0", "N", "1")
    builder.add_assignment("rowsum(i)", "bias(i) + 1.0")
    builder.begin_map("j", "0", "M", "1")
    builder.add_assignment("B(i*M + j)", "A(i*M + j) + v(j)")
    builder.end_map()
    builder.end_map()

    return builder, outer


@pytest.mark.parametrize(
    "N,M",
    [(8, 5), (1, 1), (16, 32), (33, 7)],
    ids=["8x5", "1x1", "16x32", "33x7"],
)
def test_collapse_block_map(N, M, tmp_path):
    builder, outer = _build_block_map()
    compiled = _collapse_to_compiled(builder, outer, f"block_{N}x{M}", tmp_path)

    rng = np.random.default_rng(2)
    A = rng.standard_normal((N, M)).astype(np.float32)
    v = rng.standard_normal((M,)).astype(np.float32)
    B = np.zeros((N, M), dtype=np.float32)
    bias = rng.standard_normal((N,)).astype(np.float32)
    rowsum = np.zeros((N,), dtype=np.float32)

    compiled(N, M, A, v, B, bias, rowsum)

    np.testing.assert_allclose(B, A + v[None, :], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(rowsum, bias + 1.0, rtol=1e-5, atol=1e-6)
