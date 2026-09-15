"""ROCm MMA (rocwmma) expansion tests for the ``RocmMmaTransform`` transformation.

These mirror the C++ ``ROCMMMATest`` unit tests (``opt/tests/rocm/rocm_mma_test.cpp``)
but drive the *builder* API from Python:

* ``_build_offloaded_mma`` recreates the offloaded MatMul input SDFG -- two grid
  maps (Y_GRID over the C rows, X_GRID over the C columns) whose innermost body
  holds a single ``MatMulNode`` over the per-block tile.  The tile layouts follow
  the row-major layout of the full ``M x K`` / ``K x N`` / ``M x N`` matrices, with
  the tile offset selected by the map induction variables.
* ``_apply`` builds the ``RocmMmaTransform`` transformation for a target ``RocmArch``
  and reports whether it applies.

The expander only accepts tiles whose ``M``/``N``/``K`` extents are whole multiples
of the arch's 16-wide MMA block *and* whose resulting (m_blocks, n_blocks) shape is
one the arch supports -- so we assert it applies for the aligned tiles and declines
for the unsupported / mis-aligned ones (both for gfx1201 and gfx90a).

The expanded kernel is only *compiled and executed* when running on the matching
GPU (``DOCC_ROCM_ARCH`` names the arch under test); otherwise we exercise the pure
apply/decline logic, which needs no device.
"""

from pathlib import Path

import numpy as np
import pytest

PYTEST_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "pytestOutput"

from docc.sdfg import (
    AnalysisManager,
    BufferLifecycle,
    DataTransferDirection,
    Pointer,
    PrimitiveType,
    RocmArch,
    RocmMmaTransform,
    Scalar,
    ScheduleType,
    StorageType,
    StructuredSDFGBuilder,
    TargetLevel,
    Tensor,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

HALF = Scalar(PrimitiveType.Half)
HALF_BYTES = 2


def _ceil_div(a, b):
    return -(-a // b)


def _add_tile_matmul(builder, M, N, K, tile_m, tile_n, a="A", b="B", c="C"):
    """Add the per-block ``MatMulNode`` for the current (block_row, block_col) tile.

    Strides follow the row-major layout of the full matrices; the offset selects
    the tile inside the base pointer:
        A tile [tile_m, K]      at row block_row      -> offset block_row * K
        B tile [K, tile_n]      at col block_col      -> offset block_col
        C tile [tile_m, tile_n] at (block_row, col)   -> offset block_row * N + block_col
    """
    a_type = Tensor(HALF, [str(tile_m), str(K)], [str(K), "1"], f"block_row * {K}")
    b_type = Tensor(HALF, [str(K), str(tile_n)], [str(N), "1"], "block_col")
    c_type = Tensor(
        HALF,
        [str(tile_m), str(tile_n)],
        [str(N), "1"],
        f"block_row * {N} + block_col",
    )
    return builder.add_matmul_op(a, a_type, b, b_type, c, c_type)


def _build_offloaded_mma(M, N, K, tile_m, tile_n):
    """Recreate the offloaded MatMul SDFG from the C++ ``build_offloaded_mma_structure``.

    Device pointers ``A``/``B``/``C`` are kernel arguments; the outer Y_GRID map
    walks the C rows in ``tile_m`` steps, the inner X_GRID map the C columns in
    ``tile_n`` steps, and the innermost block computes one output tile.
    """
    builder = StructuredSDFGBuilder("test_mma")
    dev = Pointer(HALF, StorageType.AMD_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    builder.add_container("A", dev, is_argument=True)
    builder.add_container("B", dev, is_argument=True)
    builder.add_container("C", dev, is_argument=True)
    builder.add_container("block_row", i32)
    builder.add_container("block_col", i32)

    builder.begin_map(
        "block_row",
        "0",
        str(M),
        str(tile_m),
        ScheduleType.rocm_offload(TargetLevel.Y_GRID, _ceil_div(M, tile_m)),
    )
    builder.begin_map(
        "block_col",
        "0",
        str(N),
        str(tile_n),
        ScheduleType.rocm_offload(TargetLevel.X_GRID, _ceil_div(N, tile_n)),
    )
    node = _add_tile_matmul(builder, M, N, K, tile_m, tile_n)
    builder.end_map()
    builder.end_map()

    return builder, node


def _apply(builder, node, arch_name):
    """Build ``RocmMmaTransform`` for ``arch_name`` and return (transform, analysis_manager)."""
    am = AnalysisManager(builder)
    xform = RocmMmaTransform(node, RocmArch.get_from_name(arch_name))
    return xform, am


ARCHES = ["gfx1201", "gfx90a"]

# (m_blocks, n_blocks) shapes the 16-wide MMA supports: 1x1, 2x2, 4x4.
ALIGNED = [
    (1024, 1024, 1024, 16, 16),
    (1024, 1024, 1024, 32, 32),
    (1024, 1024, 1024, 64, 64),
]

# Tiles the expander must decline.
UNSUPPORTED = [
    (1024, 1024, 1024, 48, 48),  # (3, 3) block shape is not supported
    (1024, 1024, 1024, 16, 48),  # (1, 3) block shape is not supported
    (1024, 1024, 1020, 16, 16),  # K not a multiple of the 16-wide MMA block
    (1024, 1024, 1024, 17, 16),  # tile_m not a multiple of the 16-wide MMA block
    (1024, 1024, 1024, 16, 17),  # tile_n not a multiple of the 16-wide MMA block
]


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n",
    ALIGNED,
    ids=[f"{m}x{n}x{k}_{tm}x{tn}" for (m, n, k, tm, tn) in ALIGNED],
)
def test_mma_expand_applies(arch, M, N, K, tile_m, tile_n):
    builder, node = _build_offloaded_mma(M, N, K, tile_m, tile_n)
    xform, am = _apply(builder, node, arch)

    assert xform.can_be_applied(
        builder, am
    ), f"aligned {tile_m}x{tile_n} tile should expand on {arch}"
    xform.apply(builder, am)
    assert xform.expanded

    sdfg = builder.move()
    sdfg.validate()


@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n",
    UNSUPPORTED,
    ids=[f"{m}x{n}x{k}_{tm}x{tn}" for (m, n, k, tm, tn) in UNSUPPORTED],
)
def test_mma_expand_declines(arch, M, N, K, tile_m, tile_n):
    builder, node = _build_offloaded_mma(M, N, K, tile_m, tile_n)
    xform, am = _apply(builder, node, arch)

    assert not xform.can_be_applied(
        builder, am
    ), f"unsupported {tile_m}x{tile_n}/K={K} tile must not expand on {arch}"


# ---------------------------------------------------------------------------
# Execution: only on the matching GPU (DOCC_ROCM_ARCH names the arch under test).
# ---------------------------------------------------------------------------


def _build_executable_mma(M, N, K, tile_m, tile_n):
    """Full runnable matmul: host args, device buffers, H2D/D2H transfers, kernel."""
    builder = StructuredSDFGBuilder("test_mma_exec")
    host = Pointer(HALF)
    dev = Pointer(HALF, StorageType.AMD_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    builder.add_container("A", host, is_argument=True)
    builder.add_container("B", host, is_argument=True)
    builder.add_container("C", host, is_argument=True)
    builder.add_container("dA", dev)
    builder.add_container("dB", dev)
    builder.add_container("dC", dev)
    builder.add_container("block_row", i32)
    builder.add_container("block_col", i32)

    def off(hc, dc, direction, lifecycle, elems):
        builder.add_rocm_offloading_block(
            hc, dc, direction, lifecycle, dev, f"{elems} * {HALF_BYTES}"
        )

    off("A", "dA", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * K)
    off("B", "dB", DataTransferDirection.H2D, BufferLifecycle.ALLOC, K * N)
    off("C", "dC", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * N)

    builder.begin_map(
        "block_row",
        "0",
        str(M),
        str(tile_m),
        ScheduleType.rocm_offload(TargetLevel.Y_GRID, _ceil_div(M, tile_m)),
    )
    builder.begin_map(
        "block_col",
        "0",
        str(N),
        str(tile_n),
        ScheduleType.rocm_offload(TargetLevel.X_GRID, _ceil_div(N, tile_n)),
    )
    node = _add_tile_matmul(builder, M, N, K, tile_m, tile_n, a="dA", b="dB", c="dC")
    builder.end_map()
    builder.end_map()

    off("C", "dC", DataTransferDirection.D2H, BufferLifecycle.FREE, M * N)
    off("dA", "dA", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)
    off("dB", "dB", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)

    return builder, node


EXEC_CASES = [
    (32, 32, 32, 16, 16),
    (64, 64, 32, 16, 16),
    (32, 32, 32, 32, 32),
]


@pytest.mark.rocm()
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n",
    EXEC_CASES,
    ids=[f"{m}x{n}x{k}_{tm}x{tn}" for (m, n, k, tm, tn) in EXEC_CASES],
)
def test_mma_expand_executes(arch, M, N, K, tile_m, tile_n):
    if RocmArch.current_name() != arch:
        pytest.skip(f"DOCC_ROCM_ARCH ({RocmArch.current_name()}) != {arch}")

    builder, node = _build_executable_mma(M, N, K, tile_m, tile_n)
    xform, am = _apply(builder, node, arch)
    assert xform.can_be_applied(builder, am)
    xform.apply(builder, am)
    assert xform.expanded
    sdfg = builder.move()

    output_dir = PYTEST_OUTPUT_DIR / f"mma_{arch}_{M}x{N}x{K}_{tile_m}x{tile_n}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sdfg.dump(str(output_dir), "expanded", True, True)
    sdfg.validate()

    lib_path = sdfg._compile(str(output_dir), "rocm")
    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))

    ref = A.astype(np.float32) @ B.astype(np.float32)
    np.testing.assert_allclose(C.astype(np.float32), ref, rtol=5e-2, atol=5e-2)
