"""ROCm e2e: rocWMMA MatMul loading its A/B operands from **shared (LDS)** memory.

This validates the target integration: instead of the MMA node loading fragments
directly from global memory, the block cooperatively stages the A- and B-tiles
into ``__shared__`` (LDS) buffers via ``TileCopyNode`` and the ``MatMulNode`` then
``rocwmma::load_matrix_sync``-es its fragments *from LDS*.  C is register-staged by
the accumulator fragment (intrinsic to the MMA node) and copied out to global.

The SDFG is hand-wired through the builder bindings (no automatic transform yet):

    for block_row (Y_GRID, step tile_m):
      for block_col (X_GRID, step tile_n):
        TileCopyNode:  dA[block_row:+tile_m, :K]  ->  sA[tile_m, K]   (cooperative, LDS)
        TileCopyNode:  dB[:K, block_col:+tile_n]  ->  sB[K, tile_n]   (cooperative, LDS)
        __syncthreads
        MatMulNode:    C_tile = sA @ sB   (rocWMMA reads sA/sB from LDS, stores to dC)

``add_tile_copy_node`` builds a whole-block cooperative ``TileCopyNode`` from two
``Tensor`` geometries (plan.src = global, plan.dst = buffer); the shared buffers are
declared as ``Array(..., StorageType.NV_Shared())`` (ROCm codegen emits NV_Shared as
LDS ``__shared__``).

Runs only on the matching GPU (``DOCC_ROCM_ARCH``); otherwise skipped.
"""

from pathlib import Path

import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    Array,
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
    tile_buffer_layout,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

PYTEST_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "pytestOutput"

HALF = Scalar(PrimitiveType.Half)
HALF_BYTES = 2

ARCHES = ["gfx90a", "gfx1201"]

# (M, N, K, tile_m, tile_n, b_col_major): K is staged whole into LDS (one mma_block_k = 16).
EXEC_CASES = [
    (128, 128, 16, 64, 64, False),  # 2x2 grid, A/B row-major LDS tiles
    (64, 64, 16, 64, 64, False),  # single block
    (
        128,
        128,
        16,
        64,
        64,
        True,
    ),  # A row-major, B col-major LDS tile (matrix-core preferred)
    (64, 64, 16, 64, 64, True),  # single block, B col-major
]


def _ceil_div(a, b):
    return -(-a // b)


def _build_shared_staged_mma(M, N, K, tile_m, tile_n, b_col_major=False):
    """Hand-wire the offloaded GEMM whose MMA operands are staged into LDS.

    With ``b_col_major`` the B-tile is staged *transposed* into a column-major LDS
    buffer (strides ``[1, K]``), so the MMA reads ``fragB`` as ``col_major`` — the
    layout the matrix cores load with contiguous-K.
    """
    b = StructuredSDFGBuilder("mma_shared")
    host = Pointer(HALF)
    dev = Pointer(HALF, StorageType.AMD_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b.add_container("A", host, is_argument=True)
    b.add_container("B", host, is_argument=True)
    b.add_container("C", host, is_argument=True)
    b.add_container("dA", dev)
    b.add_container("dB", dev)
    b.add_container("dC", dev)
    b.add_container("block_row", i32)
    b.add_container("block_col", i32)

    # LDS staging buffers: dense row-major tiles, NV_Shared -> __shared__ on ROCm.
    b.add_container("sA", Array(HALF, str(tile_m * K), StorageType.NV_Shared()))
    b.add_container("sB", Array(HALF, str(K * tile_n), StorageType.NV_Shared()))

    def off(hc, dc, direction, lifecycle, elems):
        b.add_rocm_offloading_block(
            hc, dc, direction, lifecycle, dev, f"{elems} * {HALF_BYTES}"
        )

    off("A", "dA", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * K)
    off("B", "dB", DataTransferDirection.H2D, BufferLifecycle.ALLOC, K * N)
    off("C", "dC", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * N)

    b.begin_map(
        "block_row",
        "0",
        str(M),
        str(tile_m),
        ScheduleType.rocm_offload(TargetLevel.Y_GRID, _ceil_div(M, tile_m)),
    )
    b.begin_map(
        "block_col",
        "0",
        str(N),
        str(tile_n),
        ScheduleType.rocm_offload(TargetLevel.X_GRID, _ceil_div(N, tile_n)),
    )

    # --- Stage A[block_row:+tile_m, :K] -> sA[tile_m, K] (cooperative, into LDS) ---
    a_global = Tensor(HALF, [str(tile_m), str(K)], [str(K), "1"], f"block_row * {K}")
    a_buffer = tile_buffer_layout(HALF, [str(tile_m), str(K)], "MultiDim")
    b.add_tile_copy_node("sA", "dA", a_global, a_buffer, Pointer(HALF), "in", "ROCM")

    # --- Stage B[:K, block_col:+tile_n] -> sB[K, tile_n] (cooperative, into LDS) ---
    # Buffer geometry from the tiles machinery: MultiDim (row-major) or Transposed
    # (column-major, strides [1, K]) — no hand-rolled strides.
    b_kind = "Transposed" if b_col_major else "MultiDim"
    b_global = Tensor(HALF, [str(K), str(tile_n)], [str(N), "1"], "block_col")
    b_buffer = tile_buffer_layout(HALF, [str(K), str(tile_n)], b_kind)
    b.add_tile_copy_node("sB", "dB", b_global, b_buffer, Pointer(HALF), "in", "ROCM")

    # Block barrier: all threads must finish staging before the waves read LDS.
    b.add_barrier_local_block()

    # --- MMA reads the same LDS geometries the copies wrote, stores C to global ---
    c_type = Tensor(
        HALF,
        [str(tile_m), str(tile_n)],
        [str(N), "1"],
        f"block_row * {N} + block_col",
    )
    node = b.add_matmul_op("sA", a_buffer, "sB", b_buffer, "dC", c_type)

    b.end_map()
    b.end_map()

    off("C", "dC", DataTransferDirection.D2H, BufferLifecycle.FREE, M * N)
    off("dA", "dA", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)
    off("dB", "dB", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)

    return b, node


@pytest.mark.rocm()
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n,b_col_major",
    EXEC_CASES,
    ids=[
        f"{m}x{n}x{k}_{tm}x{tn}_{'Bcol' if bc else 'Brow'}"
        for (m, n, k, tm, tn, bc) in EXEC_CASES
    ],
)
def test_mma_from_shared_executes(arch, M, N, K, tile_m, tile_n, b_col_major):
    if RocmArch.current_name() != arch:
        pytest.skip(f"DOCC_ROCM_ARCH ({RocmArch.current_name()}) != {arch}")

    builder, node = _build_shared_staged_mma(M, N, K, tile_m, tile_n, b_col_major)
    am = AnalysisManager(builder)
    xform = RocmMmaTransform(node, RocmArch.get_from_name(arch))
    assert xform.can_be_applied(builder, am), "MMA over the LDS tiles should expand"
    xform.apply(builder, am)
    assert xform.expanded

    sdfg = builder.move()
    tag = "Bcol" if b_col_major else "Brow"
    output_dir = (
        PYTEST_OUTPUT_DIR / f"mma_shared_{arch}_{M}x{N}x{K}_{tile_m}x{tile_n}_{tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sdfg.dump(str(output_dir), "expanded", True, True)
    sdfg.validate()

    lib_path = sdfg._compile(str(output_dir), "rocm")

    # The fragment loads must come from the LDS buffers, guarded by a block barrier.
    generated = "\n".join(p.read_text() for p in output_dir.rglob("*rocm.cpp"))
    assert "__shared__" in generated, "LDS staging buffers not emitted"
    assert "load_matrix_sync" in generated, "rocWMMA loads not emitted"

    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))

    ref = A.astype(np.float32) @ B.astype(np.float32)
    np.testing.assert_allclose(C.astype(np.float32), ref, rtol=5e-2, atol=5e-2)
