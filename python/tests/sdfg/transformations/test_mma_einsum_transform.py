"""ROCm MMA (rocwmma) expansion tests for ``GpuMmaEinsumTransform``.

The sibling ``test_mma_transform.py`` drives ``GpuMmaTransform`` over an explicit
``MatMulNode``. Here we instead recreate the *raw* offloaded matmul loop nest the
frontend emits (see ``__docc_GraphModule71`` dot dumps): two grid maps, two
sequential tile loops, a sequential ``Reduce`` over K and an ``fma`` tasklet
``C[i,j] += A[i,k] * B[k,j]``. ``GpuMmaEinsumTransform`` then detects that nest as
an einsum matmul and lowers it to tensor-core MMA.

The expanded kernel is only compiled & executed on the matching GPU
(``DOCC_ROCM_ARCH`` names the arch under test); otherwise we exercise the pure
apply/decline logic, which needs no device.
"""

from pathlib import Path

import numpy as np
import pytest

PYTEST_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "pytestOutput"

from docc.sdfg import (
    AnalysisManager,
    Block,
    BufferLifecycle,
    DataTransferDirection,
    GpuMmaEinsumTransform,
    IfElse,
    LocalStorage,
    StripMining,
    Pointer,
    PrimitiveType,
    RocmArch,
    Scalar,
    ScheduleType,
    Sequence,
    StorageType,
    StructuredLoop,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

HALF = Scalar(PrimitiveType.Half)
HALF_BYTES = 2


def _ceil_div(a, b):
    return -(-a // b)


# --- Bare-binding equivalents of the gym loop/access helpers (see agent_rocm.py) ---


def _find_access_in_sequence(seq, container, want_read=True):
    """Recursively find an AccessNode for ``container`` in a body sequence."""
    for i in range(len(seq)):
        child = seq[i]
        if isinstance(child, Block):
            primary = child.dataflow.reads if want_read else child.dataflow.writes
            fallback = child.dataflow.writes if want_read else child.dataflow.reads
            for node in list(primary) + list(fallback):
                if node.data == container:
                    return node
        elif isinstance(child, StructuredLoop):
            found = _find_access_in_sequence(child.body, container, want_read)
            if found is not None:
                return found
        elif isinstance(child, Sequence):
            found = _find_access_in_sequence(child, container, want_read)
            if found is not None:
                return found
        elif isinstance(child, IfElse):
            for case_idx in range(child.size):
                found = _find_access_in_sequence(
                    child.case(case_idx), container, want_read
                )
                if found is not None:
                    return found
    return None


def _access_in_loop(loop, container, want_read=True):
    node = _find_access_in_sequence(loop.body, container, want_read)
    if node is None:
        raise ValueError(f"No access to {container!r} found in loop {loop.indvar!r}")
    return node


def _localize_operands(builder, a_name, b_name, k_panel_blocks=2):
    """Stage the A and B global tiles into LDS per K-panel.

    Mirrors ``agent_rocm.py`` Step 5: the expander's K-block loop ``tile_k0`` is
    strip-mined into panels of ``k_panel_blocks`` MMA blocks and LocalStorage
    localizes each read operand on the inner (per-panel) loop ``tile_k0``.
    """
    am = AnalysisManager(builder)
    k_loop = am.loop_analysis().find_loop_by_indvar("tile_k0")
    assert k_loop is not None, "expander must create the 'tile_k0' K-block loop"
    tiling = StripMining(k_loop, k_panel_blocks, True)
    assert tiling.can_be_applied(builder, am)
    tiling.apply(builder, am)

    for container in (a_name, b_name):
        am = AnalysisManager(builder)
        k_loop = am.loop_analysis().find_loop_by_indvar("tile_k0")
        access = _access_in_loop(k_loop, container, want_read=True)
        ls = LocalStorage(k_loop, access, swizzle_layout=False, lane_contiguous=False)
        assert ls.can_be_applied(
            builder, am
        ), f"LocalStorage should apply for {container}"
        ls.apply(builder, am)


def _add_einsum_mma_nest(
    builder, M, N, K, tile_m, tile_n, a, b, c, swap_operands=False
):
    """Build the raw matmul loop nest ``c[i,j] += a[i,k] * b[k,j]`` (row-major).

    Grid maps tile the ``M x N`` output by ``tile_m x tile_n``; the two inner
    sequential loops walk one tile and the sequential ``Reduce`` accumulates over
    ``K``. Returns the ``_i1`` tile loop -- the outermost loop of the MMA block
    that ``GpuMmaEinsumTransform`` is applied to. ``swap_operands`` writes the
    product as ``b * a`` so B is the einsum's first input.
    """
    dev = Pointer(HALF, StorageType.AMD_Generic())

    # Grid: X_GRID over columns (_j1_tile0), Y_GRID over rows (_i1_tile0).
    builder.begin_map(
        "_j1_tile0",
        "0",
        str(N),
        str(tile_n),
        ScheduleType.rocm_offload(TargetLevel.X_GRID, _ceil_div(N, tile_n)),
    )
    builder.begin_map(
        "_i1_tile0",
        "0",
        str(M),
        str(tile_m),
        ScheduleType.rocm_offload(TargetLevel.Y_GRID, _ceil_div(M, tile_m)),
    )

    # Sequential tile loops + K reduction.
    loop_i1 = builder.begin_for("_i1", "_i1_tile0", f"_i1_tile0 + {tile_m}", "1")
    builder.begin_for("_j1", "_j1_tile0", f"_j1_tile0 + {tile_n}", "1")
    builder.begin_reduce("_k0", "0", str(K), "1", [("add", c)])

    blk = builder.add_block()
    a_in = builder.add_access(blk, a)
    b_in = builder.add_access(blk, b)
    c_in = builder.add_access(blk, c)
    c_out = builder.add_access(blk, c)
    fma = builder.add_tasklet(
        blk, TaskletCode.fp_fma, ["_in1", "_in2", "_in3"], ["_out"]
    )
    builder.add_memlet(
        blk, a_in, "", fma, "_in2" if swap_operands else "_in1", f"{K}*_i1 + _k0", dev
    )
    builder.add_memlet(
        blk, b_in, "", fma, "_in1" if swap_operands else "_in2", f"_j1 + {N}*_k0", dev
    )
    builder.add_memlet(blk, c_in, "", fma, "_in3", f"{N}*_i1 + _j1", dev)
    builder.add_memlet(blk, fma, "_out", c_out, "", f"{N}*_i1 + _j1", dev)

    builder.end_reduce()
    builder.end_for()
    builder.end_for()
    builder.end_map()
    builder.end_map()

    return loop_i1


def _build_offloaded_einsum_mma(M, N, K, tile_m, tile_n):
    """Device-only matmul nest (kernel arguments, no host transfers)."""
    builder = StructuredSDFGBuilder("test_mma_einsum")
    dev = Pointer(HALF, StorageType.AMD_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    builder.add_container("A", dev, is_argument=True)
    builder.add_container("B", dev, is_argument=True)
    builder.add_container("C", dev, is_argument=True)
    for nm in ("_j1_tile0", "_i1_tile0", "_i1", "_j1", "_k0"):
        builder.add_container(nm, i32)

    loop_i1 = _add_einsum_mma_nest(builder, M, N, K, tile_m, tile_n, "A", "B", "C")
    return builder, loop_i1


ARCHES = ["gfx1201", "gfx90a"]

# (m_blocks, n_blocks) tile shapes the 16-wide MMA supports: 1x1, 2x2, 4x4.
ALIGNED = [
    (1024, 1024, 1024, 16, 16),
    (1024, 1024, 1024, 32, 32),
    (1024, 1024, 1024, 64, 64),
]


@pytest.mark.parametrize("arch_name", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n",
    ALIGNED,
    ids=[f"{m}x{n}x{k}_{tm}x{tn}" for (m, n, k, tm, tn) in ALIGNED],
)
def test_mma_einsum_expand_applies(arch_name, M, N, K, tile_m, tile_n):
    arch = RocmArch.get_from_name(arch_name)
    builder, loop_i1 = _build_offloaded_einsum_mma(M, N, K, tile_m, tile_n)
    am = AnalysisManager(builder)
    xform = GpuMmaEinsumTransform(loop_i1, arch)

    assert xform.can_be_applied(
        builder, am
    ), f"aligned {tile_m}x{tile_n} einsum tile should expand on {arch_name}"
    xform.apply(builder, am)
    assert xform.matched

    sdfg = builder.move()
    sdfg.validate()


# ---------------------------------------------------------------------------
# Execution: only on the matching GPU (DOCC_ROCM_ARCH names the arch under test).
# ---------------------------------------------------------------------------


def _build_executable_einsum_mma(M, N, K, tile_m, tile_n, swap_operands=False):
    """Full runnable matmul: host args, device buffers, H2D/D2H transfers, kernel."""
    builder = StructuredSDFGBuilder("test_mma_einsum_exec")
    host = Pointer(HALF)
    dev = Pointer(HALF, StorageType.AMD_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    builder.add_container("A", host, is_argument=True)
    builder.add_container("B", host, is_argument=True)
    builder.add_container("C", host, is_argument=True)
    builder.add_container("dA", dev)
    builder.add_container("dB", dev)
    builder.add_container("dC", dev)
    for nm in ("_j1_tile0", "_i1_tile0", "_i1", "_j1", "_k0"):
        builder.add_container(nm, i32)

    def off(hc, dc, direction, lifecycle, elems):
        builder.add_rocm_offloading_block(
            hc, dc, direction, lifecycle, dev, f"{elems} * {HALF_BYTES}"
        )

    off("A", "dA", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * K)
    off("B", "dB", DataTransferDirection.H2D, BufferLifecycle.ALLOC, K * N)
    off("C", "dC", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * N)

    loop_i1 = _add_einsum_mma_nest(
        builder, M, N, K, tile_m, tile_n, "dA", "dB", "dC", swap_operands
    )

    off("C", "dC", DataTransferDirection.D2H, BufferLifecycle.FREE, M * N)
    off("dA", "dA", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)
    off("dB", "dB", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)

    return builder, loop_i1


EXEC_CASES = [
    (32, 32, 32, 16, 16),
    (64, 64, 32, 16, 16),
    (32, 32, 32, 32, 32),
]


def _assert_matmul_close(C, ref):
    # Relative L2: fp16 output rounding is ~5e-4; a wrong or transposed product is O(1).
    rel = np.linalg.norm(C.astype(np.float32) - ref) / np.linalg.norm(ref)
    assert rel < 1e-2, f"relative L2 error {rel}"


@pytest.mark.rocm()
@pytest.mark.parametrize("swap_operands", [False, True], ids=["ab", "ba"])
@pytest.mark.parametrize("arch_name", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n",
    EXEC_CASES,
    ids=[f"{m}x{n}x{k}_{tm}x{tn}" for (m, n, k, tm, tn) in EXEC_CASES],
)
def test_mma_einsum_expand_executes(arch_name, M, N, K, tile_m, tile_n, swap_operands):
    if RocmArch.current_name() != arch_name:
        pytest.skip(f"DOCC_ROCM_ARCH ({RocmArch.current_name()}) != {arch_name}")
    arch = RocmArch.get_current()

    output_dir = (
        PYTEST_OUTPUT_DIR
        / f"mma_einsum_{arch_name}_{M}x{N}x{K}_{tile_m}x{tile_n}_{swap_operands}_executes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    builder, loop_i1 = _build_executable_einsum_mma(
        M, N, K, tile_m, tile_n, swap_operands
    )
    builder.dump(output_dir, "init", True, True)

    am = AnalysisManager(builder)
    xform = GpuMmaEinsumTransform(loop_i1, arch)
    assert xform.can_be_applied(builder, am)
    xform.apply(builder, am)
    assert xform.matched
    sdfg = builder.move()

    sdfg.dump(output_dir, "expanded", True, True)
    sdfg.validate()

    lib_path = sdfg._compile(str(output_dir), "rocm")
    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))

    ref = A.astype(np.float32) @ B.astype(np.float32)
    _assert_matmul_close(C, ref)


# ---------------------------------------------------------------------------
# LocalStorage: stage A/B into LDS after the MMA einsum expansion (agent_rocm Step 5).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch_name", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n",
    ALIGNED,
    ids=[f"{m}x{n}x{k}_{tm}x{tn}" for (m, n, k, tm, tn) in ALIGNED],
)
def test_mma_einsum_local_storage_applies(arch_name, M, N, K, tile_m, tile_n):
    arch = RocmArch.get_from_name(arch_name)
    builder, loop_i1 = _build_offloaded_einsum_mma(M, N, K, tile_m, tile_n)

    am = AnalysisManager(builder)
    xform = GpuMmaEinsumTransform(loop_i1, arch)
    assert xform.can_be_applied(builder, am)
    xform.apply(builder, am)
    assert xform.matched

    _localize_operands(builder, "A", "B")

    sdfg = builder.move()
    sdfg.validate()


@pytest.mark.rocm()
@pytest.mark.parametrize("arch_name", ARCHES)
@pytest.mark.parametrize(
    "M,N,K,tile_m,tile_n",
    EXEC_CASES,
    ids=[f"{m}x{n}x{k}_{tm}x{tn}" for (m, n, k, tm, tn) in EXEC_CASES],
)
def test_mma_einsum_local_storage_executes(arch_name, M, N, K, tile_m, tile_n):
    if RocmArch.current_name() != arch_name:
        pytest.skip(f"DOCC_ROCM_ARCH ({RocmArch.current_name()}) != {arch_name}")
    arch = RocmArch.get_current()

    output_dir = (
        PYTEST_OUTPUT_DIR
        / f"mma_einsum_ls_{arch_name}_{M}x{N}x{K}_{tile_m}x{tile_n}_executes"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    builder, loop_i1 = _build_executable_einsum_mma(M, N, K, tile_m, tile_n)

    am = AnalysisManager(builder)
    xform = GpuMmaEinsumTransform(loop_i1, arch)
    assert xform.can_be_applied(builder, am)
    xform.apply(builder, am)
    assert xform.matched

    builder.dump(output_dir, "expanded", True, True)

    _localize_operands(builder, "dA", "dB")

    builder.dump(output_dir, "localized", True, True)
    sdfg = builder.move()
    sdfg.validate()

    lib_path = sdfg._compile(str(output_dir), "rocm")
    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))

    ref = A.astype(np.float32) @ B.astype(np.float32)
    _assert_matmul_close(C, ref)
