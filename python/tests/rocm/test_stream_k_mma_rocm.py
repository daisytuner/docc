"""ROCm e2e: aligned Stream-K split of an MMA matmul with a last-arriver fragment merge.

Builds the raw offloaded nest ``C[i, j] += A[i, k] * B[j, k]`` in bfloat16 (B stored K-major, as the down
projection of an MLP), lowers it to rocWMMA with ``GpuMmaEinsumTransform``, strip-mines the K sweep into
panels and applies ``StreamK`` with ``num_blocks = tiles * S`` before staging both operands in LDS and
software pipelining the panel loop whenever a block owns at least two panels -- the order the ROCm benchmark
agents use.

Each split count S is checked against an fp64 reference, including a non-zero initial C that must be added
exactly once, and over repeated launches, which must reuse the static arrival counters and reproduce the
result bit for bit.
"""

from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    Block,
    BufferLifecycle,
    DataTransferDirection,
    GpuMmaEinsumTransform,
    IfElse,
    LocalStorage,
    LoopTiling,
    Pointer,
    PrimitiveType,
    RocmArch,
    Scalar,
    ScheduleType,
    Sequence,
    SoftwarePipelining,
    StorageType,
    StreamK,
    StructuredLoop,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

pytestmark = pytest.mark.rocm()

PYTEST_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "pytestOutput"

BF16 = Scalar(PrimitiveType.BFloat)
BF16_BYTES = 2

M, N, K = 16, 128, 640
TILE_M, TILE_N = 16, 32
PANEL_BLOCKS = 4
TILES = (M // TILE_M) * (N // TILE_N)
PANELS = K // 16 // PANEL_BLOCKS


def _find_access(seq, container):
    for i in range(len(seq)):
        child = seq[i]
        if isinstance(child, Block):
            for node in list(child.dataflow.reads) + list(child.dataflow.writes):
                if node.data == container:
                    return node
        elif isinstance(child, StructuredLoop):
            found = _find_access(child.body, container)
            if found is not None:
                return found
        elif isinstance(child, Sequence):
            found = _find_access(child, container)
            if found is not None:
                return found
        elif isinstance(child, IfElse):
            for case_idx in range(child.size):
                found = _find_access(child.case(case_idx), container)
                if found is not None:
                    return found
    return None


def _loop(builder, indvar):
    loop = AnalysisManager(builder).loop_analysis().find_loop_by_indvar(indvar)
    assert loop is not None, indvar
    return loop


def _apply(builder, xform):
    am = AnalysisManager(builder)
    assert xform.can_be_applied(builder, am), repr(xform)
    xform.apply(builder, am)


def _build():
    builder = StructuredSDFGBuilder("stream_k_mma")
    host = Pointer(BF16)
    dev = Pointer(BF16, StorageType.AMD_Generic())
    i32 = Scalar(PrimitiveType.Int32)
    for name in ("A", "B", "C"):
        builder.add_container(name, host, is_argument=True)
        builder.add_container("d" + name, dev)
    for name in ("_j1_tile0", "_i1_tile0", "_i1", "_j1", "_k0"):
        builder.add_container(name, i32)

    def off(name, direction, lifecycle, elems):
        builder.add_rocm_offloading_block(
            name, "d" + name, direction, lifecycle, dev, f"{elems} * {BF16_BYTES}"
        )

    off("A", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * K)
    off("B", DataTransferDirection.H2D, BufferLifecycle.ALLOC, N * K)
    off("C", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * N)

    builder.begin_map(
        "_j1_tile0",
        "0",
        str(N),
        str(TILE_N),
        ScheduleType.rocm_offload(TargetLevel.X_GRID, N // TILE_N),
    )
    builder.begin_map(
        "_i1_tile0",
        "0",
        str(M),
        str(TILE_M),
        ScheduleType.rocm_offload(TargetLevel.Y_GRID, M // TILE_M),
    )
    loop_i1 = builder.begin_for("_i1", "_i1_tile0", f"_i1_tile0 + {TILE_M}", "1")
    builder.begin_for("_j1", "_j1_tile0", f"_j1_tile0 + {TILE_N}", "1")
    builder.begin_reduce("_k0", "0", str(K), "1", [("add", "dC")])
    blk = builder.add_block()
    a_in = builder.add_access(blk, "dA")
    b_in = builder.add_access(blk, "dB")
    c_in = builder.add_access(blk, "dC")
    c_out = builder.add_access(blk, "dC")
    fma = builder.add_tasklet(
        blk, TaskletCode.fp_fma, ["_in1", "_in2", "_in3"], ["_out"]
    )
    builder.add_memlet(blk, a_in, "", fma, "_in1", f"{K}*_i1 + _k0", dev)
    builder.add_memlet(blk, b_in, "", fma, "_in2", f"{K}*_j1 + _k0", dev)
    builder.add_memlet(blk, c_in, "", fma, "_in3", f"{N}*_i1 + _j1", dev)
    builder.add_memlet(blk, fma, "_out", c_out, "", f"{N}*_i1 + _j1", dev)
    builder.end_reduce()
    builder.end_for()
    builder.end_for()
    builder.end_map()
    builder.end_map()

    off("C", DataTransferDirection.D2H, BufferLifecycle.FREE, M * N)
    off("A", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)
    off("B", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)
    return builder, loop_i1


def _schedule(builder, loop_i1, splits):
    _apply(builder, GpuMmaEinsumTransform(loop_i1, RocmArch.get_current()))
    _apply(builder, LoopTiling(_loop(builder, "tile_k0"), PANEL_BLOCKS, True))
    _apply(builder, StreamK(_loop(builder, "_j1_tile0"), TILES * splits))
    for container in ("dA", "dB"):
        panel = _loop(builder, "tile_k0")
        _apply(builder, LocalStorage(panel, _find_access(panel.body, container)))
    if PANELS // splits >= 2:
        _apply(
            builder,
            SoftwarePipelining(
                _loop(builder, "tile_k0_tile0"), stages=2, register_staged=True
            ),
        )


def _compile(splits):
    if RocmArch.current_name() not in ("gfx90a", "gfx942"):
        pytest.skip(f"no CDNA MMA device ({RocmArch.current_name()})")
    output_dir = PYTEST_OUTPUT_DIR / f"stream_k_mma_s{splits}"
    output_dir.mkdir(parents=True, exist_ok=True)
    builder, loop_i1 = _build()
    _schedule(builder, loop_i1, splits)
    sdfg = builder.move()
    sdfg.validate()
    return CompiledSDFG(sdfg._compile(str(output_dir), "rocm"), sdfg)


def _operands(seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(ml_dtypes.bfloat16)
    B = (rng.standard_normal((N, K)) / np.sqrt(K)).astype(ml_dtypes.bfloat16)
    C = rng.standard_normal((M, N)).astype(ml_dtypes.bfloat16)
    return A, B, C


def _run(compiled, A, B, C0):
    C = C0.copy()
    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))
    return C


def _rel_l2(C, ref):
    return np.linalg.norm(C.astype(np.float64) - ref) / np.linalg.norm(ref)


@pytest.mark.parametrize("splits", [1, 2, 5, 10])
def test_stream_k_mma_split_matches_reference(splits):
    assert PANELS % splits == 0
    compiled = _compile(splits)
    A, B, C0 = _operands(0)
    ref = A.astype(np.float64) @ B.astype(np.float64).T + C0.astype(np.float64)

    C = _run(compiled, A, B, C0)
    assert _rel_l2(C, ref) < 1e-2


@pytest.mark.parametrize("splits", [2, 5])
def test_stream_k_mma_split_is_reusable_and_deterministic(splits):
    compiled = _compile(splits)
    runs = []
    for seed in (1, 2, 1):
        A, B, C0 = _operands(seed)
        ref = A.astype(np.float64) @ B.astype(np.float64).T + C0.astype(np.float64)
        C = _run(compiled, A, B, C0)
        assert _rel_l2(C, ref) < 1e-2, f"seed {seed}"
        runs.append(C)
    assert np.array_equal(runs[0].view(np.uint16), runs[2].view(np.uint16))
