"""``RedundantZeroInitElimination`` on a GPU MMA matmul.

The naive MatMul expansion emits a zero-init nest ``C = 0`` (annotated ``zero_init``) followed by the
accumulation nest ``C += A @ B``. Once ``GpuMmaEinsumTransform`` lowers the accumulation nest to MMA, the init
nest is redundant: the transformation removes it and zero-fills the MMA's C fragments instead of loading C.

The structural tests need no device. The execution test runs on the GPU named by ``DOCC_ROCM_ARCH`` and
initializes C with NaNs, so any remaining read of C's previous contents would show up in the result.
"""

import json

import numpy as np
import pytest

from docc.compiler.compiled_sdfg import CompiledSDFG
from docc.sdfg import (
    AnalysisManager,
    BufferLifecycle,
    DataTransferDirection,
    GpuMmaEinsumTransform,
    Pointer,
    PrimitiveType,
    RedundantZeroInitElimination,
    RocmArch,
    Scalar,
    ScheduleType,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
)
from test_mma_einsum_transform import (
    ARCHES,
    HALF,
    HALF_BYTES,
    PYTEST_OUTPUT_DIR,
    _add_einsum_mma_nest,
    _add_offloading_block,
    _assert_matmul_close,
    _compile_backend,
    _dev_pointer,
    _iter_library_nodes,
    _offload_schedule,
)

M, N, K, TILE = 32, 32, 32, 16


def _add_zero_init_nest(builder, arch_name, c, annotate):
    """Offloaded init nest ``c[i, j] = 0`` as emitted by the naive MatMul expansion."""
    outer = _offload_schedule(arch_name, TargetLevel.X_GRID, M)
    outer.zero_init = annotate
    init = builder.begin_map("_iz", "0", str(M), "1", outer)
    builder.begin_map(
        "_jz", "0", str(N), "1", _offload_schedule(arch_name, TargetLevel.X_BLOCK, N)
    )
    blk = builder.add_block()
    zero = builder.add_constant(blk, "0.0", HALF)
    out = builder.add_access(blk, c)
    assign = builder.add_tasklet(blk, TaskletCode.assign, ["_in"], ["_out"])
    builder.add_memlet(blk, zero, "", assign, "_in", "", HALF)
    builder.add_memlet(
        blk, assign, "_out", out, "", f"{N}*_iz + _jz", _dev_pointer(arch_name)
    )
    builder.end_map()
    builder.end_map()
    return init


def _build(arch_name, annotate=True, executable=False):
    """``C = A @ B`` as an annotated zero-init nest followed by the einsum matmul nest."""
    builder = StructuredSDFGBuilder("test_redundant_zero_init")
    dev = _dev_pointer(arch_name)
    i32 = Scalar(PrimitiveType.Int32)
    names = ("dA", "dB", "dC") if executable else ("A", "B", "C")
    if executable:
        for host in ("A", "B", "C"):
            builder.add_container(host, Pointer(HALF), is_argument=True)
        for name in names:
            builder.add_container(name, dev)
    else:
        for name in names:
            builder.add_container(name, dev, is_argument=True)
    for nm in ("_iz", "_jz", "_j1_tile0", "_i1_tile0", "_i1", "_j1", "_k0"):
        builder.add_container(nm, i32)

    def off(hc, dc, direction, lifecycle, elems):
        _add_offloading_block(
            builder,
            arch_name,
            hc,
            dc,
            direction,
            lifecycle,
            dev,
            f"{elems} * {HALF_BYTES}",
        )

    if executable:
        off("A", "dA", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * K)
        off("B", "dB", DataTransferDirection.H2D, BufferLifecycle.ALLOC, K * N)
        # Transfer C's (NaN) contents: only the init nest may hide them.
        off("C", "dC", DataTransferDirection.H2D, BufferLifecycle.ALLOC, M * N)

    init = _add_zero_init_nest(builder, arch_name, names[2], annotate)
    loop_i1 = _add_einsum_mma_nest(builder, M, N, K, TILE, TILE, *names, arch_name)

    if executable:
        off("C", "dC", DataTransferDirection.D2H, BufferLifecycle.FREE, M * N)
        off("dA", "dA", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)
        off("dB", "dB", DataTransferDirection.NONE, BufferLifecycle.FREE, 0)
    return builder, init, loop_i1


def _lower_to_mma(builder, loop_i1, arch):
    am = AnalysisManager(builder)
    xform = GpuMmaEinsumTransform(loop_i1, arch)
    assert xform.can_be_applied(builder, am)
    xform.apply(builder, am)
    assert xform.matched


def _count_zero_inits(sdfg):
    """Number of loops over the init nest's induction variable ``_iz``."""

    def count(node):
        if isinstance(node, dict):
            own = int(node.get("type") == "map" and node.get("indvar") == "_iz")
            return own + sum(count(v) for v in node.values())
        if isinstance(node, list):
            return sum(count(v) for v in node)
        return 0

    return count(json.loads(sdfg.to_json()))


def _c_fragment_ops(sdfg):
    codes = [node.code for node in _iter_library_nodes(sdfg.root)]
    return codes.count("gpu::GpuMmaFill"), codes.count("gpu::MmaFragmentLoad")


def test_schedule_type_zero_init_property():
    schedule = ScheduleType.sequential()
    assert not schedule.zero_init
    schedule.zero_init = True
    assert schedule.zero_init
    assert schedule.properties["init"] == "zero"


@pytest.mark.parametrize("arch_name", ARCHES)
def test_redundant_zero_init_elimination_applies(arch_name):
    builder, init, loop_i1 = _build(arch_name)
    _lower_to_mma(builder, loop_i1, RocmArch.get_from_name(arch_name))

    am = AnalysisManager(builder)
    t = RedundantZeroInitElimination(init)
    assert "RedundantZeroInitElimination" in repr(t)
    assert t.name == "RedundantZeroInitElimination"
    assert t.can_be_applied(builder, am)
    t.apply(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    assert _count_zero_inits(sdfg) == 0
    fills, loads = _c_fragment_ops(sdfg)
    # Accumulator fill + C fill per wave block; only the A and B fragments are still loaded.
    assert fills == 2
    assert loads == 2


@pytest.mark.parametrize("arch_name", ARCHES)
def test_redundant_zero_init_elimination_requires_annotation(arch_name):
    builder, init, loop_i1 = _build(arch_name, annotate=False)
    _lower_to_mma(builder, loop_i1, RocmArch.get_from_name(arch_name))
    am = AnalysisManager(builder)
    assert not RedundantZeroInitElimination(init).can_be_applied(builder, am)


@pytest.mark.parametrize("arch_name", ARCHES)
def test_redundant_zero_init_elimination_requires_mma(arch_name):
    builder, init, _ = _build(arch_name)
    am = AnalysisManager(builder)
    assert not RedundantZeroInitElimination(init).can_be_applied(builder, am)


@pytest.mark.rocm()
@pytest.mark.parametrize("arch_name", ARCHES)
def test_redundant_zero_init_elimination_executes(arch_name):
    if RocmArch.current_name() != arch_name:
        pytest.skip(f"DOCC_ROCM_ARCH ({RocmArch.current_name()}) != {arch_name}")
    output_dir = PYTEST_OUTPUT_DIR / f"redundant_zero_init_{arch_name}_executes"
    output_dir.mkdir(parents=True, exist_ok=True)

    builder, init, loop_i1 = _build(arch_name, executable=True)
    _lower_to_mma(builder, loop_i1, RocmArch.get_current())
    am = AnalysisManager(builder)
    t = RedundantZeroInitElimination(init)
    assert t.can_be_applied(builder, am)
    t.apply(builder, am)

    sdfg = builder.move()
    sdfg.validate()
    assert _count_zero_inits(sdfg) == 0
    sdfg.dump(output_dir, "eliminated", True, True)
    compiled = CompiledSDFG(
        sdfg._compile(str(output_dir), _compile_backend(arch_name)), sdfg
    )

    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.1).astype(np.float16)
    C = np.full((M, N), np.nan, dtype=np.float16)
    compiled(A.reshape(-1), B.reshape(-1), C.reshape(-1))

    _assert_matmul_close(C, A.astype(np.float32) @ B.astype(np.float32))
