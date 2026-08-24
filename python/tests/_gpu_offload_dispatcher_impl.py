"""
Backend-agnostic integration tests for the GPU offload dispatchers.

These tests verify the low-level GPU offload code generators (the shared
``GPUOffloadMapDispatcher`` / ``GPUOffloadReduceDispatcher`` and their CUDA and
ROCm specialisations) in isolation, WITHOUT running the scheduler or any
transformations.

Each test builds an already-offloaded SDFG entirely through the
``StructuredSDFGBuilder`` API -- host argument containers, device transients
(``NV_Generic`` for CUDA / ``AMD_Generic`` for ROCm), explicit data-offloading
blocks (alloc / H2D / D2H / free) and offloaded ``Map`` / ``Reduce`` nodes
carrying a ``backend.schedule(target_level, parallel_size)`` schedule -- then
compiles it for the backend, runs it on the GPU and compares against NumPy.

Everything backend-specific is funnelled through a :class:`GpuBackend`
descriptor (device storage type, schedule factory, offloading-block adder,
compile target, generated-source glob and warp/wavefront size), so the very
same builders and scenarios drive both the CUDA (warp size 32) and ROCm
(wavefront size 64) suites.  The thin ``cuda`` / ``rocm`` suites bind a backend
via :func:`register`.

The parallel size is deliberately varied between:
  * exact-fit  -- ``parallel_size`` divides the iteration count, and
  * ragged     -- ``parallel_size`` does not divide it (non-power-of-two),
so the grid-stride coverage loop and its boundary guard are both exercised.
"""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from docc.sdfg import (
    BufferLifecycle,
    CMathFunction,
    DataTransferDirection,
    Pointer,
    PrimitiveType,
    Scalar,
    ScheduleType,
    StorageType,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

FLOAT_BYTES = 4


# ---------------------------------------------------------------------------
# Backend descriptor: the single seam between the shared builders/scenarios and
# the concrete CUDA / ROCm targets.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GpuBackend:
    """Everything that differs between the CUDA and ROCm offload backends.

    ``warp_size``      -- warp (CUDA, 32) / wavefront (ROCm, 64) width; drives
                          the warp scenarios and the nesting legality check.
    ``target``         -- compile target string passed to ``sdfg._compile``.
    ``source_glob``    -- glob matching the generated device sources, asserted
                          to contain the offload coverage loop.
    ``_storage``       -- device ``StorageType`` factory (NV_Generic/AMD_Generic).
    ``_schedule``      -- ``ScheduleType`` offload-schedule factory.
    ``_offload_method``-- name of the builder's data-offloading-block method.
    """

    name: str
    warp_size: int
    target: str
    source_glob: str
    _storage: Callable[[], object]
    _schedule: Callable[[object, int], object]
    _offload_method: str

    def storage(self):
        return self._storage()

    def schedule(self, target_level, parallel_size):
        return self._schedule(target_level, parallel_size)

    def offload(self, builder, *args, **kwargs):
        return getattr(builder, self._offload_method)(*args, **kwargs)


CUDA_BACKEND = GpuBackend(
    name="cuda",
    warp_size=32,
    target="cuda",
    source_glob="*.cu",
    _storage=StorageType.NV_Generic,
    _schedule=ScheduleType.cuda_offload,
    _offload_method="add_cuda_offloading_block",
)

ROCM_BACKEND = GpuBackend(
    name="rocm",
    warp_size=64,
    target="rocm",
    source_glob="*rocm.cpp",
    _storage=StorageType.AMD_Generic,
    _schedule=ScheduleType.rocm_offload,
    _offload_method="add_rocm_offloading_block",
)


# Body node used inside the kernel for each supported element-wise / reduction op.
# add/mul map to plain float tasklets; float min/max have NO tasklet code and are
# emitted as fmin/fmax cmath library nodes instead (which expose the very same
# _in1/_in2 -> _out connectors, so every builder wires memlets identically).
_TASKLET = {
    "add": TaskletCode.fp_add,
    "mul": TaskletCode.fp_mul,
}
_CMATH = {
    "min": CMathFunction.fmin,
    "max": CMathFunction.fmax,
}


def _add_body_op(builder, block, op):
    """Add the binary body node computing ``_out = _in1 <op> _in2``.

    Returns a node exposing ``_in1``/``_in2`` inputs and an ``_out`` output,
    regardless of whether ``op`` is realised as a tasklet (add/mul) or as an
    fmin/fmax cmath library node (float min/max, which have no tasklet code).
    """
    if op in _TASKLET:
        return builder.add_tasklet(block, _TASKLET[op], ["_in1", "_in2"], ["_out"])
    return builder.add_cmath(block, _CMATH[op], PrimitiveType.Float)


# All seven offload target levels, addressable by short name for scenario tables.
LEVELS = {
    "xg": TargetLevel.X_GRID,
    "yg": TargetLevel.Y_GRID,
    "zg": TargetLevel.Z_GRID,
    "xb": TargetLevel.X_BLOCK,
    "yb": TargetLevel.Y_BLOCK,
    "zb": TargetLevel.Z_BLOCK,
    "w": TargetLevel.WARP,
}


def _strides(counts):
    """Row-major strides for a flattened multi-dimensional index space."""
    return [math.prod(counts[k + 1 :]) for k in range(len(counts))]


def _flat_expr(names, counts):
    """Symbolic flat (row-major) index expression over the given loop variables."""
    strides = _strides(counts)
    terms = [name if s == 1 else f"{name}*{s}" for name, s in zip(names, strides)]
    return " + ".join(terms) if terms else "0"


def _identity(op):
    return {"add": 0.0, "mul": 1.0, "min": np.inf, "max": -np.inf}[op]


# Grid dimension that each block dimension must be nested inside.
_GRID_OF = {
    TargetLevel.X_BLOCK: TargetLevel.X_GRID,
    TargetLevel.Y_BLOCK: TargetLevel.Y_GRID,
    TargetLevel.Z_BLOCK: TargetLevel.Z_GRID,
}
_BLOCK_LEVELS = set(_GRID_OF)


def _validate_nest(specs, warp_size):
    """Assert an outer->inner level nest obeys the GPU nesting rules.

    ``specs`` is a list of ``(TargetLevel, parallel_size)`` pairs, outer to inner.

    * Each block dimension must be nested inside its corresponding grid
      dimension (X_BLOCK<-X_GRID, Y_BLOCK<-Y_GRID, Z_BLOCK<-Z_GRID).
    * Nothing may be nested inside a WARP (it must be innermost).
    * A WARP must be nested inside an X_BLOCK.
    * The product of all ancestor block dimensions of a WARP must be at least
      one full warp (>= ``warp_size``), since the warp's lanes are drawn from
      the enclosing block's threads.
    """
    levels = [lvl for (lvl, _ps) in specs]
    for pos, lvl in enumerate(levels):
        if lvl == TargetLevel.WARP and pos != len(levels) - 1:
            raise AssertionError("nothing may be nested within WARP")
    seen = []
    for lvl in levels:
        if lvl in _BLOCK_LEVELS and _GRID_OF[lvl] not in seen:
            raise AssertionError(f"{lvl} must be nested within {_GRID_OF[lvl]}")
        seen.append(lvl)
    if TargetLevel.WARP in levels:
        w = levels.index(TargetLevel.WARP)
        if TargetLevel.X_BLOCK not in levels[:w]:
            raise AssertionError("WARP must be nested within an X_BLOCK")
        block_product = math.prod(ps for (lvl, ps) in specs[:w] if lvl in _BLOCK_LEVELS)
        if block_product < warp_size:
            raise AssertionError(
                f"WARP requires ancestor block-dim product >= {warp_size}, "
                f"got {block_product}"
            )


# ---------------------------------------------------------------------------
# NumPy references
# ---------------------------------------------------------------------------
def numpy_elementwise(a, b, op):
    if op == "add":
        return a + b
    if op == "mul":
        return a * b
    if op == "min":
        return np.minimum(a, b)
    if op == "max":
        return np.maximum(a, b)
    raise ValueError(op)


def numpy_reduce(a, op):
    if op == "add":
        return a.sum()
    if op == "mul":
        return a.prod()
    if op == "min":
        return a.min()
    if op == "max":
        return a.max()
    raise ValueError(op)


# ---------------------------------------------------------------------------
# SDFG builders
# ---------------------------------------------------------------------------
def build_offloaded_elementwise(backend, n, op, target_level, parallel_size):
    """Build ``C[i] = A[i] <op> B[i]`` as a single offloaded Map.

    The Map is the outermost (and only) offloaded loop, so the dispatcher
    emits the ``__global__`` kernel plus its ``<<<>>>`` launch directly.
    """
    builder = StructuredSDFGBuilder(f"offload_map_{op}")

    f = Scalar(PrimitiveType.Float)
    host_ptr = Pointer(f)
    dev_ptr = Pointer(f, backend.storage())
    i32 = Scalar(PrimitiveType.Int32)

    for name in ("A", "B", "C"):
        builder.add_container(name, host_ptr, is_argument=True)
    for name in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        builder.add_container(name, dev_ptr, is_argument=False)
    builder.add_container("i", i32, is_argument=False)

    nbytes = f"{n} * {FLOAT_BYTES}"

    for name in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        backend.offload(
            builder,
            name,
            name,
            DataTransferDirection.NONE,
            BufferLifecycle.ALLOC,
            dev_ptr,
            nbytes,
        )
    for host, dev in (("A", "__daisy_dev_A"), ("B", "__daisy_dev_B")):
        backend.offload(
            builder,
            host,
            dev,
            DataTransferDirection.H2D,
            BufferLifecycle.NO_CHANGE,
            dev_ptr,
            nbytes,
        )

    builder.begin_map(
        "i", "0", str(n), "1", backend.schedule(target_level, parallel_size)
    )
    blk = builder.add_block()
    a = builder.add_access(blk, "__daisy_dev_A")
    b = builder.add_access(blk, "__daisy_dev_B")
    c = builder.add_access(blk, "__daisy_dev_C")
    t = _add_body_op(builder, blk, op)
    builder.add_memlet(blk, a, "", t, "_in1", "i")
    builder.add_memlet(blk, b, "", t, "_in2", "i")
    builder.add_memlet(blk, t, "_out", c, "", "i")
    builder.end_map()

    backend.offload(
        builder,
        "C",
        "__daisy_dev_C",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes,
    )
    for name in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        backend.offload(
            builder,
            name,
            name,
            DataTransferDirection.NONE,
            BufferLifecycle.FREE,
            dev_ptr,
            "0",
        )

    return builder.move()


def build_offloaded_reduction(backend, n, op, target_level, parallel_size):
    """Build ``acc[0] = <op>_j A[j]`` as a single offloaded Reduce.

    The device accumulator is zero-/identity-initialised on the host and
    copied H2D before the kernel, since a grid-level reduction combines into
    the global slot with an atomic / CAS.
    """
    builder = StructuredSDFGBuilder(f"offload_reduce_{op}")

    f = Scalar(PrimitiveType.Float)
    host_ptr = Pointer(f)
    dev_ptr = Pointer(f, backend.storage())
    i32 = Scalar(PrimitiveType.Int32)

    for name in ("A", "acc"):
        builder.add_container(name, host_ptr, is_argument=True)
    for name in ("__daisy_dev_A", "__daisy_dev_acc"):
        builder.add_container(name, dev_ptr, is_argument=False)
    builder.add_container("j", i32, is_argument=False)

    nbytes_a = f"{n} * {FLOAT_BYTES}"
    nbytes_acc = f"{FLOAT_BYTES}"

    backend.offload(
        builder,
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes_a,
    )
    backend.offload(
        builder,
        "__daisy_dev_acc",
        "__daisy_dev_acc",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes_acc,
    )
    backend.offload(
        builder,
        "A",
        "__daisy_dev_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_a,
    )
    backend.offload(
        builder,
        "acc",
        "__daisy_dev_acc",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_acc,
    )

    builder.begin_reduce(
        "j",
        "0",
        str(n),
        "1",
        [(op, "__daisy_dev_acc")],
        backend.schedule(target_level, parallel_size),
    )
    blk = builder.add_block()
    a = builder.add_access(blk, "__daisy_dev_A")
    acc_in = builder.add_access(blk, "__daisy_dev_acc")
    acc_out = builder.add_access(blk, "__daisy_dev_acc")
    t = _add_body_op(builder, blk, op)
    builder.add_memlet(blk, acc_in, "", t, "_in1", "0")
    builder.add_memlet(blk, a, "", t, "_in2", "j")
    builder.add_memlet(blk, t, "_out", acc_out, "", "0")
    builder.end_reduce()

    backend.offload(
        builder,
        "acc",
        "__daisy_dev_acc",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_acc,
    )
    for name in ("__daisy_dev_A", "__daisy_dev_acc"):
        backend.offload(
            builder,
            name,
            name,
            DataTransferDirection.NONE,
            BufferLifecycle.FREE,
            dev_ptr,
            "0",
        )

    return builder.move()


def _compile(backend, sdfg, output_dir: Path):
    sdfg.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), backend.target)

    # The offloaded map/reduce must lower through the offload dispatcher
    # (grid-stride coverage loop + kernel launch), not a fallback path.
    sources = list(output_dir.rglob(backend.source_glob))
    assert sources, f"no {backend.name} device source emitted"
    generated = "\n".join(p.read_text() for p in sources)
    assert (
        "__daisy_gpu_coverage_loop_" in generated
    ), "offload dispatcher coverage loop missing from generated device source"

    return CompiledSDFG(lib_path, sdfg)


# ---------------------------------------------------------------------------
# MAP dispatcher tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "target_level",
    [TargetLevel.X_GRID, TargetLevel.X_BLOCK],
    ids=["grid", "block"],
)
@pytest.mark.parametrize("op", ["add", "mul", "min", "max"])
@pytest.mark.parametrize(
    "n,parallel_size",
    [
        (1024, 128),  # exact-fit: parallel_size divides n
        (1000, 250),  # exact-fit
        (1000, 256),  # ragged: 256 does not divide 1000
        (100, 32),  # ragged, non-power-of-two remainder
        (100, 128),  # parallel_size > n (single pass, boundary guard)
        (1, 32),  # degenerate size-1 iteration space
    ],
    ids=["1024x128", "1000x250", "1000x256", "100x32", "100x128", "size1"],
)
def test_offload_map(target_level, op, n, parallel_size, tmp_path):
    sdfg = build_offloaded_elementwise(n, op, target_level, parallel_size)
    compiled = _compile(sdfg, tmp_path / "map")

    rng = np.random.default_rng(0)
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)

    compiled(a, b, c)

    np.testing.assert_allclose(c, numpy_elementwise(a, b, op), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# REDUCE dispatcher tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("op", ["add", "mul", "min", "max"])
@pytest.mark.parametrize(
    "n,parallel_size",
    [
        (1000, 256),  # ragged
        (1000, 250),  # exact-fit
        (100, 32),  # ragged, non-power-of-two remainder
        (1, 32),  # degenerate size-1 reduction axis
    ],
    ids=["1000x256", "1000x250", "100x32", "size1"],
)
def test_offload_reduce(op, n, parallel_size, tmp_path):
    sdfg = build_offloaded_reduction(n, op, TargetLevel.X_GRID, parallel_size)
    compiled = _compile(sdfg, tmp_path / "reduce")

    a = _rng_input(op, n, 1)
    acc = np.full(1, _identity(op), dtype=np.float32)

    compiled(a, acc)

    ref = numpy_reduce(a, op)
    np.testing.assert_allclose(acc[0], ref, rtol=1e-4, atol=1e-4)


# ===========================================================================
# Multi-level scenarios: arbitrary depth and combinations of the seven levels.
#
# These builders are written purely from the intended dataflow semantics; they
# do NOT mirror the dispatcher's code generation.  Each scenario is a semantic
# identity (a plain element-wise op, or a reduction) realised through a
# particular nesting of offload target levels -- so a wrong result pins the
# blame on the level's code generator, not on the test.
# ===========================================================================
def _rng_input(op, size, seed):
    rng = np.random.default_rng(seed)
    if op == "mul":
        # products stay well-conditioned when every factor sits close to 1.
        return (1.0 + 0.01 * rng.standard_normal(size)).astype(np.float32)
    return rng.standard_normal(size).astype(np.float32)


def _declare_common(backend, builder, host_args, device_names):
    f = Scalar(PrimitiveType.Float)
    host_ptr = Pointer(f)
    dev_ptr = Pointer(f, backend.storage())
    for name in host_args:
        builder.add_container(name, host_ptr, is_argument=True)
    for name in device_names:
        builder.add_container(name, dev_ptr, is_argument=False)
    return dev_ptr


# ---------------------------------------------------------------------------
# MAP nests: C[flat] = A[flat] <op> B[flat] over an arbitrary level nest.
# ---------------------------------------------------------------------------
def build_map_nest(backend, specs, op):
    """``specs`` = list of (TargetLevel, count, parallel_size) from outer to inner."""
    _validate_nest([(lvl, ps) for (lvl, _c, ps) in specs], backend.warp_size)
    counts = [c for (_, c, _) in specs]
    n = math.prod(counts)
    names = [f"i{k}" for k in range(len(specs))]
    flat = _flat_expr(names, counts)

    builder = StructuredSDFGBuilder(f"map_nest_{op}")
    dev_ptr = _declare_common(
        backend,
        builder,
        ("A", "B", "C"),
        ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"),
    )
    i32 = Scalar(PrimitiveType.Int32)
    for name in names:
        builder.add_container(name, i32, is_argument=False)

    nbytes = f"{n} * {FLOAT_BYTES}"
    for name in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        backend.offload(
            builder,
            name,
            name,
            DataTransferDirection.NONE,
            BufferLifecycle.ALLOC,
            dev_ptr,
            nbytes,
        )
    for host, dev in (("A", "__daisy_dev_A"), ("B", "__daisy_dev_B")):
        backend.offload(
            builder,
            host,
            dev,
            DataTransferDirection.H2D,
            BufferLifecycle.NO_CHANGE,
            dev_ptr,
            nbytes,
        )

    for name, (level, count, psize) in zip(names, specs):
        builder.begin_map(name, "0", str(count), "1", backend.schedule(level, psize))

    blk = builder.add_block()
    a = builder.add_access(blk, "__daisy_dev_A")
    b = builder.add_access(blk, "__daisy_dev_B")
    c = builder.add_access(blk, "__daisy_dev_C")
    t = _add_body_op(builder, blk, op)
    builder.add_memlet(blk, a, "", t, "_in1", flat)
    builder.add_memlet(blk, b, "", t, "_in2", flat)
    builder.add_memlet(blk, t, "_out", c, "", flat)

    for _ in specs:
        builder.end_map()

    backend.offload(
        builder,
        "C",
        "__daisy_dev_C",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes,
    )
    for name in ("__daisy_dev_A", "__daisy_dev_B", "__daisy_dev_C"):
        backend.offload(
            builder,
            name,
            name,
            DataTransferDirection.NONE,
            BufferLifecycle.FREE,
            dev_ptr,
            "0",
        )

    return builder.move(), n


# ---------------------------------------------------------------------------
# REDUCE nests: acc[row] = <op>_col A[row, col].  The reduction axis and/or the
# parallel (row) axis can each be spread over an arbitrary level nest; when the
# reduction spans several levels they all target the SAME accumulator (the
# "same variable on different levels" case).
# ---------------------------------------------------------------------------
def build_reduce_nest(backend, map_specs, reduce_specs, op):
    _validate_nest(
        [(lvl, ps) for (lvl, _c, ps) in map_specs + reduce_specs], backend.warp_size
    )
    map_counts = [c for (_, c, _) in map_specs]
    red_counts = [c for (_, c, _) in reduce_specs]
    rows = math.prod(map_counts)
    cols = math.prod(red_counts)
    n = rows * cols

    m_names = [f"m{k}" for k in range(len(map_specs))]
    r_names = [f"r{k}" for k in range(len(reduce_specs))]
    row_flat = _flat_expr(m_names, map_counts)
    red_flat = _flat_expr(r_names, red_counts)
    in_index = f"({row_flat}) * {cols} + ({red_flat})"

    builder = StructuredSDFGBuilder(f"reduce_nest_{op}")
    dev_ptr = _declare_common(
        backend, builder, ("A", "acc"), ("__daisy_dev_A", "__daisy_dev_acc")
    )
    i32 = Scalar(PrimitiveType.Int32)
    for name in m_names + r_names:
        builder.add_container(name, i32, is_argument=False)

    nbytes_a = f"{n} * {FLOAT_BYTES}"
    nbytes_acc = f"{rows} * {FLOAT_BYTES}"
    backend.offload(
        builder,
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes_a,
    )
    backend.offload(
        builder,
        "__daisy_dev_acc",
        "__daisy_dev_acc",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes_acc,
    )
    backend.offload(
        builder,
        "A",
        "__daisy_dev_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_a,
    )
    backend.offload(
        builder,
        "acc",
        "__daisy_dev_acc",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_acc,
    )

    for name, (level, count, psize) in zip(m_names, map_specs):
        builder.begin_map(name, "0", str(count), "1", backend.schedule(level, psize))
    for name, (level, count, psize) in zip(r_names, reduce_specs):
        builder.begin_reduce(
            name,
            "0",
            str(count),
            "1",
            [(op, "__daisy_dev_acc")],
            backend.schedule(level, psize),
        )

    blk = builder.add_block()
    a = builder.add_access(blk, "__daisy_dev_A")
    acc_in = builder.add_access(blk, "__daisy_dev_acc")
    acc_out = builder.add_access(blk, "__daisy_dev_acc")
    t = _add_body_op(builder, blk, op)
    builder.add_memlet(blk, acc_in, "", t, "_in1", row_flat)
    builder.add_memlet(blk, a, "", t, "_in2", in_index)
    builder.add_memlet(blk, t, "_out", acc_out, "", row_flat)

    for _ in reduce_specs:
        builder.end_reduce()
    for _ in map_specs:
        builder.end_map()

    backend.offload(
        builder,
        "acc",
        "__daisy_dev_acc",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_acc,
    )
    for name in ("__daisy_dev_A", "__daisy_dev_acc"):
        backend.offload(
            builder,
            name,
            name,
            DataTransferDirection.NONE,
            BufferLifecycle.FREE,
            dev_ptr,
            "0",
        )

    return builder.move(), rows, cols


# ---------------------------------------------------------------------------
# One reduce node, several accumulators: multiple reductions at a single level.
# ---------------------------------------------------------------------------
def build_multi_reduction_node(backend, reduce_specs, wrap_specs, ops):
    """Reduction nest with several accumulators, one operator each.

    ``reduce_specs`` (outer->inner) is the reduction axis -- possibly spread
    over several levels that all fold into every accumulator -- while
    ``wrap_specs`` are trivial single-iteration maps that only exist to give a
    block/warp reduction its required grid parent(s).  Every accumulator sees
    the SAME input element, so this exercises N independent reductions sharing
    one reduce node.
    """
    _validate_nest(
        [(lvl, ps) for (lvl, _c, ps) in wrap_specs + reduce_specs], backend.warp_size
    )
    assert all(
        c == 1 for (_l, c, _p) in wrap_specs
    ), "wrap maps must be single-iteration"

    red_counts = [c for (_, c, _) in reduce_specs]
    n = math.prod(red_counts)
    r_names = [f"j{k}" for k in range(len(reduce_specs))]
    g_names = [f"g{k}" for k in range(len(wrap_specs))]
    red_flat = _flat_expr(r_names, red_counts)

    accs = [f"acc{i}" for i in range(len(ops))]
    dev_accs = [f"__daisy_dev_{a}" for a in accs]

    builder = StructuredSDFGBuilder("multi_reduction")
    dev_ptr = _declare_common(
        backend, builder, ["A"] + accs, ["__daisy_dev_A"] + dev_accs
    )
    i32 = Scalar(PrimitiveType.Int32)
    for name in r_names + g_names:
        builder.add_container(name, i32, is_argument=False)

    nbytes_a = f"{n} * {FLOAT_BYTES}"
    nbytes_acc = f"{FLOAT_BYTES}"
    backend.offload(
        builder,
        "__daisy_dev_A",
        "__daisy_dev_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes_a,
    )
    backend.offload(
        builder,
        "A",
        "__daisy_dev_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_a,
    )
    for dev in dev_accs:
        backend.offload(
            builder,
            dev,
            dev,
            DataTransferDirection.NONE,
            BufferLifecycle.ALLOC,
            dev_ptr,
            nbytes_acc,
        )
    for host, dev in zip(accs, dev_accs):
        backend.offload(
            builder,
            host,
            dev,
            DataTransferDirection.H2D,
            BufferLifecycle.NO_CHANGE,
            dev_ptr,
            nbytes_acc,
        )

    for name, (level, count, psize) in zip(g_names, wrap_specs):
        builder.begin_map(name, "0", str(count), "1", backend.schedule(level, psize))
    for name, (level, count, psize) in zip(r_names, reduce_specs):
        builder.begin_reduce(
            name,
            "0",
            str(count),
            "1",
            [(op, dev) for op, dev in zip(ops, dev_accs)],
            backend.schedule(level, psize),
        )
    blk = builder.add_block()
    a = builder.add_access(blk, "__daisy_dev_A")
    for op, dev in zip(ops, dev_accs):
        acc_in = builder.add_access(blk, dev)
        acc_out = builder.add_access(blk, dev)
        t = _add_body_op(builder, blk, op)
        builder.add_memlet(blk, acc_in, "", t, "_in1", "0")
        builder.add_memlet(blk, a, "", t, "_in2", red_flat)
        builder.add_memlet(blk, t, "_out", acc_out, "", "0")
    for _ in reduce_specs:
        builder.end_reduce()
    for _ in wrap_specs:
        builder.end_map()

    for host, dev in zip(accs, dev_accs):
        backend.offload(
            builder,
            host,
            dev,
            DataTransferDirection.D2H,
            BufferLifecycle.NO_CHANGE,
            dev_ptr,
            nbytes_acc,
        )
    for dev in ["__daisy_dev_A"] + dev_accs:
        backend.offload(
            builder,
            dev,
            dev,
            DataTransferDirection.NONE,
            BufferLifecycle.FREE,
            dev_ptr,
            "0",
        )

    return builder.move()


# ---------------------------------------------------------------------------
# Nested reduces into DIFFERENT accumulators at different levels: the outer
# reduce folds P[r0]; nested inside it, an inner reduce folds Q[r0, r1].
# ---------------------------------------------------------------------------
def build_reduce_different_vars(
    backend, outer, inner, op_outer, op_inner, grid_wrap=None
):
    outer_level, outer_count, outer_ps = outer
    inner_level, inner_count, inner_ps = inner
    wrap = grid_wrap or []
    _validate_nest(
        [(lvl, ps) for (lvl, _c, ps) in wrap]
        + [(outer_level, outer_ps), (inner_level, inner_ps)],
        backend.warp_size,
    )
    n_q = outer_count * inner_count

    builder = StructuredSDFGBuilder("reduce_diff_vars")
    dev_ptr = _declare_common(
        backend,
        builder,
        ("P", "Q", "acc_out", "acc_in"),
        (
            "__daisy_dev_P",
            "__daisy_dev_Q",
            "__daisy_dev_acc_out",
            "__daisy_dev_acc_in",
        ),
    )
    i32 = Scalar(PrimitiveType.Int32)
    builder.add_container("r0", i32, is_argument=False)
    builder.add_container("r1", i32, is_argument=False)
    g_names = [f"g{k}" for k in range(len(wrap))]
    for name in g_names:
        builder.add_container(name, i32, is_argument=False)

    nbytes_p = f"{outer_count} * {FLOAT_BYTES}"
    nbytes_q = f"{n_q} * {FLOAT_BYTES}"
    nbytes_s = f"{FLOAT_BYTES}"
    for dev, nb in (
        ("__daisy_dev_P", nbytes_p),
        ("__daisy_dev_Q", nbytes_q),
        ("__daisy_dev_acc_out", nbytes_s),
        ("__daisy_dev_acc_in", nbytes_s),
    ):
        backend.offload(
            builder,
            dev,
            dev,
            DataTransferDirection.NONE,
            BufferLifecycle.ALLOC,
            dev_ptr,
            nb,
        )
    for host, dev, nb in (
        ("P", "__daisy_dev_P", nbytes_p),
        ("Q", "__daisy_dev_Q", nbytes_q),
        ("acc_out", "__daisy_dev_acc_out", nbytes_s),
        ("acc_in", "__daisy_dev_acc_in", nbytes_s),
    ):
        backend.offload(
            builder,
            host,
            dev,
            DataTransferDirection.H2D,
            BufferLifecycle.NO_CHANGE,
            dev_ptr,
            nb,
        )

    for name, (gl, gc, gp) in zip(g_names, wrap):
        builder.begin_map(name, "0", str(gc), "1", backend.schedule(gl, gp))
    builder.begin_reduce(
        "r0",
        "0",
        str(outer_count),
        "1",
        [(op_outer, "__daisy_dev_acc_out")],
        backend.schedule(outer_level, outer_ps),
    )
    builder.begin_reduce(
        "r1",
        "0",
        str(inner_count),
        "1",
        [(op_inner, "__daisy_dev_acc_in")],
        backend.schedule(inner_level, inner_ps),
    )
    blk_in = builder.add_block()
    q = builder.add_access(blk_in, "__daisy_dev_Q")
    ai_in = builder.add_access(blk_in, "__daisy_dev_acc_in")
    ai_out = builder.add_access(blk_in, "__daisy_dev_acc_in")
    ti = _add_body_op(builder, blk_in, op_inner)
    builder.add_memlet(blk_in, ai_in, "", ti, "_in1", "0")
    builder.add_memlet(blk_in, q, "", ti, "_in2", f"r0*{inner_count} + r1")
    builder.add_memlet(blk_in, ti, "_out", ai_out, "", "0")
    builder.end_reduce()

    blk_out = builder.add_block()
    p = builder.add_access(blk_out, "__daisy_dev_P")
    ao_in = builder.add_access(blk_out, "__daisy_dev_acc_out")
    ao_out = builder.add_access(blk_out, "__daisy_dev_acc_out")
    to = _add_body_op(builder, blk_out, op_outer)
    builder.add_memlet(blk_out, ao_in, "", to, "_in1", "0")
    builder.add_memlet(blk_out, p, "", to, "_in2", "r0")
    builder.add_memlet(blk_out, to, "_out", ao_out, "", "0")
    builder.end_reduce()

    for _ in wrap:
        builder.end_map()

    for host, dev in (
        ("acc_out", "__daisy_dev_acc_out"),
        ("acc_in", "__daisy_dev_acc_in"),
    ):
        backend.offload(
            builder,
            host,
            dev,
            DataTransferDirection.D2H,
            BufferLifecycle.NO_CHANGE,
            dev_ptr,
            nbytes_s,
        )
    for dev in (
        "__daisy_dev_P",
        "__daisy_dev_Q",
        "__daisy_dev_acc_out",
        "__daisy_dev_acc_in",
    ):
        backend.offload(
            builder,
            dev,
            dev,
            DataTransferDirection.NONE,
            BufferLifecycle.FREE,
            dev_ptr,
            "0",
        )

    return builder.move()


# ---------------------------------------------------------------------------
# Non-perfectly-nested SIBLING loops: a parent grid map whose body holds several
# INDEPENDENT sibling sub-nests -- map(map, map) and map(block, reduce).  The
# offload dispatcher emits NO __syncthreads() between siblings, so a shared
# element is race-free only when its producer and consumer are the SAME thread
# (identical level / count / parallel_size / index -- elementwise).  Cross-thread
# sharing (index reversal, per-row broadcast) is instead made legal by an
# explicit block-local barrier (__syncthreads) placed between the producing and
# consuming sibling; this is only valid at block/warp level, since a grid
# parent's coverage loop is block-uniform (all threads of a block reach it).
# ---------------------------------------------------------------------------
def _dev(name):
    return f"__daisy_dev_{name}"


def _alloc(backend, builder, dev_ptr, name, nbytes):
    backend.offload(
        builder,
        _dev(name),
        _dev(name),
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes,
    )


def _h2d(backend, builder, dev_ptr, name, nbytes):
    backend.offload(
        builder,
        name,
        _dev(name),
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes,
    )


def _d2h(backend, builder, dev_ptr, name, nbytes):
    backend.offload(
        builder,
        name,
        _dev(name),
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes,
    )


def _free(backend, builder, dev_ptr, name):
    backend.offload(
        builder,
        _dev(name),
        _dev(name),
        DataTransferDirection.NONE,
        BufferLifecycle.FREE,
        dev_ptr,
        "0",
    )


def build_sibling_nest(backend, parent, sib, containers, siblings):
    """Parent grid map (or None) whose body holds several sibling sub-nests.

    ``parent``     -- ``(grid_short, R, Pg)`` or ``None`` for top-level grid
                      siblings (each becomes its own kernel launch).
    ``sib``        -- shared column nest ``[(level_short, count, psize), ...]``
                      used by every map/reduce sibling; equal across siblings so
                      a shared element maps to the same thread.
    ``containers`` -- ``name -> (role, nelems)``; role in {'in','out','acc','tmp'}.
                      'in' is copied host->device; 'out'/'acc' device->host; 'acc'
                      is additionally identity-initialised host->device; 'tmp' is
                      a device-only transient (no host argument, no transfer).
    ``siblings``   -- ordered body entries:
        ``('map', op, out, [(in, idx), (in, idx)], out_idx[, sib_override])``
        ``('reduce', op, acc, acc_idx, in, in_idx[, sib_override])``
        ``('barrier',)``  -> a block-local __syncthreads between siblings.
      A ``sib_override`` replaces the shared column nest for that sibling (its
      counts must still multiply to the same ``C``); this lets independent
      siblings run at different parallel sizes to exercise blockDim adaptation.
      Index strings are ``str.format`` templates over ``{i} {row} {c} {col}
      {elem} {rev} {C} {Cm1}``.

    Returns ``(sdfg, R, C, host_names)``.
    """
    has_parent = parent is not None
    R = parent[1] if has_parent else 1
    C = math.prod(c for (_l, c, _p) in sib)

    branch = ([(LEVELS[parent[0]], parent[2])] if has_parent else []) + [
        (LEVELS[l], p) for (l, _c, p) in sib
    ]
    _validate_nest(branch, backend.warp_size)

    builder = StructuredSDFGBuilder("sibling_nest")
    host_names = [n for n, (role, _ne) in containers.items() if role != "tmp"]
    dev_ptr = _declare_common(
        backend, builder, host_names, [_dev(n) for n in containers]
    )
    i32 = Scalar(PrimitiveType.Int32)
    if has_parent:
        builder.add_container("i", i32, is_argument=False)

    def _sib_of(entry):
        if entry[0] == "map" and len(entry) == 6:
            return entry[5]
        if entry[0] == "reduce" and len(entry) == 7:
            return entry[6]
        return sib

    sib_vars = []
    sib_nests = []
    for k, entry in enumerate(siblings):
        if entry[0] == "barrier":
            sib_vars.append(None)
            sib_nests.append(None)
            continue
        this_sib = _sib_of(entry)
        sib_nests.append(this_sib)
        vs = [f"k{k}_c{d}" for d in range(len(this_sib))]
        for v in vs:
            builder.add_container(v, i32, is_argument=False)
        sib_vars.append(vs)

    def nbytes(name):
        return f"{containers[name][1]} * {FLOAT_BYTES}"

    for name in containers:
        _alloc(backend, builder, dev_ptr, name, nbytes(name))
    for name, (role, _ne) in containers.items():
        if role in ("in", "acc"):
            _h2d(backend, builder, dev_ptr, name, nbytes(name))

    if has_parent:
        builder.begin_map(
            "i",
            "0",
            str(R),
            "1",
            backend.schedule(LEVELS[parent[0]], parent[2]),
        )

    i_val = "i" if has_parent else "0"
    for k, entry in enumerate(siblings):
        if entry[0] == "barrier":
            builder.add_barrier_local_block()
            continue
        vs = sib_vars[k]
        this_sib = sib_nests[k]
        counts = [c for (_l, c, _p) in this_sib]
        col_flat = _flat_expr(vs, counts)
        elem = f"({col_flat})" if not has_parent else f"(i)*{C} + ({col_flat})"
        rev_col = f"{C - 1} - ({vs[0]})"
        rev = f"({rev_col})" if not has_parent else f"(i)*{C} + ({rev_col})"
        fmt = dict(
            i=i_val,
            row=i_val,
            c=vs[0],
            col=col_flat,
            elem=elem,
            rev=rev,
            C=C,
            Cm1=C - 1,
        )

        if entry[0] == "map":
            op, out, in_specs, out_idx = entry[1], entry[2], entry[3], entry[4]
            for v, (level, count, psize) in zip(vs, this_sib):
                builder.begin_map(
                    v,
                    "0",
                    str(count),
                    "1",
                    backend.schedule(LEVELS[level], psize),
                )
            blk = builder.add_block()
            node = _add_body_op(builder, blk, op)
            for conn, (cont, idx) in zip(("_in1", "_in2"), in_specs):
                src = builder.add_access(blk, _dev(cont))
                builder.add_memlet(blk, src, "", node, conn, idx.format(**fmt))
            dst = builder.add_access(blk, _dev(out))
            builder.add_memlet(blk, node, "_out", dst, "", out_idx.format(**fmt))
            for _ in this_sib:
                builder.end_map()
        else:  # reduce
            op, acc, acc_idx, cont, in_idx = (
                entry[1],
                entry[2],
                entry[3],
                entry[4],
                entry[5],
            )
            for v, (level, count, psize) in zip(vs, this_sib):
                builder.begin_reduce(
                    v,
                    "0",
                    str(count),
                    "1",
                    [(op, _dev(acc))],
                    backend.schedule(LEVELS[level], psize),
                )
            blk = builder.add_block()
            node = _add_body_op(builder, blk, op)
            acc_in = builder.add_access(blk, _dev(acc))
            src = builder.add_access(blk, _dev(cont))
            acc_out = builder.add_access(blk, _dev(acc))
            builder.add_memlet(blk, acc_in, "", node, "_in1", acc_idx.format(**fmt))
            builder.add_memlet(blk, src, "", node, "_in2", in_idx.format(**fmt))
            builder.add_memlet(blk, node, "_out", acc_out, "", acc_idx.format(**fmt))
            for _ in this_sib:
                builder.end_reduce()

    if has_parent:
        builder.end_map()

    for name, (role, _ne) in containers.items():
        if role in ("out", "acc"):
            _d2h(backend, builder, dev_ptr, name, nbytes(name))
    for name in containers:
        _free(backend, builder, dev_ptr, name)

    return builder.move(), R, C, host_names


def _spec(table):
    return [(LEVELS[name], count, psize) for (name, count, psize) in table]


# (count, parallel_size) pairs realising every relationship between a loop's
# iteration count and its schedule parallel_size.  Sweeping these drives the
# grid/block coverage loop and its boundary guard into every regime:
#   smaller  -- count < parallel_size: single pass, boundary guard active
#   equals   -- count == parallel_size: single exact pass
#   multiple -- count == k*parallel_size (k>1): whole coverage tiles, no remainder
#   ragged   -- count > parallel_size, not a multiple: coverage tiles + remainder
_SIZE_RELATIONS = {
    "smaller": (3, 8),
    "equals": (8, 8),
    "multiple": (16, 8),
    "ragged": (10, 4),
}


def _warp_relations(ws):
    """Warp count/parallel_size regimes for a given warp/wavefront width ``ws``.

    A warp's parallel_size is architecturally fixed at ``ws`` (its lanes ARE the
    hardware lanes), but the iteration COUNT it folds is not: each lane walks its
    slice of the count sequentially before the single cross-lane shuffle.  Sweep
    the warp coverage loop into every regime while keeping parallel_size == ``ws``:
      smaller  -- count < ws: some lanes fold nothing (identity), guard active
      equals   -- count == ws: one element per lane
      multiple -- count == k*ws: every lane folds k elements, no remainder
      ragged   -- count > ws, not a multiple: full tiles + partial remainder
    """
    return {
        "smaller": (ws - 12, ws),
        "equals": (ws, ws),
        "multiple": (3 * ws, ws),
        "ragged": (ws + ws // 2 + 2, ws),
    }


# Grid dimension that each block dimension must be nested inside (short names).
_GRID_SHORT = {"xb": "xg", "yb": "yg", "zb": "zg"}


def _with_grid_parent(level, count, psize):
    """Spec table for a single dimension, adding the trivial size-1 grid parent
    that a block dimension requires (grids may stand alone)."""
    if level in _GRID_SHORT:
        return [(_GRID_SHORT[level], 1, 1), (level, count, psize)]
    return [(level, count, psize)]


def _single_dim_size_scenarios():
    """One offloaded dimension per scenario, sweeping every count/parallel_size
    relationship across every grid and block level."""
    out = []
    for level in ("xg", "yg", "zg", "xb", "yb", "zb"):
        for rel, (c, p) in _SIZE_RELATIONS.items():
            out.append((f"size_{level}_{rel}", _with_grid_parent(level, c, p), "add"))
    return out


def _two_dim_size_tables(a_level, b_level, parents):
    """Full cross product of relationships across two offloaded dimensions,
    prefixed by any required size-1 grid parents.  Yields (id, table)."""
    for (ra, (ca, pa)), (rb, (cb, pb)) in itertools.product(
        _SIZE_RELATIONS.items(), repeat=2
    ):
        table = list(parents) + [(a_level, ca, pa), (b_level, cb, pb)]
        yield (f"size_{a_level}_{ra}_{b_level}_{rb}", table)


# ---------------------------------------------------------------------------
# MAP: arbitrary depth / combination of levels.
# ---------------------------------------------------------------------------
def make_map_nest_scenarios(ws):
    """Element-wise MAP nest scenarios for warp/wavefront width ``ws``.

    Every warp scenario is expressed in terms of ``ws`` so the enclosing full-warp
    block (``ws`` threads) and the warp's ``ws``-wide parallel size adapt to both
    the CUDA (32) and ROCm (64) architectures.
    """
    warp = _warp_relations(ws)
    scns = [
        # grid dimensions may stand alone
        ("yg_single", [("yg", 10, 4)], "add"),
        ("zg_single", [("zg", 8, 4)], "mul"),
        # a block dimension must sit under its corresponding grid dimension
        ("yg_yb", [("yg", 3, 2), ("yb", 5, 4)], "add"),
        ("zg_zb", [("zg", 2, 2), ("zb", 6, 3)], "mul"),
        ("xg_yg", [("xg", 6, 4), ("yg", 5, 2)], "add"),
        ("xg_yg_zg", [("xg", 4, 2), ("yg", 5, 2), ("zg", 3, 2)], "add"),
        ("xg_xb", [("xg", 6, 4), ("xb", 5, 4)], "add"),
        (
            "xg_xb_yg_yb",
            [("xg", 3, 2), ("xb", 4, 4), ("yg", 4, 2), ("yb", 3, 3)],
            "mul",
        ),
        (
            "xg_yg_xb_yb",
            [("xg", 3, 2), ("yg", 4, 2), ("xb", 4, 4), ("yb", 3, 3)],
            "add",
        ),
        (
            "all_six",
            [
                ("xg", 2, 2),
                ("yg", 3, 2),
                ("zg", 2, 2),
                ("xb", 2, 2),
                ("yb", 3, 3),
                ("zb", 2, 2),
            ],
            "add",
        ),
        # WARP must be innermost and nested under an X_BLOCK (itself under X_GRID)
        ("xg_xb_warp", [("xg", 4, 2), ("xb", ws, ws), ("w", 3, ws)], "add"),
        (
            "xg_yg_xb_yb_warp",
            [("xg", 2, 2), ("yg", 2, 2), ("xb", ws, ws), ("yb", 2, 2), ("w", 2, ws)],
            "add",
        ),
        # the complete seven-level stack: all three grids, all three blocks, a warp
        (
            "all_seven",
            [
                ("xg", 2, 2),
                ("yg", 2, 2),
                ("zg", 2, 2),
                ("xb", ws, ws),
                ("yb", 2, 2),
                ("zb", 2, 2),
                ("w", 2, ws),
            ],
            "add",
        ),
        # min / max element-wise maps over a couple of nests
        ("xg_yg_min", [("xg", 6, 4), ("yg", 5, 2)], "min"),
        ("xg_xb_warp_max", [("xg", 4, 2), ("xb", ws, ws), ("w", 3, ws)], "max"),
    ]

    # Systematic count-vs-parallel_size sweeps (see _SIZE_RELATIONS): every single
    # dimension in each regime, plus full cross products over two dimensions -- pure
    # grids (xg,yg), grid+block on the SAME axis (xg,xb: grid coverage nested inside
    # block coverage) and two block dimensions (xb,yb) under trivial grid parents --
    # and a three-dimensional diagonal for both grids and blocks.
    scns += _single_dim_size_scenarios()
    scns += [
        (sid, table, "add") for (sid, table) in _two_dim_size_tables("xg", "yg", [])
    ]
    scns += [
        (sid, table, "add") for (sid, table) in _two_dim_size_tables("xg", "xb", [])
    ]
    scns += [
        (sid, table, "add")
        for (sid, table) in _two_dim_size_tables(
            "xb", "yb", [("xg", 1, 1), ("yg", 1, 1)]
        )
    ]
    for rel, (c, p) in _SIZE_RELATIONS.items():
        scns.append(
            (
                f"size_xyz_grid_{rel}",
                [("xg", c, p), ("yg", c, p), ("zg", c, p)],
                "add",
            )
        )
        scns.append(
            (
                f"size_xyz_block_{rel}",
                [
                    ("xg", 1, 1),
                    ("yg", 1, 1),
                    ("zg", 1, 1),
                    ("xb", c, p),
                    ("yb", c, p),
                    ("zb", c, p),
                ],
                "add",
            )
        )

    # Warp coverage-loop size sweeps (parallel_size fixed at ws, count varied): the
    # warp innermost under its X_BLOCK/X_GRID owner, alone and crossed with an
    # enclosing grid whose own coverage loop is swept in the same regimes.
    for wrel, (wc, wp) in warp.items():
        scns.append(
            (
                f"size_warp_{wrel}",
                [("xg", 1, 1), ("xb", ws, ws), ("w", wc, wp)],
                "add",
            )
        )
    for (grel, (gc, gp)), (wrel, (wc, wp)) in itertools.product(
        _SIZE_RELATIONS.items(), warp.items()
    ):
        scns.append(
            (
                f"size_xg_{grel}_warp_{wrel}",
                [("xg", gc, gp), ("xb", ws, ws), ("w", wc, wp)],
                "add",
            )
        )
    # warp count crossed with its X_BLOCK owner's own count regime: the block folds
    # its coverage tiles (count == k*ws over a fixed ws parallel_size) while the warp
    # independently folds its own swept coverage loop.
    for (brel, (bc, bp)), (wrel, (wc, wp)) in itertools.product(
        _SIZE_RELATIONS.items(), warp.items()
    ):
        scns.append(
            (
                f"size_xb_{brel}_warp_{wrel}",
                [("xg", 1, 1), ("xb", bc * ws, ws), ("w", wc, wp)],
                "add",
            )
        )
    return scns


# ---------------------------------------------------------------------------
# REDUCE: reduction axis over an arbitrary level nest (same accumulator), and
# map+reduce combinations across different levels.
# ---------------------------------------------------------------------------
def _reduce_size_scenarios(ws):
    """Systematic count-vs-parallel_size sweeps for reductions (see
    _SIZE_RELATIONS): every single reduction dimension in each regime, plus full
    cross products over two reduced dimensions -- grid+block on the SAME axis
    (xg,xb folded into one scalar), two block dimensions (xb,yb), and a mapped
    (parallel) row axis crossed with a reduced column axis (map xg x reduce xb)
    -- and a three-dimensional block-reduction diagonal.  ``add`` keeps the
    reference numerically exact regardless of element count.
    """
    out = []
    rels = list(_SIZE_RELATIONS.items())

    # single reduced grid dimension (grids stand alone)
    for rel, (c, p) in rels:
        out.append((f"rsize_xg_{rel}", [], [("xg", c, p)], "add"))
    # single reduced block dimension, under its trivial size-1 grid parent
    for level in ("xb", "yb", "zb"):
        grid = _GRID_SHORT[level]
        for rel, (c, p) in rels:
            out.append((f"rsize_{level}_{rel}", [(grid, 1, 1)], [(level, c, p)], "add"))

    # grid + block on the SAME axis, both folding the one scalar accumulator
    for (ra, (ca, pa)), (rb, (cb, pb)) in itertools.product(rels, repeat=2):
        out.append(
            (
                f"rsize_xg_{ra}_xb_{rb}",
                [],
                [("xg", ca, pa), ("xb", cb, pb)],
                "add",
            )
        )
    # two reduced block dimensions under trivial grid parents
    for (ra, (ca, pa)), (rb, (cb, pb)) in itertools.product(rels, repeat=2):
        out.append(
            (
                f"rsize_xb_{ra}_yb_{rb}",
                [("xg", 1, 1), ("yg", 1, 1)],
                [("xb", ca, pa), ("yb", cb, pb)],
                "add",
            )
        )
    # parallel (mapped) rows crossed with a reduced column axis
    for (ra, (ca, pa)), (rb, (cb, pb)) in itertools.product(rels, repeat=2):
        out.append(
            (
                f"rsize_map_xg_{ra}_red_xb_{rb}",
                [("xg", ca, pa)],
                [("xb", cb, pb)],
                "add",
            )
        )
    # three reduced block dimensions, diagonal (all in the same regime)
    for rel, (c, p) in rels:
        out.append(
            (
                f"rsize_xyz_block_{rel}",
                [("xg", 1, 1), ("yg", 1, 1), ("zg", 1, 1)],
                [("xb", c, p), ("yb", c, p), ("zb", c, p)],
                "add",
            )
        )

    # reduced warp, parallel_size fixed at ws, coverage-loop count swept.
    # The warp is reduced under its owning X_BLOCK (also reduced) and grid.
    warp_rels = list(_warp_relations(ws).items())
    for wrel, (wc, wp) in warp_rels:
        out.append(
            (
                f"rsize_warp_{wrel}",
                [("xg", 1, 1)],
                [("xb", ws, ws), ("w", wc, wp)],
                "add",
            )
        )
    # warp count crossed with the reduced X_BLOCK owner's own count regime: the
    # block folds its coverage tiles while the warp folds its (independent) ones.
    for (brel, (bc, bp)), (wrel, (wc, wp)) in itertools.product(rels, warp_rels):
        out.append(
            (
                f"rsize_xb_{brel}_warp_{wrel}",
                [("xg", 1, 1)],
                [("xb", bc * ws, ws), ("w", wc, wp)],
                "add",
            )
        )
    # mapped (parallel) grid rows crossed with a reduced warp of swept count
    for (grel, (gc, gp)), (wrel, (wc, wp)) in itertools.product(rels, warp_rels):
        out.append(
            (
                f"rsize_map_xg_{grel}_red_warp_{wrel}",
                [("xg", gc, gp)],
                [("xb", ws, ws), ("w", wc, wp)],
                "add",
            )
        )
    return out


def make_reduce_nest_scenarios(ws):
    """Reduction MAP+REDUCE nest scenarios for warp/wavefront width ``ws``."""
    scns = [
        # full reduction over a single grid level (grids may stand alone)
        ("reduce_xg", [], [("xg", 1000, 256)], "add"),
        # a block-level reduction must sit under its grid -> trivial single-row grid
        ("reduce_xg_xb", [("xg", 1, 1)], [("xb", 64, 64)], "add"),
        ("reduce_yg_yb", [("yg", 1, 1)], [("yb", 8, 4)], "add"),
        # reduction axis split across several levels, all folding the SAME accumulator
        ("reduce_xg_xb_same", [], [("xg", 8, 4), ("xb", 5, 5)], "add"),
        (
            "reduce_xg_xb_warp_same",
            [],
            [("xg", 4, 2), ("xb", ws, ws), ("w", 4, ws)],
            "add",
        ),
        # map (parallel rows) + reduction (columns) on validly nested levels
        ("map_xg_reduce_xb", [("xg", 8, 4)], [("xb", 16, 8)], "add"),
        ("map_xg_yg_reduce_xb", [("xg", 3, 2), ("yg", 4, 2)], [("xb", 8, 4)], "mul"),
        (
            "map_xg_reduce_xb_warp",
            [("xg", 4, 2)],
            [("xb", ws, ws), ("w", 3, ws)],
            "add",
        ),
        # min / max reductions
        ("reduce_xg_min", [], [("xg", 1000, 256)], "min"),
        ("reduce_xg_xb_max_same", [], [("xg", 8, 4), ("xb", 5, 5)], "max"),
        ("map_xg_reduce_xb_min", [("xg", 8, 4)], [("xb", 16, 8)], "min"),
        # multi-dimensional BLOCK reductions: a single reduce node folding over two
        # or three block dimensions (xb+yb[+zb]), each nested under a trivial grid.
        # The "_cover" variants use count > parallel_size so the block coverage loop
        # iterates multiple times; the fully-covered variants use count == psize.
        (
            "reduce_xb_yb",
            [("xg", 1, 1), ("yg", 1, 1)],
            [("xb", 8, 8), ("yb", 4, 4)],
            "add",
        ),
        (
            "reduce_xb_yb_cover",
            [("xg", 1, 1), ("yg", 1, 1)],
            [("xb", 8, 4), ("yb", 4, 4)],
            "add",
        ),
        (
            "reduce_xb_yb_zb",
            [("xg", 1, 1), ("yg", 1, 1), ("zg", 1, 1)],
            [("xb", 8, 8), ("yb", 2, 2), ("zb", 2, 2)],
            "add",
        ),
        (
            "reduce_xb_yb_zb_cover",
            [("xg", 1, 1), ("yg", 1, 1), ("zg", 1, 1)],
            [("xb", 8, 4), ("yb", 2, 2), ("zb", 2, 2)],
            "add",
        ),
        (
            "reduce_xb32_yb",
            [("xg", 1, 1), ("yg", 1, 1)],
            [("xb", 32, 32), ("yb", 2, 2)],
            "add",
        ),
        (
            "reduce_xb_yb_warp",
            [("xg", 1, 1), ("yg", 1, 1)],
            [("xb", ws, ws), ("yb", 2, 2), ("w", 2, ws)],
            "add",
        ),
        # full seven-level stack: parallel rows over all three grids, reduction
        # (columns) over all three blocks and a warp, folding the SAME accumulator
        (
            "map_xyzg_reduce_xyzb_warp",
            [("xg", 2, 2), ("yg", 2, 2), ("zg", 2, 2)],
            [("xb", ws, ws), ("yb", 2, 2), ("zb", 2, 2), ("w", 2, ws)],
            "add",
        ),
        (
            "map_xyzg_reduce_xyzb_warp_max",
            [("xg", 2, 2), ("yg", 2, 2), ("zg", 2, 2)],
            [("xb", ws, ws), ("yb", 2, 2), ("zb", 2, 2), ("w", 2, ws)],
            "max",
        ),
        # the entire stack folded into ONE scalar: many different dimensions all
        # accumulating into the same value
        (
            "reduce_full_scalar_max",
            [],
            [
                ("xg", 2, 2),
                ("yg", 2, 2),
                ("zg", 2, 2),
                ("xb", ws, ws),
                ("yb", 2, 2),
                ("zb", 2, 2),
                ("w", 2, ws),
            ],
            "max",
        ),
        (
            "reduce_full_scalar_mul",
            [],
            [
                ("xg", 2, 2),
                ("yg", 2, 2),
                ("zg", 2, 2),
                ("xb", ws, ws),
                ("yb", 2, 2),
                ("zb", 2, 2),
                ("w", 2, ws),
            ],
            "mul",
        ),
    ]
    scns += _reduce_size_scenarios(ws)
    return scns


# ---------------------------------------------------------------------------
# Multiple reductions in ONE node (several accumulators at one level).
# ---------------------------------------------------------------------------
def make_multi_reduction_scenarios(ws):
    """Several-accumulator single-reduce-node scenarios for width ``ws``."""
    return [
        ("multi_xg", [("xg", 1000, 256)], [], ["add", "mul"]),
        ("multi_xg_minmax", [("xg", 1000, 256)], [], ["min", "max"]),
        ("multi_xg_all", [("xg", 1000, 256)], [], ["add", "mul", "min", "max"]),
        ("multi_xg_xb", [("xb", ws, ws)], [("xg", 1, 1)], ["add", "mul"]),
        ("multi_xg_xb_narrow", [("xb", 8, 8)], [("xg", 1, 1)], ["add", "mul"]),
        # several accumulators over a full block + warp reduction stack under a grid
        (
            "multi_full_stack",
            [("xb", ws, ws), ("yb", 2, 2), ("zb", 2, 2), ("w", 2, ws)],
            [("xg", 1, 1), ("yg", 1, 1), ("zg", 1, 1)],
            ["add", "mul", "min", "max"],
        ),
    ]


# ---------------------------------------------------------------------------
# Nested reduces into different accumulators at different levels.
# ---------------------------------------------------------------------------
def make_diff_var_scenarios(ws):
    """Nested-different-accumulator scenarios for width ``ws``."""
    return [
        ("diff_xg_xb", ("xg", 8, 4), ("xb", 5, 5), "add", "add", None),
        ("diff_xg_xb_amul", ("xg", 6, 3), ("xb", 4, 4), "add", "mul", None),
        ("diff_xg_xb_minmax", ("xg", 8, 4), ("xb", 5, 5), "min", "max", None),
        (
            "diff_xg_xb_warp",
            ("xb", ws, ws),
            ("w", ws, ws),
            "add",
            "add",
            [("xg", 1, 1)],
        ),
    ]


# ---------------------------------------------------------------------------
# Sibling (non-perfectly-nested) scenarios.  Each factory returns the container
# table, the ordered sibling body and a NumPy reference; legality is analysed in
# the module docstring above ``build_sibling_nest``.  Only race-free shapes are
# exercised: independent outputs, same-thread producer->consumer chains, and
# cross-thread reversal / broadcast that are guarded by an explicit barrier.
# ---------------------------------------------------------------------------
def _p_independent(R, C):
    return (
        {
            "A": ("in", R * C),
            "B": ("in", R * C),
            "C1": ("out", R * C),
            "C2": ("out", R * C),
        },
        [
            ("map", "add", "C1", [("A", "{elem}"), ("B", "{elem}")], "{elem}"),
            ("map", "mul", "C2", [("A", "{elem}"), ("B", "{elem}")], "{elem}"),
        ],
        lambda ins, R, C: {"C1": ins["A"] + ins["B"], "C2": ins["A"] * ins["B"]},
    )


def _p_chain(R, C):
    return (
        {
            "A": ("in", R * C),
            "B": ("in", R * C),
            "T": ("tmp", R * C),
            "Cout": ("out", R * C),
        },
        [
            ("map", "add", "T", [("A", "{elem}"), ("B", "{elem}")], "{elem}"),
            ("map", "mul", "Cout", [("T", "{elem}"), ("T", "{elem}")], "{elem}"),
        ],
        lambda ins, R, C: {"Cout": (ins["A"] + ins["B"]) ** 2},
    )


def _p_map_reduce(R, C):
    return (
        {
            "A": ("in", R * C),
            "B": ("in", R * C),
            "T": ("tmp", R * C),
            "acc": ("acc", R),
        },
        [
            ("map", "add", "T", [("A", "{elem}"), ("B", "{elem}")], "{elem}"),
            ("reduce", "add", "acc", "{row}", "T", "{elem}"),
        ],
        lambda ins, R, C: {"acc": (ins["A"] + ins["B"]).reshape(R, C).sum(axis=1)},
    )


def _p_reverse(R, C):
    # cross-thread: thread j reads T[C-1-j]; legal ONLY with the barrier below.
    return (
        {
            "A": ("in", R * C),
            "B": ("in", R * C),
            "T": ("tmp", R * C),
            "Cout": ("out", R * C),
        },
        [
            ("map", "add", "T", [("A", "{elem}"), ("B", "{elem}")], "{elem}"),
            ("barrier",),
            ("map", "add", "Cout", [("T", "{rev}"), ("A", "{elem}")], "{elem}"),
        ],
        lambda ins, R, C: {
            "Cout": (
                (ins["A"] + ins["B"]).reshape(R, C)[:, ::-1] + ins["A"].reshape(R, C)
            ).ravel()
        },
    )


def _p_broadcast(R, C):
    # cross-thread: every lane reads the row scalar s[i]; legal ONLY with barrier.
    return (
        {"A": ("in", R * C), "s": ("acc", R), "Cout": ("out", R * C)},
        [
            ("reduce", "add", "s", "{row}", "A", "{elem}"),
            ("barrier",),
            ("map", "mul", "Cout", [("A", "{elem}"), ("s", "{row}")], "{elem}"),
        ],
        lambda ins, R, C: {
            "s": ins["A"].reshape(R, C).sum(axis=1),
            "Cout": (
                ins["A"].reshape(R, C)
                * ins["A"].reshape(R, C).sum(axis=1, keepdims=True)
            ).ravel(),
        },
    )


def _sib_scn(sid, parent, sib, factory):
    R = parent[1] if parent else 1
    C = math.prod(c for (_l, c, _p) in sib)
    conts, sibs, ref = factory(R, C)
    return dict(
        id=sid, parent=parent, sib=sib, containers=conts, siblings=sibs, ref=ref
    )


def _sib_indep_mixed(sid, parent, C, p1, p2):
    """Two INDEPENDENT siblings over the same column count ``C`` but different
    parallel sizes, so the parent's blockDim must adapt to the larger of the two
    (the ``get_nested_schedule_types`` largest-wins rule)."""
    R = parent[1]
    conts = {
        "A": ("in", R * C),
        "B": ("in", R * C),
        "C1": ("out", R * C),
        "C2": ("out", R * C),
    }
    sibs = [
        (
            "map",
            "add",
            "C1",
            [("A", "{elem}"), ("B", "{elem}")],
            "{elem}",
            [("xb", C, p1)],
        ),
        (
            "map",
            "mul",
            "C2",
            [("A", "{elem}"), ("B", "{elem}")],
            "{elem}",
            [("xb", C, p2)],
        ),
    ]
    return dict(
        id=sid,
        parent=parent,
        sib=[("xb", C, max(p1, p2))],
        containers=conts,
        siblings=sibs,
        ref=lambda ins, R, C: {"C1": ins["A"] + ins["B"], "C2": ins["A"] * ins["B"]},
    )


def make_sibling_scenarios(ws):
    """Sibling (non-perfectly-nested) scenarios for warp/wavefront width ``ws``."""
    scns = [
        # Independent siblings write disjoint arrays -> race-free, no barrier needed;
        # swept across every block level (each under its own grid) and two top-level
        # grid siblings (each lowered to its own kernel launch).
        _sib_scn("indep_xb", ("xg", 4, 2), [("xb", 8, 8)], _p_independent),
        _sib_scn("indep_yb", ("yg", 4, 2), [("yb", 8, 8)], _p_independent),
        _sib_scn("indep_zb", ("zg", 4, 2), [("zb", 8, 8)], _p_independent),
        _sib_scn("indep_grid", None, [("xg", 32, 8)], _p_independent),
        # Same-thread producer -> consumer chain: T written and read at the same
        # (row, col) -> race-free without a barrier.
        _sib_scn("chain_xb", ("xg", 4, 2), [("xb", 8, 8)], _p_chain),
        _sib_scn("chain_warp", ("xg", 2, 2), [("xb", ws, ws), ("w", 4, ws)], _p_chain),
        # A block map feeding a block/warp reduction -- map(block, reduce); the
        # reduce reads each thread's own T element, so no barrier is required.
        _sib_scn("map_reduce_xb", ("xg", 4, 2), [("xb", 8, 8)], _p_map_reduce),
        _sib_scn("map_reduce_yb", ("yg", 4, 2), [("yb", 8, 8)], _p_map_reduce),
        _sib_scn(
            "map_reduce_warp",
            ("xg", 2, 2),
            [("xb", ws, ws), ("w", 4, ws)],
            _p_map_reduce,
        ),
        # Cross-thread index reversal made legal by a block-local barrier between the
        # producing and consuming sibling (full-cover and coverage-loop variants).
        _sib_scn("reverse_barrier_xb", ("xg", 4, 2), [("xb", 8, 8)], _p_reverse),
        _sib_scn("reverse_barrier_xb_cover", ("xg", 4, 2), [("xb", 8, 4)], _p_reverse),
        # Reduce -> per-row broadcast made legal by a block-local barrier.
        _sib_scn("broadcast_barrier_xb", ("xg", 4, 2), [("xb", 8, 8)], _p_broadcast),
        _sib_scn(
            "broadcast_barrier_warp",
            ("xg", 2, 2),
            [("xb", ws, ws), ("w", 4, ws)],
            _p_broadcast,
        ),
    ]

    # Sweep the shared sibling column dimension across every count/parallel_size
    # regime for each sibling pattern, so each sibling's block coverage loop and
    # boundary guard are driven into all four regimes while it sits beside another
    # sibling under the same grid parent.
    patterns = [
        ("indep", _p_independent),
        ("chain", _p_chain),
        ("mapred", _p_map_reduce),
        ("revbar", _p_reverse),
        ("bcastbar", _p_broadcast),
    ]
    for pname, factory in patterns:
        for rel, (c, p) in _SIZE_RELATIONS.items():
            scns.append(
                _sib_scn(
                    f"size_{pname}_xb_{rel}", ("xg", 4, 2), [("xb", c, p)], factory
                )
            )
    # a reduced warp sibling whose coverage-loop count is swept while its owning
    # X_BLOCK stays a full warp -- map(block-map, warp-reduce) with a varied warp.
    for wrel, (wc, wp) in _warp_relations(ws).items():
        scns.append(
            _sib_scn(
                f"size_mapred_warp_{wrel}",
                ("xg", 2, 2),
                [("xb", ws, ws), ("w", wc, wp)],
                _p_map_reduce,
            )
        )

    # Independent siblings sharing a column count but running at DIFFERENT parallel
    # sizes; the parent blockDim must cover the larger sibling (largest-wins).
    scns += [
        _sib_indep_mixed("indep_mix_8_16", ("xg", 4, 2), 16, 8, 16),
        _sib_indep_mixed("indep_mix_16_8", ("xg", 4, 2), 16, 16, 8),
        _sib_indep_mixed("indep_mix_4_16", ("xg", 4, 2), 16, 4, 16),
        _sib_indep_mixed("indep_mix_ragged", ("xg", 4, 2), 10, 4, 8),
    ]
    return scns


# ===========================================================================
# Suite registration: bind every parametrized test to a concrete backend and
# inject it into the calling (thin) suite module's globals.  The thin suite's
# module-level ``pytestmark`` (cuda / rocm) then applies to all injected tests.
# ===========================================================================
def register(namespace, backend):
    """Define every offload-dispatcher test, bound to ``backend``, into
    ``namespace`` (a thin suite module's ``globals()``)."""
    ws = backend.warp_size

    @pytest.mark.parametrize(
        "target_level",
        [TargetLevel.X_GRID, TargetLevel.X_BLOCK],
        ids=["grid", "block"],
    )
    @pytest.mark.parametrize("op", ["add", "mul", "min", "max"])
    @pytest.mark.parametrize(
        "n,parallel_size",
        [
            (1024, 128),  # exact-fit: parallel_size divides n
            (1000, 250),  # exact-fit
            (1000, 256),  # ragged: 256 does not divide 1000
            (100, 32),  # ragged, non-power-of-two remainder
            (100, 128),  # parallel_size > n (single pass, boundary guard)
            (1, 32),  # degenerate size-1 iteration space
        ],
        ids=["1024x128", "1000x250", "1000x256", "100x32", "100x128", "size1"],
    )
    def test_offload_map(target_level, op, n, parallel_size, tmp_path):
        sdfg = build_offloaded_elementwise(backend, n, op, target_level, parallel_size)
        compiled = _compile(backend, sdfg, tmp_path / "map")

        rng = np.random.default_rng(0)
        a = rng.standard_normal(n).astype(np.float32)
        b = rng.standard_normal(n).astype(np.float32)
        c = np.zeros(n, dtype=np.float32)

        compiled(a, b, c)

        np.testing.assert_allclose(c, numpy_elementwise(a, b, op), rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("op", ["add", "mul", "min", "max"])
    @pytest.mark.parametrize(
        "n,parallel_size",
        [
            (1000, 256),  # ragged
            (1000, 250),  # exact-fit
            (100, 32),  # ragged, non-power-of-two remainder
            (1, 32),  # degenerate size-1 reduction axis
        ],
        ids=["1000x256", "1000x250", "100x32", "size1"],
    )
    def test_offload_reduce(op, n, parallel_size, tmp_path):
        sdfg = build_offloaded_reduction(
            backend, n, op, TargetLevel.X_GRID, parallel_size
        )
        compiled = _compile(backend, sdfg, tmp_path / "reduce")

        a = _rng_input(op, n, 1)
        acc = np.full(1, _identity(op), dtype=np.float32)

        compiled(a, acc)

        ref = numpy_reduce(a, op)
        np.testing.assert_allclose(acc[0], ref, rtol=1e-4, atol=1e-4)

    _map_scns = make_map_nest_scenarios(ws)

    @pytest.mark.parametrize(
        "specs,op",
        [(_spec(t), op) for (_id, t, op) in _map_scns],
        ids=[s[0] for s in _map_scns],
    )
    def test_map_nest(specs, op, tmp_path):
        sdfg, n = build_map_nest(backend, specs, op)
        compiled = _compile(backend, sdfg, tmp_path / "map_nest")

        a = _rng_input(op, n, 10)
        b = _rng_input(op, n, 11)
        c = np.zeros(n, dtype=np.float32)
        compiled(a, b, c)

        np.testing.assert_allclose(c, numpy_elementwise(a, b, op), rtol=1e-4, atol=1e-5)

    _reduce_scns = make_reduce_nest_scenarios(ws)

    @pytest.mark.parametrize(
        "map_specs,reduce_specs,op",
        [
            pytest.param(_spec(m), _spec(r), op, id=_id)
            for (_id, m, r, op) in _reduce_scns
        ],
    )
    def test_reduce_nest(map_specs, reduce_specs, op, tmp_path):
        sdfg, rows, cols = build_reduce_nest(backend, map_specs, reduce_specs, op)
        compiled = _compile(backend, sdfg, tmp_path / "reduce_nest")

        a = _rng_input(op, rows * cols, 20)
        acc = np.full(rows, _identity(op), dtype=np.float32)
        compiled(a, acc)

        mat = a.reshape(rows, cols)
        ref = {"add": mat.sum, "mul": mat.prod, "min": mat.min, "max": mat.max}[op](
            axis=1
        )
        np.testing.assert_allclose(acc, ref.astype(np.float32), rtol=1e-4, atol=1e-4)

    _multi_scns = make_multi_reduction_scenarios(ws)

    @pytest.mark.parametrize(
        "reduce_specs,wrap_specs,ops",
        [
            pytest.param(_spec(r), _spec(w), ops, id=_id)
            for (_id, r, w, ops) in _multi_scns
        ],
    )
    def test_multi_reduction_one_node(reduce_specs, wrap_specs, ops, tmp_path):
        n = math.prod(c for (_l, c, _p) in reduce_specs)
        sdfg = build_multi_reduction_node(backend, reduce_specs, wrap_specs, ops)
        compiled = _compile(backend, sdfg, tmp_path / "multi_reduction")

        # near-1 factors keep every accumulator (including the product) well-conditioned
        a = _rng_input("mul", n, 30)
        outs = [np.full(1, _identity(op), dtype=np.float32) for op in ops]
        compiled(a, *outs)

        for op, out in zip(ops, outs):
            np.testing.assert_allclose(
                out[0], numpy_reduce(a, op), rtol=1e-4, atol=1e-4
            )

    _diff_scns = make_diff_var_scenarios(ws)

    @pytest.mark.parametrize(
        "outer,inner,op_outer,op_inner,grid_wrap",
        [
            (
                (LEVELS[o[0]], o[1], o[2]),
                (LEVELS[i[0]], i[1], i[2]),
                oo,
                oi,
                _spec(gw) if gw else None,
            )
            for (_id, o, i, oo, oi, gw) in _diff_scns
        ],
        ids=[s[0] for s in _diff_scns],
    )
    def test_reduce_different_vars(
        outer, inner, op_outer, op_inner, grid_wrap, tmp_path
    ):
        outer_count = outer[1]
        inner_count = inner[1]
        sdfg = build_reduce_different_vars(
            backend, outer, inner, op_outer, op_inner, grid_wrap
        )
        compiled = _compile(backend, sdfg, tmp_path / "diff_vars")

        p = _rng_input(op_outer, outer_count, 40)
        q = _rng_input(op_inner, outer_count * inner_count, 41)
        acc_out = np.full(1, _identity(op_outer), dtype=np.float32)
        acc_in = np.full(1, _identity(op_inner), dtype=np.float32)
        compiled(p, q, acc_out, acc_in)

        np.testing.assert_allclose(
            acc_out[0], numpy_reduce(p, op_outer), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            acc_in[0], numpy_reduce(q, op_inner), rtol=1e-4, atol=1e-4
        )

    _sib_scns = make_sibling_scenarios(ws)

    @pytest.mark.parametrize("scn", _sib_scns, ids=[s["id"] for s in _sib_scns])
    def test_sibling_nest(scn, tmp_path):
        sdfg, R, C, host_names = build_sibling_nest(
            backend, scn["parent"], scn["sib"], scn["containers"], scn["siblings"]
        )
        compiled = _compile(backend, sdfg, tmp_path / "sibling_nest")

        containers = scn["containers"]
        ins = {}
        args = []
        seed = 50
        for name in host_names:
            role, nel = containers[name]
            if role == "in":
                arr = _rng_input("add", nel, seed)
                seed += 1
                ins[name] = arr
            elif role == "acc":  # reduction accumulator: identity for 'add'
                arr = np.full(nel, 0.0, dtype=np.float32)
            else:  # 'out'
                arr = np.zeros(nel, dtype=np.float32)
            args.append(arr)

        compiled(*args)
        argmap = dict(zip(host_names, args))

        for name, expected in scn["ref"](ins, R, C).items():
            np.testing.assert_allclose(
                argmap[name].ravel(),
                np.asarray(expected, dtype=np.float32).ravel(),
                rtol=1e-4,
                atol=1e-4,
            )

    namespace.update(
        test_offload_map=test_offload_map,
        test_offload_reduce=test_offload_reduce,
        test_map_nest=test_map_nest,
        test_reduce_nest=test_reduce_nest,
        test_multi_reduction_one_node=test_multi_reduction_one_node,
        test_reduce_different_vars=test_reduce_different_vars,
        test_sibling_nest=test_sibling_nest,
    )
