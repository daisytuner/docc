"""
Integration test for the CUDA offload dispatchers.

These tests verify the low-level CUDA offload code generators
(``CUDAOffloadMapDispatcher`` and ``CUDAOffloadReduceDispatcher``) in
isolation, WITHOUT running the scheduler or any transformations.

Each test builds an already-offloaded SDFG entirely through the
``StructuredSDFGBuilder`` API -- host argument containers, ``NV_Generic``
device transients, explicit CUDA data-offloading blocks (alloc / H2D / D2H /
free) and offloaded ``Map`` / ``Reduce`` nodes carrying a
``ScheduleType.cuda_offload(target_level, parallel_size)`` schedule -- then
compiles it for CUDA, runs it on the GPU and compares against NumPy.

The parallel size is deliberately varied between:
  * exact-fit  -- ``parallel_size`` divides the iteration count, and
  * ragged     -- ``parallel_size`` does not divide it (non-power-of-two),
so the grid-stride coverage loop and its boundary guard are both exercised.
"""

from pathlib import Path

import numpy as np
import pytest

from docc.sdfg import (
    BufferLifecycle,
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

pytestmark = pytest.mark.cuda()

FLOAT_BYTES = 4

# TaskletCode used inside the body for each supported element-wise / reduction op.
_TASKLET = {
    "add": TaskletCode.fp_add,
    "mul": TaskletCode.fp_mul,
}


# ---------------------------------------------------------------------------
# NumPy references
# ---------------------------------------------------------------------------
def numpy_elementwise(a, b, op):
    if op == "add":
        return a + b
    if op == "mul":
        return a * b
    raise ValueError(op)


def numpy_reduce(a, op):
    if op == "add":
        return a.sum()
    if op == "mul":
        return a.prod()
    raise ValueError(op)


# ---------------------------------------------------------------------------
# SDFG builders
# ---------------------------------------------------------------------------
def build_offloaded_elementwise(n, op, target_level, parallel_size):
    """Build ``C[i] = A[i] <op> B[i]`` as a single offloaded Map.

    The Map is the outermost (and only) offloaded loop, so the dispatcher
    emits the ``__global__`` kernel plus its ``<<<>>>`` launch directly.
    """
    builder = StructuredSDFGBuilder(f"offload_map_{op}")

    f = Scalar(PrimitiveType.Float)
    host_ptr = Pointer(f)
    dev_ptr = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    for name in ("A", "B", "C"):
        builder.add_container(name, host_ptr, is_argument=True)
    for name in ("__daisy_cuda_A", "__daisy_cuda_B", "__daisy_cuda_C"):
        builder.add_container(name, dev_ptr, is_argument=False)
    builder.add_container("i", i32, is_argument=False)

    nbytes = f"{n} * {FLOAT_BYTES}"

    for name in ("__daisy_cuda_A", "__daisy_cuda_B", "__daisy_cuda_C"):
        builder.add_cuda_offloading_block(
            name,
            name,
            DataTransferDirection.NONE,
            BufferLifecycle.ALLOC,
            dev_ptr,
            nbytes,
        )
    for host, dev in (("A", "__daisy_cuda_A"), ("B", "__daisy_cuda_B")):
        builder.add_cuda_offloading_block(
            host,
            dev,
            DataTransferDirection.H2D,
            BufferLifecycle.NO_CHANGE,
            dev_ptr,
            nbytes,
        )

    builder.begin_map(
        "i", "0", str(n), "1", ScheduleType.cuda_offload(target_level, parallel_size)
    )
    blk = builder.add_block()
    a = builder.add_access(blk, "__daisy_cuda_A")
    b = builder.add_access(blk, "__daisy_cuda_B")
    c = builder.add_access(blk, "__daisy_cuda_C")
    t = builder.add_tasklet(blk, _TASKLET[op], ["_in1", "_in2"], ["_out"])
    builder.add_memlet(blk, a, "", t, "_in1", "i")
    builder.add_memlet(blk, b, "", t, "_in2", "i")
    builder.add_memlet(blk, t, "_out", c, "", "i")
    builder.end_map()

    builder.add_cuda_offloading_block(
        "C",
        "__daisy_cuda_C",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes,
    )
    for name in ("__daisy_cuda_A", "__daisy_cuda_B", "__daisy_cuda_C"):
        builder.add_cuda_offloading_block(
            name, name, DataTransferDirection.NONE, BufferLifecycle.FREE, dev_ptr, "0"
        )

    return builder.move()


def build_offloaded_reduction(n, op, target_level, parallel_size):
    """Build ``acc[0] = <op>_j A[j]`` as a single offloaded Reduce.

    The device accumulator is zero-/identity-initialised on the host and
    copied H2D before the kernel, since a grid-level reduction combines into
    the global slot with an atomic / CAS.
    """
    builder = StructuredSDFGBuilder(f"offload_reduce_{op}")

    f = Scalar(PrimitiveType.Float)
    host_ptr = Pointer(f)
    dev_ptr = Pointer(f, StorageType.NV_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    for name in ("A", "acc"):
        builder.add_container(name, host_ptr, is_argument=True)
    for name in ("__daisy_cuda_A", "__daisy_cuda_acc"):
        builder.add_container(name, dev_ptr, is_argument=False)
    builder.add_container("j", i32, is_argument=False)

    nbytes_a = f"{n} * {FLOAT_BYTES}"
    nbytes_acc = f"{FLOAT_BYTES}"

    builder.add_cuda_offloading_block(
        "__daisy_cuda_A",
        "__daisy_cuda_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes_a,
    )
    builder.add_cuda_offloading_block(
        "__daisy_cuda_acc",
        "__daisy_cuda_acc",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        dev_ptr,
        nbytes_acc,
    )
    builder.add_cuda_offloading_block(
        "A",
        "__daisy_cuda_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_a,
    )
    builder.add_cuda_offloading_block(
        "acc",
        "__daisy_cuda_acc",
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
        [(op, "__daisy_cuda_acc")],
        ScheduleType.cuda_offload(target_level, parallel_size),
    )
    blk = builder.add_block()
    a = builder.add_access(blk, "__daisy_cuda_A")
    acc_in = builder.add_access(blk, "__daisy_cuda_acc")
    acc_out = builder.add_access(blk, "__daisy_cuda_acc")
    t = builder.add_tasklet(blk, _TASKLET[op], ["_in1", "_in2"], ["_out"])
    builder.add_memlet(blk, acc_in, "", t, "_in1", "0")
    builder.add_memlet(blk, a, "", t, "_in2", "j")
    builder.add_memlet(blk, t, "_out", acc_out, "", "0")
    builder.end_reduce()

    builder.add_cuda_offloading_block(
        "acc",
        "__daisy_cuda_acc",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        dev_ptr,
        nbytes_acc,
    )
    for name in ("__daisy_cuda_A", "__daisy_cuda_acc"):
        builder.add_cuda_offloading_block(
            name, name, DataTransferDirection.NONE, BufferLifecycle.FREE, dev_ptr, "0"
        )

    return builder.move()


def _compile(sdfg, output_dir: Path):
    sdfg.validate()
    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), "cuda")

    # The offloaded map/reduce must lower through the offload dispatcher
    # (grid-stride coverage loop + kernel launch), not a fallback path.
    cu_sources = list(output_dir.rglob("*.cu"))
    assert cu_sources, "no CUDA source emitted"
    generated = "\n".join(p.read_text() for p in cu_sources)
    assert (
        "__daisy_gpu_coverage_loop_" in generated
    ), "offload dispatcher coverage loop missing from generated CUDA"

    return CompiledSDFG(lib_path, sdfg)


# ---------------------------------------------------------------------------
# MAP dispatcher tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "target_level",
    [TargetLevel.X_GRID, TargetLevel.X_BLOCK],
    ids=["grid", "block"],
)
@pytest.mark.parametrize("op", ["add", "mul"])
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
@pytest.mark.cuda()
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
@pytest.mark.parametrize("op", ["add", "mul"])
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
@pytest.mark.cuda()
def test_offload_reduce(op, n, parallel_size, tmp_path):
    sdfg = build_offloaded_reduction(n, op, TargetLevel.X_GRID, parallel_size)
    compiled = _compile(sdfg, tmp_path / "reduce")

    rng = np.random.default_rng(1)
    if op == "mul":
        # keep magnitudes near 1 so the product stays numerically stable
        a = (1.0 + 0.01 * rng.standard_normal(n)).astype(np.float32)
    else:
        a = rng.standard_normal(n).astype(np.float32)
    acc = np.zeros(1, dtype=np.float32)
    if op == "mul":
        acc[0] = 1.0  # multiplicative identity

    compiled(a, acc)

    ref = numpy_reduce(a, op)
    np.testing.assert_allclose(acc[0], ref, rtol=1e-4, atol=1e-4)
