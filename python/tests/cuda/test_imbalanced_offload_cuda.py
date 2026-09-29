import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    BufferLifecycle,
    CUDAOffloadNestedLoop,
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
M = 1024
BLOCK = 256


def _build_offloaded_kernel():
    """An ``X_GRID`` kernel (one block, explicit device buffers) with two sibling
    passes sharing a kernel-local ``__d_buf``:

        i (X_GRID)
            j (sequential): __d_buf[j] = __d_A[j]        # cooperative producer
            k (sequential): __d_out[k] = __d_buf[M-1-k]  # consumer (foreign entry)

    Both passes start sequential; the test folds ``j`` onto ``X_BLOCK``. Once ``j``
    is cooperative, ``k`` (replicated across the block) reads buffer entries other
    threads wrote, so the fold needs a barrier between the two. Returns the builder
    and the producer map handle.
    """
    f = Scalar(PrimitiveType.Float)
    idx = Scalar(PrimitiveType.Int64)
    dev = Pointer(f, StorageType.NV_Generic())

    b = StructuredSDFGBuilder("imbalanced_coop_map")
    b.add_container("A", Pointer(f), is_argument=True)
    b.add_container("out", Pointer(f), is_argument=True)
    b.add_container("__d_A", dev, is_argument=False)
    b.add_container("__d_out", dev, is_argument=False)
    b.add_container("__d_buf", dev, is_argument=False)
    b.add_container("i", idx)
    b.add_container("j", idx)
    b.add_container("k", idx)

    def off(host_c, dev_c, direction, lifecycle, size):
        b.add_cuda_offloading_block(host_c, dev_c, direction, lifecycle, dev, size)

    off(
        "__d_A",
        "__d_A",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{M} * {FLOAT_BYTES}",
    )
    off(
        "__d_out",
        "__d_out",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{M} * {FLOAT_BYTES}",
    )
    off(
        "__d_buf",
        "__d_buf",
        DataTransferDirection.NONE,
        BufferLifecycle.ALLOC,
        f"{M} * {FLOAT_BYTES}",
    )
    off(
        "A",
        "__d_A",
        DataTransferDirection.H2D,
        BufferLifecycle.NO_CHANGE,
        f"{M} * {FLOAT_BYTES}",
    )

    b.begin_map("i", "0", "1", "1", ScheduleType.cuda_offload(TargetLevel.X_GRID, 1))

    # Producer: sequential for now; the test folds it onto X_BLOCK.
    producer = b.begin_map("j", "0", str(M), "1")
    blk = b.add_block()
    a = b.add_access(blk, "__d_A")
    bd = b.add_access(blk, "__d_buf")
    t = b.add_tasklet(blk, TaskletCode.assign, ["_in"], ["_out"])
    b.add_memlet(blk, a, "", t, "_in", "j", dev)
    b.add_memlet(blk, t, "_out", bd, "", "j", dev)
    b.end_map()

    # Consumer: sequential (replicated across the block), out[k] = buf[M-1-k].
    b.begin_map("k", "0", str(M), "1")
    blk2 = b.add_block()
    bd2 = b.add_access(blk2, "__d_buf")
    o = b.add_access(blk2, "__d_out")
    t2 = b.add_tasklet(blk2, TaskletCode.assign, ["_in"], ["_out"])
    b.add_memlet(blk2, bd2, "", t2, "_in", f"{M} - 1 - k", dev)
    b.add_memlet(blk2, t2, "_out", o, "", "k", dev)
    b.end_map()

    b.end_map()

    off(
        "out",
        "__d_out",
        DataTransferDirection.D2H,
        BufferLifecycle.NO_CHANGE,
        f"{M} * {FLOAT_BYTES}",
    )
    off("__d_A", "__d_A", DataTransferDirection.NONE, BufferLifecycle.FREE, "0")
    off("__d_out", "__d_out", DataTransferDirection.NONE, BufferLifecycle.FREE, "0")
    off("__d_buf", "__d_buf", DataTransferDirection.NONE, BufferLifecycle.FREE, "0")
    return b, producer


def _compile_and_run(sdfg, tmp_path):
    sdfg.validate()
    out_dir = tmp_path / sdfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = sdfg._compile(str(out_dir), "cuda")

    generated = "\n".join(p.read_text() for p in out_dir.rglob("*.cu"))
    n_sync = generated.count("__syncthreads")

    compiled = CompiledSDFG(lib, sdfg)
    A = np.arange(1, M + 1, dtype=np.float32)
    out = np.zeros(M, dtype=np.float32)
    call = [A if name == "A" else out for name in sdfg.arguments]
    res = compiled(*call)
    out = np.asarray(res if isinstance(res, np.ndarray) else out).reshape(M)
    return n_sync, out


def _outermost_map_body_kinds(sdfg):
    am = AnalysisManager(sdfg)
    la = am.loop_analysis()
    outer = max(la.outermost_loops(), key=lambda l: la.loop_info(l).max_depth)
    body = outer.body
    return [type(body[i]).__name__ for i in range(len(body))]


def test_imbalanced_cooperative_map_fold_inserts_barrier(tmp_path):
    """Folding the cooperative producer onto ``X_BLOCK`` while its consumer stays
    sequential is accepted (a block-level dependency is not a dead-end); ``apply``
    inserts a ``BarrierLocalNode`` between them, so the kernel is
    correct-by-construction (``out == reverse(A)``) instead of racing on ``__d_buf``.
    """
    builder, producer = _build_offloaded_kernel()

    am = AnalysisManager(builder)
    fold = CUDAOffloadNestedLoop(producer, TargetLevel.X_BLOCK, BLOCK)
    assert fold.can_be_applied(builder, am) is True  # barrierable => not a dead-end
    fold.apply(builder, am)

    sdfg = builder.move()

    # A barrier Block now sits between the producer map and the consumer map.
    assert _outermost_map_body_kinds(sdfg) == ["Map", "Block", "Map"]

    # ...and that inserted __syncthreads makes the result correct.
    n_sync, out = _compile_and_run(sdfg, tmp_path)
    assert n_sync == 1
    np.testing.assert_allclose(
        out, np.arange(1, M + 1, dtype=np.float32)[::-1], rtol=1e-6, atol=1e-6
    )
