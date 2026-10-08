"""ROCm e2e for wave-granular X_BLOCK maps (``lanes`` = wavefront size).

Each iteration of the wave map is executed by one whole wavefront, so the block
launches ``parallel_size * lanes`` threads and the map index is
``threadIdx.x / lanes``. The kernel copies ``out[4g + 2w + c] = A[4g + 2w + c]``;
a thread-granular mis-mapping would write far outside ``out[:16]`` and corrupt the
sentinel region.
"""

import numpy as np
import pytest

from docc.sdfg import (
    BufferLifecycle,
    DataTransferDirection,
    Pointer,
    PrimitiveType,
    RocmArch,
    Scalar,
    ScheduleType,
    StorageType,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

pytestmark = pytest.mark.rocm()

FLOAT_BYTES = 4
SIZE = 1024
WRITTEN = 16  # 4 grid x 2 waves x 2 cols


def _wavefront_size() -> int:
    return 32 if RocmArch.current_name().startswith("gfx1") else 64


def _build(lanes):
    f = Scalar(PrimitiveType.Float)
    host = Pointer(f)
    dev = Pointer(f, StorageType.AMD_Generic())
    i32 = Scalar(PrimitiveType.Int32)

    b = StructuredSDFGBuilder("wave_map_rocm")
    b.add_container("A", host, is_argument=True)
    b.add_container("out", host, is_argument=True)
    b.add_container("dA", dev)
    b.add_container("dout", dev)
    for v in ("g", "w", "c"):
        b.add_container(v, i32)

    nbytes = f"{SIZE} * {FLOAT_BYTES}"
    b.add_rocm_offloading_block(
        "A", "dA", DataTransferDirection.H2D, BufferLifecycle.ALLOC, dev, nbytes
    )
    b.add_rocm_offloading_block(
        "out", "dout", DataTransferDirection.H2D, BufferLifecycle.ALLOC, dev, nbytes
    )

    b.begin_map("g", "0", "4", "1", ScheduleType.rocm_offload(TargetLevel.X_GRID, 4))
    b.begin_map(
        "w",
        "0",
        "2",
        "1",
        ScheduleType.rocm_offload(TargetLevel.X_BLOCK, 2, lanes=lanes),
    )
    b.begin_map("c", "0", "2", "1", ScheduleType.rocm_offload(TargetLevel.Y_BLOCK, 2))
    blk = b.add_block()
    a = b.add_access(blk, "dA")
    o = b.add_access(blk, "dout")
    t = b.add_tasklet(blk, TaskletCode.assign, ["_in"], ["_out"])
    idx = "4*g + 2*w + c"
    b.add_memlet(blk, a, "", t, "_in", idx, dev)
    b.add_memlet(blk, t, "_out", o, "", idx, dev)
    b.end_map()
    b.end_map()
    b.end_map()

    b.add_rocm_offloading_block(
        "out", "dout", DataTransferDirection.D2H, BufferLifecycle.FREE, dev, nbytes
    )
    b.add_rocm_offloading_block(
        "dA", "dA", DataTransferDirection.NONE, BufferLifecycle.FREE, dev, "0"
    )
    return b


def test_lanes_schedule_property():
    s = ScheduleType.rocm_offload(TargetLevel.X_BLOCK, 2, lanes=64)
    assert s.properties["lanes"] == "64"
    with pytest.raises(Exception):
        ScheduleType.rocm_offload(TargetLevel.Y_BLOCK, 2, lanes=64)


def test_wave_map_executes(tmp_path):
    lanes = _wavefront_size()
    sdfg = _build(lanes).move()
    sdfg.validate()
    lib_path = sdfg._compile(str(tmp_path), "rocm")

    generated = "\n".join(p.read_text() for p in tmp_path.rglob("*.cpp"))
    assert f"(threadIdx.x / {lanes})" in generated
    assert f"dim3((int)({2 * lanes}), (int)(2), (int)(1))" in generated

    rng = np.random.default_rng(0)
    A = rng.standard_normal(SIZE).astype(np.float32)
    out = np.full(SIZE, -1.0, dtype=np.float32)
    CompiledSDFG(lib_path, sdfg)(A, out)

    np.testing.assert_array_equal(out[:WRITTEN], A[:WRITTEN])
    np.testing.assert_array_equal(out[WRITTEN:], -1.0)
