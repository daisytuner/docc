"""ROCm e2e: split-K GEMV ``y[j] = sum_k W[j, k] * x[k]`` scheduled like vLLM's wvSplitK.

Starts from the plain sequential nest and applies the decode-GEMV schedule: rows on the grid
(optionally ``WV`` rows per block on ``Y_BLOCK``), the K reduction split over ``BLOCK`` lanes on
``X_BLOCK`` with a contiguous ``CHUNK`` per lane, whose weight slice is staged in registers and
vectorized. ``stage_x`` additionally stages ``x`` once per block in LDS. Results are checked
against an fp64 reference.
"""

import ml_dtypes
import numpy as np
import pytest

from docc.sdfg import (
    AnalysisManager,
    Block,
    LocalStorage,
    LoopTiling,
    Pointer,
    PrimitiveType,
    ROCMOffloadNestedLoop,
    ROCMOffloadTransform,
    Scalar,
    Sequence,
    StructuredLoop,
    StructuredSDFGBuilder,
    TargetLevel,
    TaskletCode,
    TileStreamingCopy,
    TileVectorizer,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

pytestmark = pytest.mark.rocm()

DTYPES = {
    "f32": (PrimitiveType.Float, np.float32, 4),
    "bf16": (PrimitiveType.BFloat, ml_dtypes.bfloat16, 8),
}


def _loop(builder, indvar):
    loop = AnalysisManager(builder).loop_analysis().find_loop_by_indvar(indvar)
    assert loop is not None, indvar
    return loop


def _find_access(seq, container):
    for i in range(len(seq)):
        child = seq[i]
        if isinstance(child, Block):
            for node in child.dataflow.reads:
                # Offloading renames the host argument to a prefixed device copy.
                if node.data == container or node.data.endswith("_" + container):
                    return node
        elif isinstance(child, StructuredLoop):
            found = _find_access(child.body, container)
            if found is not None:
                return found
        elif isinstance(child, Sequence):
            found = _find_access(child, container)
            if found is not None:
                return found
    return None


def _apply(builder, xform):
    am = AnalysisManager(builder)
    assert xform.can_be_applied(builder, am), repr(xform)
    xform.apply(builder, am)


def _build(n, k, primitive):
    builder = StructuredSDFGBuilder("splitk_gemv")
    ptr = Pointer(Scalar(primitive))
    for name in ("W", "x", "y"):
        builder.add_container(name, ptr, is_argument=True)
    for name in ("j", "k"):
        builder.add_container(name, Scalar(PrimitiveType.Int32))

    builder.begin_map("j", "0", str(n), "1")
    builder.begin_reduce("k", "0", str(k), "1", [("add", "y")])
    blk = builder.add_block()
    w = builder.add_access(blk, "W")
    x = builder.add_access(blk, "x")
    y_in = builder.add_access(blk, "y")
    y_out = builder.add_access(blk, "y")
    fma = builder.add_tasklet(
        blk, TaskletCode.fp_fma, ["_in1", "_in2", "_in3"], ["_out"]
    )
    builder.add_memlet(blk, w, "", fma, "_in1", f"j*{k} + k")
    builder.add_memlet(blk, x, "", fma, "_in2", "k")
    builder.add_memlet(blk, y_in, "", fma, "_in3", "j")
    builder.add_memlet(blk, fma, "_out", y_out, "", "j")
    builder.end_reduce()
    builder.end_map()
    return builder


def _schedule(builder, n, wv, block, chunk, stage_x):
    if wv is None:
        grid_loop = "j"
        _apply(
            builder, ROCMOffloadTransform(_loop(builder, "j"), n, TargetLevel.X_GRID)
        )
    else:
        grid_loop = "j_tile0"
        _apply(builder, LoopTiling(_loop(builder, "j"), wv, True))
        _apply(
            builder,
            ROCMOffloadTransform(
                _loop(builder, grid_loop), n // wv, TargetLevel.X_GRID
            ),
        )
        _apply(
            builder, ROCMOffloadNestedLoop(_loop(builder, "j"), TargetLevel.Y_BLOCK, wv)
        )
    _apply(builder, LoopTiling(_loop(builder, "k"), chunk, True))
    _apply(
        builder,
        ROCMOffloadNestedLoop(_loop(builder, "k_tile0"), TargetLevel.X_BLOCK, block),
    )
    if stage_x:
        grid = _loop(builder, grid_loop)
        _apply(builder, LocalStorage(grid, _find_access(grid.body, "x")))
        _apply(builder, TileVectorizer(_loop(builder, grid_loop)))
    panel = _loop(builder, "k")
    _apply(builder, LocalStorage(panel, _find_access(panel.body, "W")))
    _apply(builder, TileVectorizer(_loop(builder, grid_loop)))
    # Each lane's weight slice is read once per launch: load it nontemporally.
    _apply(builder, TileStreamingCopy(_loop(builder, "k_tile0")))


@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize(
    "n,k,wv,block,stage_x",
    [
        (32, 1024, None, 64, False),
        (32, 2048, None, 128, False),
        (32, 1024, 4, 64, False),
        (32, 1024, 4, 64, True),
    ],
    ids=["grid_rows_1wave", "grid_rows_2waves", "block_rows", "block_rows_stage_x"],
)
def test_splitk_gemv(dtype, n, k, wv, block, stage_x, tmp_path):
    primitive, np_dtype, chunk = DTYPES[dtype]
    builder = _build(n, k, primitive)
    _schedule(builder, n, wv, block, chunk, stage_x)
    sdfg = builder.move()
    sdfg.validate()
    compiled = CompiledSDFG(sdfg._compile(str(tmp_path), "rocm"), sdfg)

    rng = np.random.default_rng(0)
    W = rng.standard_normal((n, k)).astype(np_dtype)
    x = rng.standard_normal(k).astype(np_dtype)
    y = np.zeros(n, dtype=np_dtype)
    compiled(W.reshape(-1), x, y)

    ref = W.astype(np.float64) @ x.astype(np.float64)
    np.testing.assert_allclose(
        y.astype(np.float64), ref, rtol=2e-2, atol=2e-2 * np.sqrt(k)
    )
