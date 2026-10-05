"""
Backend-agnostic integration tests for offloading tensor (layer) library nodes.

Each test builds an SDFG holding a single tensor library node directly through
the ``StructuredSDFGBuilder`` API (no PyTorch frontend), runs the regular
expand -> simplify -> normalize -> schedule pipeline for the GPU target,
compiles it, runs it on the device and compares against a NumPy reference.

The library node must survive the pipeline unexpanded: the target expansion
marks it ``<Target>WithTransfers`` and post-scheduling extracts its transfers
into explicit offloading blocks (``<Target>WithoutTransfers``), so it reaches
code generation through its dedicated GPU library node dispatcher.

Layers are described declaratively by a :class:`LayerSpec` and collected in
:data:`LAYERS`; adding a layer only requires a new spec, the harness and the
CUDA / ROCm thin suites pick it up automatically via :func:`register`.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ml_dtypes
import numpy as np
import pytest

from docc.sdfg import (
    Pointer,
    PrimitiveType,
    Scalar,
    ScheduleType,
    StructuredSDFGBuilder,
    TargetOptions,
    TaskletCode,
    Tensor,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

_PRIMITIVE = {
    np.dtype(np.float16): PrimitiveType.Half,
    np.dtype(ml_dtypes.bfloat16): PrimitiveType.BFloat,
    np.dtype(np.float32): PrimitiveType.Float,
    np.dtype(np.float64): PrimitiveType.Double,
    np.dtype(np.int32): PrimitiveType.Int32,
    np.dtype(np.int64): PrimitiveType.Int64,
}

# Implementation-type prefix of each backend's library node dispatchers.
_IMPL_PREFIX = {"cuda": "CUDA", "rocm": "ROCM"}


@dataclass(frozen=True)
class Operand:
    """One pointer argument of the layer, declared in call order."""

    name: str
    dtype: type
    shape: tuple
    is_output: bool = False


@dataclass(frozen=True)
class LayerSpec:
    """Everything needed to build, feed and check one tensor library node.

    ``code``          -- library node code that must survive to code generation.
    ``kernel_marker`` -- substring identifying the dispatcher's device kernel.
    ``operands``      -- ``params -> [Operand]`` in host-argument order.
    ``add_node``      -- adds the library node given ``{name: Tensor}`` types.
    ``inputs``        -- ``(params, rng) -> {name: array}`` for non-output operands.
    ``reference``     -- ``(params, inputs) -> {name: array}`` expected outputs.
    ``cases``         -- ``((id, params), ...)`` parametrisation of the layer.
    ``rtol``/``atol`` -- tolerances; both zero demands bit-exact outputs.
    """

    name: str
    code: str
    kernel_marker: str
    operands: Callable[[dict], list]
    add_node: Callable[[StructuredSDFGBuilder, dict], None]
    inputs: Callable[[dict, np.random.Generator], dict]
    reference: Callable[[dict, dict], dict]
    cases: tuple
    rtol: float = 0.0
    atol: float = 0.0


# ---------------------------------------------------------------------------
# Embedding: Y[b..., d] = W[I[b...], d]
# ---------------------------------------------------------------------------
def _embedding_operands(p):
    w_dtype = p.get("w_dtype", np.float32)
    i_dtype = p.get("i_dtype", np.int64)
    return [
        Operand("W", w_dtype, (p["vocab"], p["dim"])),
        Operand("I", i_dtype, tuple(p["index_shape"])),
        Operand("Y", w_dtype, tuple(p["index_shape"]) + (p["dim"],), True),
    ]


def _embedding_add_node(builder, types):
    builder.add_embedding_op("W", types["W"], "I", types["I"], "Y", types["Y"])


def _embedding_inputs(p, rng):
    w_dtype = p.get("w_dtype", np.float32)
    i_dtype = p.get("i_dtype", np.int64)
    return {
        "W": rng.standard_normal((p["vocab"], p["dim"])).astype(w_dtype),
        "I": rng.integers(0, p["vocab"], size=p["index_shape"], dtype=i_dtype),
    }


def _embedding_reference(_p, ins):
    return {"Y": ins["W"][ins["I"]]}


EMBEDDING = LayerSpec(
    name="embedding",
    code="ml::Embedding",
    kernel_marker="embedding_kernel_",
    operands=_embedding_operands,
    add_node=_embedding_add_node,
    inputs=_embedding_inputs,
    reference=_embedding_reference,
    cases=(
        ("1d_v10_d4", dict(vocab=10, dim=4, index_shape=(4,))),
        ("1d_v1000_d128", dict(vocab=1000, dim=128, index_shape=(256,))),
        ("2d_v512_d64", dict(vocab=512, dim=64, index_shape=(8, 32))),
        ("3d_ragged_v100_d33", dict(vocab=100, dim=33, index_shape=(2, 3, 17))),
        # tiny vocabulary forces many repeated rows to be gathered concurrently
        ("repeated_v3_d256", dict(vocab=3, dim=256, index_shape=(500,))),
        ("dim1_v64", dict(vocab=64, dim=1, index_shape=(1000,))),
        ("single_index", dict(vocab=7, dim=16, index_shape=(1,))),
        ("f64_v50_d24", dict(vocab=50, dim=24, index_shape=(64,), w_dtype=np.float64)),
        ("f16_v50_d24", dict(vocab=50, dim=24, index_shape=(64,), w_dtype=np.float16)),
        (
            "bf16_v50_d24",
            dict(vocab=50, dim=24, index_shape=(64,), w_dtype=ml_dtypes.bfloat16),
        ),
        (
            "i32_v1000_d32",
            dict(vocab=1000, dim=32, index_shape=(4, 16), i_dtype=np.int32),
        ),
        (
            "f16_i32_ragged",
            dict(
                vocab=33,
                dim=7,
                index_shape=(3, 5),
                w_dtype=np.float16,
                i_dtype=np.int32,
            ),
        ),
    ),
)


LAYERS = (EMBEDDING,)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def build_layer_sdfg(layer, params, consume_output=False):
    """Build the layer SDFG; returns ``(sdfg, operands)`` in host-argument order.

    With ``consume_output`` a sequential map ``Z = Y + Y`` over the first output
    follows the node, so the scheduler offloads a loop next to the library node.
    """
    builder = StructuredSDFGBuilder(f"offload_{layer.name}")
    operands = list(layer.operands(params))
    types = {}
    for op in operands:
        element = Scalar(_PRIMITIVE[np.dtype(op.dtype)])
        builder.add_container(op.name, Pointer(element), is_argument=True)
        types[op.name] = Tensor(element, [str(d) for d in op.shape])
    layer.add_node(builder, types)

    if consume_output:
        out = next(op for op in operands if op.is_output)
        consumer = Operand("Z", out.dtype, out.shape, True)
        builder.add_container(
            consumer.name,
            Pointer(Scalar(_PRIMITIVE[np.dtype(out.dtype)])),
            is_argument=True,
        )
        builder.add_container("_e", Scalar(PrimitiveType.Int64), is_argument=False)
        builder.begin_map(
            "_e", "0", str(math.prod(out.shape)), "1", ScheduleType.sequential()
        )
        blk = builder.add_block()
        src = builder.add_access(blk, out.name)
        dst = builder.add_access(blk, consumer.name)
        t = builder.add_tasklet(blk, TaskletCode.fp_add, ["_in1", "_in2"], ["_out"])
        builder.add_memlet(blk, src, "", t, "_in1", "_e")
        builder.add_memlet(blk, src, "", t, "_in2", "_e")
        builder.add_memlet(blk, t, "_out", dst, "", "_e")
        builder.end_map()
        operands.append(consumer)

    return builder.move(), operands


def _offload_and_compile(backend, layer, sdfg, output_dir: Path):
    sdfg.validate()
    opts = TargetOptions(backend.target, "server")
    sdfg.expand(opts)
    sdfg.simplify()
    sdfg.normalize()
    opts.already_normalized = True
    sdfg.schedule(opts)
    sdfg.validate()

    impl = _IMPL_PREFIX[backend.name]
    graph = sdfg.to_json()
    assert layer.code in graph, f"{layer.code} was expanded instead of dispatched"
    assert (
        f"{impl}WithoutTransfers" in graph
    ), f"{layer.code} transfers were not extracted during schedule"
    assert (
        f"{impl}WithTransfers" not in graph
    ), f"{layer.code} still carries {impl}WithTransfers after schedule"

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), backend.target)

    sources = list(output_dir.rglob(backend.source_glob))
    assert sources, f"no {backend.name} device source emitted"
    generated = "\n".join(p.read_text() for p in sources)
    assert (
        layer.kernel_marker in generated
    ), f"{layer.code} dispatcher kernel missing from generated device source"

    return CompiledSDFG(lib_path, sdfg)


def _assert_output(actual, expected, rtol, atol):
    expected = np.ascontiguousarray(expected, dtype=actual.dtype)
    if rtol == 0.0 and atol == 0.0:
        bits = f"u{actual.dtype.itemsize}"
        np.testing.assert_array_equal(actual.view(bits), expected.view(bits))
    else:
        np.testing.assert_allclose(
            actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
        )


def _run_layer(backend, layer, params, tmp_path, consume_output):
    sdfg, operands = build_layer_sdfg(layer, params, consume_output)
    compiled = _offload_and_compile(backend, layer, sdfg, tmp_path / layer.name)

    ins = layer.inputs(params, np.random.default_rng(0))
    args = {
        op.name: (np.zeros(op.shape, dtype=op.dtype) if op.is_output else ins[op.name])
        for op in operands
    }
    compiled(*(args[op.name] for op in operands))

    expected = layer.reference(params, ins)
    if consume_output:
        out = next(op for op in operands if op.is_output)
        expected["Z"] = expected[out.name] + expected[out.name]
    for name, values in expected.items():
        _assert_output(args[name], values, layer.rtol, layer.atol)


def _float32_outputs(layer, params):
    return all(
        np.dtype(op.dtype) == np.float32
        for op in layer.operands(params)
        if op.is_output
    )


def register(namespace, backend):
    """Define the layer offloading tests, bound to ``backend``, into ``namespace``."""

    @pytest.mark.parametrize(
        "layer,params",
        [
            pytest.param(layer, params, id=f"{layer.name}-{case_id}")
            for layer in LAYERS
            for (case_id, params) in layer.cases
        ],
    )
    def test_offload_layer(layer, params, tmp_path):
        _run_layer(backend, layer, params, tmp_path, consume_output=False)

    # The consumer's fp_add tasklet is only exercised on float32 outputs.
    @pytest.mark.parametrize(
        "layer,params",
        [
            pytest.param(layer, params, id=f"{layer.name}-{case_id}")
            for layer in LAYERS
            for (case_id, params) in layer.cases
            if _float32_outputs(layer, params)
        ],
    )
    def test_offload_layer_with_consumer(layer, params, tmp_path):
        _run_layer(backend, layer, params, tmp_path, consume_output=True)

    namespace.update(
        test_offload_layer=test_offload_layer,
        test_offload_layer_with_consumer=test_offload_layer_with_consumer,
    )
