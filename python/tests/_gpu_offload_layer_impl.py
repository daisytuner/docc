"""
Backend-agnostic integration tests for offloading tensor (layer) library nodes.

Each test builds an SDFG holding a single tensor library node directly through
the ``StructuredSDFGBuilder`` API (no PyTorch frontend), runs the regular
expand -> simplify -> normalize -> schedule pipeline for the GPU target,
compiles it, runs it on the device and compares against a NumPy reference.

Layers are described declaratively by a :class:`LayerSpec` and collected in
:data:`LAYERS`; adding a layer only requires a new spec, the harness and the
CUDA / ROCm thin suites pick it up automatically via :func:`register`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from docc.sdfg import (
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
    TargetOptions,
    Tensor,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

_PRIMITIVE = {
    np.dtype(np.float32): PrimitiveType.Float,
    np.dtype(np.float64): PrimitiveType.Double,
    np.dtype(np.int32): PrimitiveType.Int32,
    np.dtype(np.int64): PrimitiveType.Int64,
}


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

    ``operands``  -- ``params -> [Operand]`` in host-argument order.
    ``add_node``  -- adds the library node given ``{name: Tensor}`` types.
    ``inputs``    -- ``(params, rng) -> {name: array}`` for non-output operands.
    ``reference`` -- ``(params, inputs) -> {name: array}`` expected outputs.
    ``cases``     -- ``((id, params), ...)`` parametrisation of the layer.
    """

    name: str
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
    return [
        Operand("W", np.float32, (p["vocab"], p["dim"])),
        Operand("I", np.int64, tuple(p["index_shape"])),
        Operand("Y", np.float32, tuple(p["index_shape"]) + (p["dim"],), True),
    ]


def _embedding_add_node(builder, types):
    builder.add_embedding_op("W", types["W"], "I", types["I"], "Y", types["Y"])


def _embedding_inputs(p, rng):
    return {
        "W": rng.standard_normal((p["vocab"], p["dim"])).astype(np.float32),
        "I": rng.integers(0, p["vocab"], size=p["index_shape"], dtype=np.int64),
    }


def _embedding_reference(_p, ins):
    return {"Y": ins["W"][ins["I"]]}


EMBEDDING = LayerSpec(
    name="embedding",
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
    ),
)


LAYERS = (EMBEDDING,)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def build_layer_sdfg(layer, params):
    builder = StructuredSDFGBuilder(f"offload_{layer.name}")
    types = {}
    for op in layer.operands(params):
        element = Scalar(_PRIMITIVE[np.dtype(op.dtype)])
        builder.add_container(op.name, Pointer(element), is_argument=True)
        types[op.name] = Tensor(element, [str(d) for d in op.shape])
    layer.add_node(builder, types)
    return builder.move()


def _offload_and_compile(backend, sdfg, output_dir: Path):
    sdfg.validate()
    opts = TargetOptions(backend.target, "server")
    sdfg.expand(opts)
    sdfg.simplify()
    sdfg.normalize()
    opts.already_normalized = True
    sdfg.schedule(opts)
    sdfg.validate()

    output_dir.mkdir(parents=True, exist_ok=True)
    lib_path = sdfg._compile(str(output_dir), backend.target)

    sources = list(output_dir.rglob(backend.source_glob))
    assert sources, f"no {backend.name} device source emitted"
    generated = "\n".join(p.read_text() for p in sources)
    assert "__global__" in generated, "layer was not offloaded to a device kernel"

    return CompiledSDFG(lib_path, sdfg)


def register(namespace, backend):
    """Define the layer offloading test, bound to ``backend``, into ``namespace``."""

    @pytest.mark.parametrize(
        "layer,params",
        [
            pytest.param(layer, params, id=f"{layer.name}-{case_id}")
            for layer in LAYERS
            for (case_id, params) in layer.cases
        ],
    )
    def test_offload_layer(layer, params, tmp_path):
        sdfg = build_layer_sdfg(layer, params)
        compiled = _offload_and_compile(backend, sdfg, tmp_path / layer.name)

        operands = layer.operands(params)
        ins = layer.inputs(params, np.random.default_rng(0))
        args = {
            op.name: (
                np.zeros(op.shape, dtype=op.dtype) if op.is_output else ins[op.name]
            )
            for op in operands
        }
        compiled(*(args[op.name] for op in operands))

        for name, expected in layer.reference(params, ins).items():
            np.testing.assert_allclose(
                args[name], expected, rtol=layer.rtol, atol=layer.atol
            )

    namespace.update(test_offload_layer=test_offload_layer)
