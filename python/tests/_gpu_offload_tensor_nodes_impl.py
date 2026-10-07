"""
Backend-agnostic check that every tensor library node can be offloaded to the GPU.

Each case builds an SDFG holding a single tensor library node directly through the
``StructuredSDFGBuilder`` API (no PyTorch frontend), runs the regular
expand -> simplify -> normalize -> schedule pipeline for the GPU target, compiles
it, runs it and compares against a NumPy reference.

A case passes only if nothing is computed on the host after scheduling: every loop
is GPU-scheduled (or nested in one) and every remaining library node is either a
data transfer or has a GPU implementation type. Failures name the reason:
``NOT_OFFLOADED``, ``PIPELINE_ERROR`` or ``WRONG_RESULT``.

Cases run in a forked child so a native abort only fails that case.
"""

import json
import math
import multiprocessing
import queue as queue_module
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import ml_dtypes
import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from docc.sdfg import (
    CMathFunction,
    Pointer,
    PrimitiveType,
    Scalar,
    StructuredSDFGBuilder,
    TargetOptions,
    TaskletCode,
    Tensor,
)
from docc.compiler.compiled_sdfg import CompiledSDFG

PRIMITIVE = {
    np.dtype(np.bool_): PrimitiveType.Bool,
    np.dtype(np.float16): PrimitiveType.Half,
    np.dtype(ml_dtypes.bfloat16): PrimitiveType.BFloat,
    np.dtype(np.float32): PrimitiveType.Float,
    np.dtype(np.float64): PrimitiveType.Double,
    np.dtype(np.int32): PrimitiveType.Int32,
    np.dtype(np.int64): PrimitiveType.Int64,
}

GPU_SCHEDULES = {"CUDA_Offload", "ROCM_Offload", "CUDA", "ROCM"}
GPU_IMPL_PREFIX = {"cuda": "CUDA", "rocm": "ROCM"}
CASE_TIMEOUT_S = 600

f32, f64, i32, i64, b8 = np.float32, np.float64, np.int32, np.int64, np.bool_


@dataclass(frozen=True)
class Operand:
    """One pointer argument of the node, declared in call order."""

    name: str
    dtype: type
    shape: tuple
    is_output: bool = False


@dataclass(frozen=True)
class NodeSpec:
    """One tensor library node and its cases.

    ``operands``  -- ``params -> [Operand]`` in host-argument order.
    ``add_node``  -- ``(builder, {name: Tensor}, params)`` adds the node.
    ``inputs``    -- ``(params, rng) -> {name: array}``; an output listed here starts with that value.
    ``reference`` -- ``(params, inputs) -> {name: array}`` expected outputs.
    ``cases``     -- ``((id, params), ...)``; params may override ``rtol``/``atol``.
    """

    name: str
    operands: Callable[[dict], list]
    add_node: Callable[[StructuredSDFGBuilder, dict, dict], None]
    inputs: Callable[[dict, np.random.Generator], dict]
    reference: Callable[[dict, dict], dict]
    cases: tuple
    rtol: float = 1e-5
    atol: float = 1e-6


def _scalar(dtype):
    return Scalar(PRIMITIVE[np.dtype(dtype)])


def _tensor(dtype, shape):
    return Tensor(_scalar(dtype), [str(d) for d in shape])


def _random(rng, shape, dtype, positive=False):
    if np.dtype(dtype) == np.bool_:
        return rng.integers(0, 2, size=shape).astype(np.bool_)
    if np.issubdtype(dtype, np.integer):
        return rng.integers(1 if positive else -10, 10, size=shape).astype(dtype)
    if positive:
        return rng.uniform(0.5, 2.0, size=shape).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


def _assert_output(actual, expected, rtol, atol):
    expected = np.ascontiguousarray(expected, dtype=actual.dtype)
    if rtol == 0.0 and atol == 0.0:
        bits = f"u{actual.dtype.itemsize}"
        np.testing.assert_array_equal(actual.view(bits), expected.view(bits))
    else:
        np.testing.assert_allclose(
            actual.astype(np.float64), expected.astype(np.float64), rtol=rtol, atol=atol
        )


# ---------------------------------------------------------------------------
# Element-wise
# ---------------------------------------------------------------------------
def _binary(name, op_type, ref, cases, positive=False, add=None, rtol=1e-5):
    def operands(p):
        s, d = p["shape"], p["dtype"]
        return [Operand("A", d, s), Operand("B", d, s), Operand("C", d, s, True)]

    def add_node(b, t, p):
        if add is None:
            b.add_elementwise_op(op_type, "A", t["A"], "B", t["B"], "C", t["C"])
        else:
            add(b, t, p)

    def inputs(p, rng):
        return {n: _random(rng, p["shape"], p["dtype"], positive) for n in ("A", "B")}

    def reference(p, ins):
        return {"C": ref(ins["A"], ins["B"]).astype(p["dtype"])}

    return NodeSpec(name, operands, add_node, inputs, reference, cases, rtol=rtol)


def _unary(name, add, ref, cases, positive=False, rtol=1e-5, atol=1e-6):
    def operands(p):
        s = p["shape"]
        return [
            Operand("X", p["dtype"], s),
            Operand("Y", p.get("out_dtype", p["dtype"]), s, True),
        ]

    def inputs(p, rng):
        return {"X": _random(rng, p["shape"], p["dtype"], positive)}

    def reference(p, ins):
        return {"Y": np.asarray(ref(ins["X"])).astype(p.get("out_dtype", p["dtype"]))}

    return NodeSpec(name, operands, add, inputs, reference, cases, rtol=rtol, atol=atol)


def _unary_op(op_type):
    return lambda b, t, p: b.add_elementwise_unary_op(op_type, "X", t["X"], "Y", t["Y"])


_erf = np.vectorize(math.erf)
FLOAT_CASES = (
    ("2d_f32", dict(shape=(16, 32), dtype=f32)),
    ("3d_f64", dict(shape=(2, 3, 17), dtype=f64)),
)
INT_CASE = ("2d_i32", dict(shape=(16, 32), dtype=i32))

BINARY = (
    _binary("add", "add", np.add, FLOAT_CASES + (INT_CASE,)),
    _binary("sub", "sub", np.subtract, FLOAT_CASES + (INT_CASE,)),
    _binary("mul", "mul", np.multiply, FLOAT_CASES + (INT_CASE,)),
    _binary("div", "div", np.divide, FLOAT_CASES, positive=True),
    _binary("pow", "pow", np.power, FLOAT_CASES, positive=True, rtol=1e-4),
    _binary("maximum", "max", np.maximum, FLOAT_CASES + (INT_CASE,)),
    _binary("minimum", "min", np.minimum, FLOAT_CASES + (INT_CASE,)),
    _binary(
        "cmath_fmax",
        None,
        np.fmax,
        FLOAT_CASES,
        add=lambda b, t, p: b.add_elementwise_cmath_op(
            CMathFunction.fmax, "A", t["A"], "B", t["B"], "C", t["C"]
        ),
    ),
)

UNARY = (
    _unary("abs", _unary_op("abs"), np.abs, FLOAT_CASES + (INT_CASE,)),
    _unary("sqrt", _unary_op("sqrt"), np.sqrt, FLOAT_CASES, positive=True),
    _unary(
        "rsqrt",
        _unary_op("rsqrt"),
        lambda x: 1 / np.sqrt(x),
        FLOAT_CASES,
        positive=True,
        rtol=1e-4,
    ),
    _unary("tanh", _unary_op("tanh"), np.tanh, FLOAT_CASES, rtol=1e-4),
    _unary("exp", _unary_op("exp"), np.exp, FLOAT_CASES, rtol=1e-4),
    _unary(
        "sigmoid",
        _unary_op("sigmoid"),
        lambda x: 1 / (1 + np.exp(-x)),
        FLOAT_CASES,
        rtol=1e-4,
    ),
    _unary(
        "logical_not",
        _unary_op("logical_not"),
        np.logical_not,
        (("2d_bool", dict(shape=(16, 32), dtype=b8)),),
    ),
    _unary(
        "relu",
        lambda b, t, p: b.add_relu("X", t["X"], "Y", t["Y"]),
        lambda x: np.maximum(x, 0),
        FLOAT_CASES,
    ),
    _unary(
        "gelu",
        lambda b, t, p: b.add_gelu("X", t["X"], "Y", t["Y"], p["tanh"]),
        lambda x: 0.5 * x * (1 + _erf(x / math.sqrt(2))),
        (("exact_f32", dict(shape=(16, 32), dtype=f32, tanh=False)),),
        rtol=1e-4,
        atol=1e-5,
    ),
    _unary(
        "gelu_tanh",
        lambda b, t, p: b.add_gelu("X", t["X"], "Y", t["Y"], p["tanh"]),
        lambda x: 0.5
        * x
        * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))),
        (("tanh_f32", dict(shape=(16, 32), dtype=f32, tanh=True)),),
        rtol=1e-4,
        atol=1e-5,
    ),
    _unary(
        "elu",
        lambda b, t, p: b.add_elu("X", t["X"], "1.0", _scalar(p["dtype"]), "Y", t["Y"]),
        lambda x: np.where(x > 0, x, np.exp(x) - 1),
        FLOAT_CASES,
        rtol=1e-4,
    ),
    _unary(
        "erf",
        lambda b, t, p: b.add_erf("X", t["X"], "Y", t["Y"]),
        _erf,
        FLOAT_CASES,
        rtol=1e-4,
        atol=1e-5,
    ),
    _unary(
        "hard_sigmoid",
        lambda b, t, p: b.add_hard_sigmoid(
            "X",
            t["X"],
            "0.2",
            _scalar(p["dtype"]),
            "0.5",
            _scalar(p["dtype"]),
            "Y",
            t["Y"],
        ),
        lambda x: np.clip(0.2 * x + 0.5, 0, 1),
        FLOAT_CASES,
    ),
    _unary(
        "leaky_relu",
        lambda b, t, p: b.add_leaky_relu(
            "X", t["X"], "0.01", _scalar(p["dtype"]), "Y", t["Y"]
        ),
        lambda x: np.where(x > 0, x, 0.01 * x),
        FLOAT_CASES,
    ),
    _unary(
        "cmath_sin",
        lambda b, t, p: b.add_elementwise_unary_cmath_op(
            CMathFunction.sin, "X", t["X"], "Y", t["Y"]
        ),
        np.sin,
        FLOAT_CASES,
        rtol=1e-4,
        atol=1e-6,
    ),
    _unary(
        "cast",
        lambda b, t, p: b.add_cast_op("X", t["X"], "Y", t["Y"]),
        lambda x: x,
        (
            ("f32_to_f64", dict(shape=(16, 32), dtype=f32, out_dtype=f64)),
            ("f32_to_i32", dict(shape=(16, 32), dtype=f32, out_dtype=i32)),
            ("i64_to_f32", dict(shape=(16, 32), dtype=i64, out_dtype=f32)),
        ),
    ),
)


def _fma_operands(p):
    s, d = p["shape"], p["dtype"]
    return [
        Operand("A", d, s),
        Operand("B", d, s),
        Operand("C", d, s),
        Operand("Y", d, s, True),
    ]


TASKLET = NodeSpec(
    "tasklet_fma",
    _fma_operands,
    lambda b, t, p: b.add_elementwise_tasklet_op(
        TaskletCode.fp_fma, ["A", "B", "C"], [t["A"], t["B"], t["C"]], "Y", t["Y"]
    ),
    lambda p, rng: {n: _random(rng, p["shape"], p["dtype"]) for n in ("A", "B", "C")},
    lambda p, ins: {"Y": (ins["A"] * ins["B"] + ins["C"]).astype(p["dtype"])},
    FLOAT_CASES,
    rtol=1e-5,
    atol=1e-6,
)

FILL = NodeSpec(
    "fill",
    lambda p: [Operand("Y", p["dtype"], p["shape"], True)],
    lambda b, t, p: b.add_fill_op("2.5", _scalar(p["dtype"]), "Y", t["Y"]),
    lambda p, rng: {},
    lambda p, ins: {"Y": np.full(p["shape"], 2.5, dtype=p["dtype"])},
    FLOAT_CASES,
)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------
def _reduce(name, ref, cases, rtol=1e-4, atol=1e-5):
    def out_shape(p):
        shape, axes = p["shape"], [a % len(p["shape"]) for a in p["axes"]]
        if name == "softmax":
            return shape
        if p["keepdims"]:
            return tuple(1 if i in axes else d for i, d in enumerate(shape))
        return tuple(d for i, d in enumerate(shape) if i not in axes)

    def operands(p):
        return [
            Operand("X", p["dtype"], p["shape"]),
            Operand("Y", p["dtype"], out_shape(p), True),
        ]

    def add_node(b, t, p):
        b.add_reduce_op(name, "X", t["X"], "Y", t["Y"], list(p["axes"]), p["keepdims"])

    def inputs(p, rng):
        return {"X": _random(rng, p["shape"], p["dtype"])}

    def reference(p, ins):
        y = ref(ins["X"].astype(np.float64), tuple(p["axes"]), p["keepdims"])
        return {"Y": np.asarray(y).reshape(out_shape(p)).astype(p["dtype"])}

    return NodeSpec(
        name, operands, add_node, inputs, reference, cases, rtol=rtol, atol=atol
    )


REDUCE_CASES = (
    ("last_axis_f32", dict(shape=(6, 40), axes=(1,), keepdims=False, dtype=f32)),
    ("first_axis_f32", dict(shape=(6, 40), axes=(0,), keepdims=False, dtype=f32)),
    (
        "two_axes_keepdims_f64",
        dict(shape=(4, 5, 6), axes=(1, 2), keepdims=True, dtype=f64),
    ),
)
REDUCE_INT_CASE = (
    "last_axis_i32",
    dict(shape=(6, 40), axes=(1,), keepdims=False, dtype=i32),
)


def _softmax(x, axes, _keepdims):
    e = np.exp(x - x.max(axis=axes, keepdims=True))
    return e / e.sum(axis=axes, keepdims=True)


REDUCE = (
    _reduce(
        "sum",
        lambda x, a, k: x.sum(axis=a, keepdims=k),
        REDUCE_CASES + (REDUCE_INT_CASE,),
    ),
    _reduce(
        "max",
        lambda x, a, k: x.max(axis=a, keepdims=k),
        REDUCE_CASES + (REDUCE_INT_CASE,),
    ),
    _reduce(
        "min",
        lambda x, a, k: x.min(axis=a, keepdims=k),
        REDUCE_CASES + (REDUCE_INT_CASE,),
    ),
    _reduce("mean", lambda x, a, k: x.mean(axis=a, keepdims=k), REDUCE_CASES),
    # ml::Std computes the population (ddof=0) standard deviation.
    _reduce("std", lambda x, a, k: x.std(axis=a, keepdims=k), REDUCE_CASES),
    _reduce(
        "softmax",
        _softmax,
        (
            (
                "last_axis_f32",
                dict(shape=(6, 40), axes=(1,), keepdims=False, dtype=f32),
            ),
            (
                "first_axis_f32",
                dict(shape=(6, 40), axes=(0,), keepdims=False, dtype=f32),
            ),
            (
                "3d_last_axis_f64",
                dict(shape=(2, 3, 17), axes=(2,), keepdims=False, dtype=f64),
            ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def _copy_add(b, t, p):
    x_type = t["X"]
    if p.get("transposed"):
        rows, cols = p["shape"]
        # X holds a (cols, rows) row-major buffer viewed as its (rows, cols) transpose.
        x_type = Tensor(
            _scalar(p["dtype"]), [str(rows), str(cols)], ["1", str(rows)], "0"
        )
    b.add_copy_op("X", x_type, "Y", t["Y"])


COPY = NodeSpec(
    "copy",
    lambda p: [
        Operand(
            "X", p["dtype"], p["shape"][::-1] if p.get("transposed") else p["shape"]
        ),
        Operand("Y", p["dtype"], p["shape"], True),
    ],
    _copy_add,
    lambda p, rng: {
        "X": _random(
            rng, p["shape"][::-1] if p.get("transposed") else p["shape"], p["dtype"]
        )
    },
    lambda p, ins: {"Y": ins["X"].T.copy() if p.get("transposed") else ins["X"].copy()},
    (
        ("contiguous_f32", dict(shape=(16, 24), dtype=f32)),
        ("transposed_f32", dict(shape=(16, 24), dtype=f32, transposed=True)),
        ("contiguous_i64", dict(shape=(16, 24), dtype=i64)),
    ),
    rtol=0.0,
    atol=0.0,
)


def _concat_operands(p):
    out = list(p["shapes"][0])
    out[p["dim"]] = sum(s[p["dim"]] for s in p["shapes"])
    ins = [Operand(f"X{i}", p["dtype"], s) for i, s in enumerate(p["shapes"])]
    return ins + [Operand("Y", p["dtype"], tuple(out), True)]


CONCAT = NodeSpec(
    "concat",
    _concat_operands,
    lambda b, t, p: b.add_concat_op(
        [f"X{i}" for i in range(len(p["shapes"]))],
        [t[f"X{i}"] for i in range(len(p["shapes"]))],
        "Y",
        t["Y"],
        p["dim"],
    ),
    lambda p, rng: {
        f"X{i}": _random(rng, s, p["dtype"]) for i, s in enumerate(p["shapes"])
    },
    lambda p, ins: {
        "Y": np.concatenate(
            [ins[f"X{i}"] for i in range(len(p["shapes"]))], axis=p["dim"]
        )
    },
    (
        ("dim0_f32", dict(shapes=((2, 3), (4, 3)), dim=0, dtype=f32)),
        ("dim1_f32", dict(shapes=((3, 2), (3, 5)), dim=1, dtype=f32)),
        ("three_inputs_f64", dict(shapes=((2, 4), (1, 4), (3, 4)), dim=0, dtype=f64)),
    ),
    rtol=0.0,
    atol=0.0,
)


def _pad_out_shape(p):
    shape, pads = list(p["shape"]), p["pads"]
    for j in range(len(pads) // 2):
        shape[len(shape) - 1 - j] += pads[2 * j] + pads[2 * j + 1]
    return tuple(shape)


# Pads follow torch.nn.functional.pad: pairs start at the last dimension.
def _pad_reference(p, ins):
    pads, x = p["pads"], ins["X"]
    k = len(pads) // 2
    width = [(0, 0)] * (x.ndim - k) + [
        (pads[2 * j], pads[2 * j + 1]) for j in range(k)
    ][::-1]
    return {"Y": np.pad(x, width, constant_values=0.5).astype(p["dtype"])}


CONST_PADDING = NodeSpec(
    "const_padding",
    lambda p: [
        Operand("X", p["dtype"], p["shape"]),
        Operand("Y", p["dtype"], _pad_out_shape(p), True),
    ],
    lambda b, t, p: b.add_const_padding_op(
        "Y",
        t["Y"],
        "X",
        t["X"],
        "0.5",
        _scalar(p["dtype"]),
        [str(v) for v in p["pads"]],
    ),
    lambda p, rng: {"X": _random(rng, p["shape"], p["dtype"])},
    _pad_reference,
    (
        ("2d_f32", dict(shape=(3, 4), pads=(1, 2, 0, 3), dtype=f32)),
        ("3d_last_dim_f64", dict(shape=(2, 3, 4), pads=(2, 1), dtype=f64)),
    ),
    rtol=0.0,
    atol=0.0,
)

BROADCAST = NodeSpec(
    "broadcast",
    lambda p: [
        Operand("X", p["dtype"], p["in"]),
        Operand("Y", p["dtype"], p["out"], True),
    ],
    lambda b, t, p: b.add_broadcast_op(
        "X", t["X"], "Y", t["Y"], [str(d) for d in p["in"]], [str(d) for d in p["out"]]
    ),
    lambda p, rng: {"X": _random(rng, p["in"], p["dtype"])},
    lambda p, ins: {"Y": np.broadcast_to(ins["X"], p["out"]).copy()},
    (
        ("rows_f32", dict(**{"in": (1, 8)}, out=(4, 8), dtype=f32)),
        ("middle_f64", dict(**{"in": (4, 1, 8)}, out=(4, 6, 8), dtype=f64)),
    ),
    rtol=0.0,
    atol=0.0,
)

CONDITIONAL_COPY = NodeSpec(
    "conditional_copy",
    lambda p: [
        Operand("M", b8, p["shape"]),
        Operand("X1", p["dtype"], p["shape"]),
        Operand("X2", p["dtype"], p["shape"]),
        Operand("Y", p["dtype"], p["shape"], True),
    ],
    lambda b, t, p: b.add_conditional_copy_op(
        "M", t["M"], "X1", t["X1"], "X2", t["X2"], "Y", t["Y"]
    ),
    lambda p, rng: {
        "M": _random(rng, p["shape"], b8),
        "X1": _random(rng, p["shape"], p["dtype"]),
        "X2": _random(rng, p["shape"], p["dtype"]),
    },
    lambda p, ins: {"Y": np.where(ins["M"], ins["X1"], ins["X2"])},
    (
        ("2d_f32", dict(shape=(16, 24), dtype=f32)),
        ("3d_i32", dict(shape=(2, 3, 17), dtype=i32)),
    ),
    rtol=0.0,
    atol=0.0,
)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _index_out_shape(p):
    x, pos, n = p["x"], p["positions"], p["count"]
    if all(b - a == 1 for a, b in zip(pos, pos[1:])):
        return x[: pos[0]] + (n,) + x[pos[-1] + 1 :]
    return (n,) + tuple(d for i, d in enumerate(x) if i not in pos)


def _index_reference(p, ins):
    key = [slice(None)] * len(p["x"])
    for i, pos in enumerate(p["positions"]):
        key[pos] = ins[f"I{i}"]
    return {"Y": ins["X"][tuple(key)]}


INDEX = NodeSpec(
    "index",
    lambda p: [Operand("X", p["dtype"], p["x"])]
    + [
        Operand(f"I{i}", p["index_dtype"], (p["count"],))
        for i in range(len(p["positions"]))
    ]
    + [Operand("Y", p["dtype"], _index_out_shape(p), True)],
    lambda b, t, p: b.add_index_op(
        "Y",
        t["Y"],
        "X",
        t["X"],
        [f"I{i}" for i in range(len(p["positions"]))],
        [t[f"I{i}"] for i in range(len(p["positions"]))],
        list(p["positions"]),
    ),
    lambda p, rng: {"X": _random(rng, p["x"], p["dtype"])}
    | {
        f"I{i}": rng.integers(0, p["x"][pos], size=p["count"]).astype(p["index_dtype"])
        for i, pos in enumerate(p["positions"])
    },
    _index_reference,
    (
        (
            "select_dim0_f32",
            dict(x=(8, 5), positions=(0,), count=3, dtype=f32, index_dtype=i64),
        ),
        (
            "select_dim1_f32",
            dict(x=(4, 7), positions=(1,), count=5, dtype=f32, index_dtype=i64),
        ),
        (
            "select_3d_middle_i32_idx",
            dict(x=(2, 6, 3), positions=(1,), count=9, dtype=f32, index_dtype=i32),
        ),
        (
            "adjacent_two_f64",
            dict(x=(5, 6, 7), positions=(1, 2), count=4, dtype=f64, index_dtype=i64),
        ),
        (
            "separated_two_f32",
            dict(x=(5, 6, 7), positions=(0, 2), count=4, dtype=f32, index_dtype=i64),
        ),
    ),
    rtol=0.0,
    atol=0.0,
)

EMBEDDING = NodeSpec(
    "embedding",
    lambda p: [
        Operand("W", p["dtype"], (p["vocab"], p["dim"])),
        Operand("I", p["index_dtype"], p["index_shape"]),
        Operand("Y", p["dtype"], p["index_shape"] + (p["dim"],), True),
    ],
    lambda b, t, p: b.add_embedding_op("W", t["W"], "I", t["I"], "Y", t["Y"]),
    lambda p, rng: {
        "W": _random(rng, (p["vocab"], p["dim"]), p["dtype"]),
        "I": rng.integers(0, p["vocab"], size=p["index_shape"]).astype(
            p["index_dtype"]
        ),
    },
    lambda p, ins: {"Y": ins["W"][ins["I"]]},
    (
        (
            "1d_f32",
            dict(vocab=50, dim=16, index_shape=(12,), dtype=f32, index_dtype=i64),
        ),
        (
            "2d_f64_i32_idx",
            dict(vocab=20, dim=7, index_shape=(3, 5), dtype=f64, index_dtype=i32),
        ),
        (
            "1d_v1000_d128",
            dict(vocab=1000, dim=128, index_shape=(256,), dtype=f32, index_dtype=i64),
        ),
        (
            "2d_v512_d64",
            dict(vocab=512, dim=64, index_shape=(8, 32), dtype=f32, index_dtype=i64),
        ),
        (
            "3d_ragged_v100_d33",
            dict(vocab=100, dim=33, index_shape=(2, 3, 17), dtype=f32, index_dtype=i64),
        ),
        # tiny vocabulary forces many repeated rows to be gathered concurrently
        (
            "repeated_v3_d256",
            dict(vocab=3, dim=256, index_shape=(500,), dtype=f32, index_dtype=i64),
        ),
        (
            "dim1_v64",
            dict(vocab=64, dim=1, index_shape=(1000,), dtype=f32, index_dtype=i64),
        ),
        (
            "single_index",
            dict(vocab=7, dim=16, index_shape=(1,), dtype=f32, index_dtype=i64),
        ),
        (
            "f16_v50_d24",
            dict(
                vocab=50, dim=24, index_shape=(64,), dtype=np.float16, index_dtype=i64
            ),
        ),
        (
            "bf16_v50_d24",
            dict(
                vocab=50,
                dim=24,
                index_shape=(64,),
                dtype=ml_dtypes.bfloat16,
                index_dtype=i64,
            ),
        ),
        (
            "f16_i32_ragged",
            dict(
                vocab=33, dim=7, index_shape=(3, 5), dtype=np.float16, index_dtype=i32
            ),
        ),
    ),
    rtol=0.0,
    atol=0.0,
)


def _renorm_reference(p, ins):
    y = ins["Y"].astype(np.float64)
    for row in np.unique(ins["I"]):
        norm = np.linalg.norm(y[row], ord=2)
        y[row] *= min(1.0, 1.0 / (norm + 1e-7))
    return {"Y": y.astype(p["dtype"])}


EMBEDDING_RENORM = NodeSpec(
    "embedding_renorm",
    lambda p: [
        Operand("Y", p["dtype"], (p["vocab"], p["dim"]), True),
        Operand("W", p["dtype"], (p["vocab"], p["dim"])),
        Operand("I", i64, (p["count"],)),
    ],
    lambda b, t, p: b.add_embedding_renorm_op(
        "Y",
        t["Y"],
        "W",
        t["W"],
        "I",
        t["I"],
        "1.0",
        _scalar(p["dtype"]),
        "2.0",
        _scalar(p["dtype"]),
    ),
    lambda p, rng: (
        lambda w: {
            "W": w,
            "Y": w.copy(),
            "I": rng.integers(0, p["vocab"], size=p["count"]),
        }
    )((rng.standard_normal((p["vocab"], p["dim"])) * 2).astype(p["dtype"])),
    _renorm_reference,
    (
        ("f32", dict(vocab=10, dim=4, count=6, dtype=f32)),
        ("f64", dict(vocab=12, dim=8, count=20, dtype=f64)),
    ),
    rtol=1e-5,
    atol=1e-6,
)

ARANGE = NodeSpec(
    "arange",
    lambda p: [
        Operand(
            "Y", p["dtype"], (len(np.arange(p["start"], p["end"], p["step"])),), True
        )
    ],
    lambda b, t, p: b.add_arange(
        str(p["start"]),
        _scalar(p["dtype"]),
        str(p["end"]),
        _scalar(p["dtype"]),
        str(p["step"]),
        _scalar(p["dtype"]),
        "Y",
        t["Y"],
    ),
    lambda p, rng: {},
    lambda p, ins: {"Y": np.arange(p["start"], p["end"], p["step"]).astype(p["dtype"])},
    (
        ("int64", dict(start=0, end=10, step=1, dtype=i64)),
        ("float32", dict(start=0.5, end=3.0, step=0.25, dtype=f32)),
    ),
)


# ---------------------------------------------------------------------------
# Linear algebra, convolution, pooling, normalization, upsampling
# ---------------------------------------------------------------------------
MATMUL = NodeSpec(
    "matmul",
    lambda p: [Operand("A", p["dtype"], p["a"]), Operand("B", p["dtype"], p["b"])]
    + [Operand("Y", p["dtype"], p["a"][:-1] + p["b"][-1:], True)],
    lambda b, t, p: b.add_matmul_op("A", t["A"], "B", t["B"], "Y", t["Y"]),
    lambda p, rng: {
        "A": _random(rng, p["a"], p["dtype"]),
        "B": _random(rng, p["b"], p["dtype"]),
    },
    lambda p, ins: {
        "Y": np.matmul(ins["A"].astype(f64), ins["B"].astype(f64)).astype(p["dtype"])
    },
    (
        ("2d_f32", dict(a=(8, 16), b=(16, 12), dtype=f32)),
        ("batched_f32", dict(a=(3, 8, 16), b=(3, 16, 12), dtype=f32)),
        ("2d_f64", dict(a=(8, 16), b=(16, 12), dtype=f64)),
    ),
    rtol=1e-4,
    atol=1e-4,
)


def _conv_reference(p, ins):
    x, w = ins["X"].astype(f64), ins["W"].astype(f64)
    nd, g, pad, s = x.ndim - 2, p["group"], p["pad"], p["stride"]
    x = np.pad(x, [(0, 0), (0, 0)] + [(pad, pad)] * nd)
    win = sliding_window_view(x, w.shape[2:], axis=tuple(range(2, 2 + nd)))
    win = win[(slice(None), slice(None)) + (slice(None, None, s),) * nd]
    cin, cout = w.shape[1], w.shape[0] // g
    outs = []
    for gi in range(g):
        xg = win[:, gi * cin : (gi + 1) * cin]
        wg = w[gi * cout : (gi + 1) * cout]
        spatial = "hw"[:nd] if nd <= 2 else "hwd"
        kern = "klm"[:nd]
        outs.append(np.einsum(f"nc{spatial}{kern},oc{kern}->no{spatial}", xg, wg))
    y = np.concatenate(outs, axis=1)
    if p.get("bias"):
        y += ins["B"].astype(f64).reshape((1, -1) + (1,) * nd)
    return {"Y": y.astype(p["dtype"])}


def _conv_out(p):
    x, k, pad, s = p["x"], p["w"][2:], p["pad"], p["stride"]
    spatial = tuple((x[2 + i] + 2 * pad - k[i]) // s + 1 for i in range(len(k)))
    return (x[0], p["w"][0]) + spatial


def _conv_add(b, t, p):
    nd = len(p["x"]) - 2
    args = (
        [str(d) for d in p["x"]],
        [str(d) for d in p["w"][2:]],
        [str(p["stride"])] * nd,
        [str(p["pad"])] * (2 * nd),
        ["1"] * nd,
        str(p["w"][0]),
        str(p["group"]),
    )
    if p.get("bias"):
        b.add_conv_with_bias("X", t["X"], "W", t["W"], "Y", t["Y"], "B", t["B"], *args)
    else:
        b.add_conv("X", t["X"], "W", t["W"], "Y", t["Y"], *args)


CONV = NodeSpec(
    "conv",
    lambda p: [Operand("X", p["dtype"], p["x"]), Operand("W", p["dtype"], p["w"])]
    + [Operand("Y", p["dtype"], _conv_out(p), True)]
    + ([Operand("B", p["dtype"], (p["w"][0],))] if p.get("bias") else []),
    _conv_add,
    lambda p, rng: {
        "X": _random(rng, p["x"], p["dtype"]),
        "W": _random(rng, p["w"], p["dtype"]),
    }
    | ({"B": _random(rng, (p["w"][0],), p["dtype"])} if p.get("bias") else {}),
    _conv_reference,
    (
        (
            "2d_f32",
            dict(x=(1, 2, 6, 6), w=(4, 2, 3, 3), stride=1, pad=0, group=1, dtype=f32),
        ),
        (
            "2d_bias_f32",
            dict(
                x=(1, 2, 6, 6),
                w=(4, 2, 3, 3),
                stride=1,
                pad=0,
                group=1,
                dtype=f32,
                bias=True,
            ),
        ),
        (
            "2d_groups_pad_f32",
            dict(x=(1, 4, 5, 5), w=(4, 2, 3, 3), stride=1, pad=1, group=2, dtype=f32),
        ),
        (
            "1d_stride2_f64",
            dict(x=(2, 3, 10), w=(4, 3, 3), stride=2, pad=0, group=1, dtype=f64),
        ),
    ),
    rtol=1e-4,
    atol=1e-4,
)


def _pool_out(p):
    x, k, s, pad = p["x"], p["k"], p["stride"], p["pad"]
    return x[:2] + tuple((x[2 + i] + 2 * pad - k) // s + 1 for i in range(len(x) - 2))


def _pool_reference(p, ins):
    x, k, s, pad = ins["X"].astype(f64), p["k"], p["stride"], p["pad"]
    x = np.pad(x, [(0, 0), (0, 0), (pad, pad), (pad, pad)], constant_values=-np.inf)
    win = sliding_window_view(x, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    y = {
        "max": win.max(axis=(-2, -1)),
        "sum": win.sum(axis=(-2, -1)),
        "avg": win.mean(axis=(-2, -1)),
    }[p["mode"]]
    return {"Y": y.astype(p["dtype"])}


POOLING = NodeSpec(
    "pooling",
    lambda p: [
        Operand("X", p["dtype"], p["x"]),
        Operand("Y", p["dtype"], _pool_out(p), True),
    ],
    lambda b, t, p: b.add_pooling(
        p["mode"],
        "X",
        t["X"],
        "Y",
        t["Y"],
        [str(d) for d in p["x"]],
        [str(p["k"])] * 2,
        [str(p["stride"])] * 2,
        [str(p["pad"])] * 4,
        ["1"] * 2,
    ),
    lambda p, rng: {"X": _random(rng, p["x"], p["dtype"])},
    _pool_reference,
    (
        (
            "max_2x2_f32",
            dict(mode="max", x=(1, 2, 6, 6), k=2, stride=2, pad=0, dtype=f32),
        ),
        (
            "max_3x3_pad_f32",
            dict(mode="max", x=(1, 2, 7, 7), k=3, stride=2, pad=1, dtype=f32),
        ),
        (
            "avg_3x3_f64",
            dict(mode="avg", x=(1, 1, 5, 5), k=3, stride=1, pad=0, dtype=f64),
        ),
        (
            "sum_2x2_f32",
            dict(mode="sum", x=(2, 3, 4, 4), k=2, stride=2, pad=0, dtype=f32),
        ),
    ),
    rtol=1e-5,
    atol=1e-5,
)


def _batchnorm_reference(p, ins):
    shape = (1, -1) + (1,) * (len(p["x"]) - 2)
    c = {n: ins[n].astype(f64).reshape(shape) for n in ("Var", "E", "Gamma", "Beta")}
    y = (ins["X"].astype(f64) - c["E"]) / np.sqrt(c["Var"] + 1e-5) * c["Gamma"] + c[
        "Beta"
    ]
    return {"Y": y.astype(p["dtype"])}


BATCHNORM = NodeSpec(
    "batchnorm",
    lambda p: [Operand("X", p["dtype"], p["x"])]
    + [Operand(n, p["dtype"], (p["x"][1],)) for n in ("Var", "E", "Gamma", "Beta")]
    + [Operand("Y", p["dtype"], p["x"], True)],
    lambda b, t, p: b.add_batchnorm_with_bias(
        "X",
        t["X"],
        "Var",
        t["Var"],
        "E",
        t["E"],
        "Gamma",
        t["Gamma"],
        "Beta",
        t["Beta"],
        "1e-5",
        _scalar(p["dtype"]),
        "Y",
        t["Y"],
    ),
    lambda p, rng: {
        "X": _random(rng, p["x"], p["dtype"]),
        "Var": _random(rng, (p["x"][1],), p["dtype"], True),
    }
    | {n: _random(rng, (p["x"][1],), p["dtype"]) for n in ("E", "Gamma", "Beta")},
    _batchnorm_reference,
    (
        ("4d_f32", dict(x=(2, 3, 4, 4), dtype=f32)),
        ("3d_f64", dict(x=(2, 5, 7), dtype=f64)),
    ),
    rtol=1e-4,
    atol=1e-5,
)


def _layernorm_stat_shape(p):
    return p["x"][: -len(p["norm"])] or (1,)


def _layernorm_operands(p):
    ops = [Operand("X", p["dtype"], p["x"])]
    if p["variant"] in ("affine", "affine_bias"):
        ops.append(Operand("Gamma", p["dtype"], p["norm"]))
    if p["variant"] == "affine_bias":
        ops.append(Operand("Beta", p["dtype"], p["norm"]))
    stat = _layernorm_stat_shape(p)
    return ops + [
        Operand("Y", p["dtype"], p["x"], True),
        Operand("Mean", p["dtype"], stat, True),
        Operand("Rstd", p["dtype"], stat, True),
    ]


def _layernorm_add(b, t, p):
    eps = ("1e-5", _scalar(p["dtype"]))
    out = (
        "Y",
        t["Y"],
        "Mean",
        t["Mean"],
        "Rstd",
        t["Rstd"],
        [str(d) for d in p["norm"]],
    )
    if p["variant"] == "plain":
        b.add_layernorm("X", t["X"], *eps, *out)
    elif p["variant"] == "affine":
        b.add_layernorm_affine("X", t["X"], *eps, "Gamma", t["Gamma"], *out)
    else:
        b.add_layernorm_affine_with_bias(
            "X", t["X"], *eps, "Gamma", t["Gamma"], "Beta", t["Beta"], *out
        )


def _layernorm_reference(p, ins):
    axes = tuple(range(len(p["x"]) - len(p["norm"]), len(p["x"])))
    x = ins["X"].astype(f64)
    mean = x.mean(axis=axes, keepdims=True)
    rstd = 1 / np.sqrt(x.var(axis=axes, keepdims=True) + 1e-5)
    y = (x - mean) * rstd
    if "Gamma" in ins:
        y = y * ins["Gamma"].astype(f64)
    if "Beta" in ins:
        y = y + ins["Beta"].astype(f64)
    stat = _layernorm_stat_shape(p)
    return {
        "Y": y.astype(p["dtype"]),
        "Mean": mean.reshape(stat).astype(p["dtype"]),
        "Rstd": rstd.reshape(stat).astype(p["dtype"]),
    }


LAYERNORM = NodeSpec(
    "layernorm",
    _layernorm_operands,
    _layernorm_add,
    lambda p, rng: {
        op.name: _random(rng, op.shape, op.dtype)
        for op in _layernorm_operands(p)
        if op.name in ("X", "Gamma", "Beta")
    },
    _layernorm_reference,
    (
        ("plain_f32", dict(x=(4, 8), norm=(8,), variant="plain", dtype=f32)),
        ("affine_f32", dict(x=(4, 8), norm=(8,), variant="affine", dtype=f32)),
        (
            "affine_bias_3d_f64",
            dict(x=(2, 3, 8), norm=(3, 8), variant="affine_bias", dtype=f64),
        ),
    ),
    rtol=1e-4,
    atol=1e-5,
)


def _bilinear_axis(n_in, n_out, align):
    o = np.arange(n_out, dtype=f64)
    if align:
        src = o * (n_in - 1) / (n_out - 1) if n_out > 1 else np.zeros(n_out)
    else:
        src = np.maximum(0.0, (o + 0.5) * (n_in / n_out) - 0.5)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, src - i0


def _upsample_reference(p, ins):
    x = ins["X"].astype(f64)
    h0, h1, lh = _bilinear_axis(p["x"][2], p["out"][0], p["align"])
    w0, w1, lw = _bilinear_axis(p["x"][3], p["out"][1], p["align"])
    lh, lw = lh[:, None], lw[None, :]
    top = x[:, :, h0][:, :, :, w0] * (1 - lw) + x[:, :, h0][:, :, :, w1] * lw
    bottom = x[:, :, h1][:, :, :, w0] * (1 - lw) + x[:, :, h1][:, :, :, w1] * lw
    return {"Y": (top * (1 - lh) + bottom * lh).astype(p["dtype"])}


UPSAMPLE = NodeSpec(
    "upsample_bilinear2d",
    lambda p: [
        Operand("X", p["dtype"], p["x"]),
        Operand("Y", p["dtype"], p["x"][:2] + p["out"], True),
    ],
    lambda b, t, p: b.add_upsample_bilinear2d(
        "X",
        t["X"],
        "Y",
        t["Y"],
        [str(d) for d in p["x"]],
        [str(d) for d in p["x"][:2] + p["out"]],
        p["align"],
        [],
    ),
    lambda p, rng: {"X": _random(rng, p["x"], p["dtype"])},
    _upsample_reference,
    (
        ("x2_f32", dict(x=(1, 2, 4, 4), out=(8, 8), align=False, dtype=f32)),
        ("align_corners_f64", dict(x=(1, 2, 4, 5), out=(7, 9), align=True, dtype=f64)),
    ),
    rtol=1e-5,
    atol=1e-5,
)


NODES = (
    BINARY
    + UNARY
    + (TASKLET, FILL)
    + REDUCE
    + (COPY, CONCAT, CONST_PADDING, BROADCAST, CONDITIONAL_COPY)
    + (INDEX, EMBEDDING, EMBEDDING_RENORM, ARANGE)
    + (MATMUL, CONV, POOLING, BATCHNORM, LAYERNORM, UPSAMPLE)
)

# Cases that already fail on the CPU target, keyed by test id.
BROKEN_ON_CPU = {
    **{
        f"{name}-{case}": "untested expand"
        for name in ("elu", "erf", "leaky_relu")
        for case in ("2d_f32", "3d_f64")
    },
    "hard_sigmoid-2d_f32": "node invalid: input Y is not scalar",
    "hard_sigmoid-3d_f64": "node invalid: input Y is not scalar",
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def build_sdfg(spec, params):
    builder = StructuredSDFGBuilder(f"offload_{spec.name}")
    types = {}
    for op in spec.operands(params):
        builder.add_container(op.name, Pointer(_scalar(op.dtype)), is_argument=True)
        types[op.name] = _tensor(op.dtype, op.shape)
    spec.add_node(builder, types, params)
    return builder.move()


def host_compute(graph, backend_name):
    """Describe every loop and library node left on the host after scheduling."""
    impl_prefix = GPU_IMPL_PREFIX[backend_name]
    found = []

    def walk(node):
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        if not isinstance(node, dict):
            return
        kind = node.get("type")
        if kind in ("map", "for", "while"):
            schedule = node.get("schedule_type", {})
            value = schedule.get("value") if isinstance(schedule, dict) else schedule
            if value in GPU_SCHEDULES:
                return
            found.append(f"{kind} {node.get('indvar', '')} [{value}]".strip())
        elif kind == "library_node":
            code, impl = node.get("code", ""), node.get("implementation_type", "")
            if not code.endswith("Offloading") and not impl.startswith(impl_prefix):
                found.append(f"libnode {code} [{impl or 'NONE'}]")
        for value in node.values():
            walk(value)

    walk(graph["root"])
    return found


def run_case(spec, params, target, output_dir: Path, check_offload=True):
    """Return ``(status, message)`` for one case; status is OK or a failure reason."""
    try:
        sdfg = build_sdfg(spec, params)
        sdfg.validate()
        opts = TargetOptions(target, "server")
        sdfg.expand(opts)
        sdfg.simplify()
        sdfg.normalize()
        opts.already_normalized = True
        sdfg.schedule(opts)
        sdfg.validate()
        host = host_compute(json.loads(sdfg.to_json()), target) if check_offload else []
        output_dir.mkdir(parents=True, exist_ok=True)
        compiled = CompiledSDFG(sdfg._compile(str(output_dir), target), sdfg)
    except Exception as error:
        return (
            "PIPELINE_ERROR",
            f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}",
        )

    operands = spec.operands(params)
    ins = spec.inputs(params, np.random.default_rng(0))
    args = {
        op.name: (
            np.array(ins[op.name])
            if op.name in ins
            else np.zeros(op.shape, dtype=op.dtype)
        )
        for op in operands
    }
    try:
        compiled(*(args[op.name] for op in operands))
        rtol, atol = params.get("rtol", spec.rtol), params.get("atol", spec.atol)
        for name, expected in spec.reference(params, ins).items():
            _assert_output(args[name], expected, rtol, atol)
        correct = None
    except Exception as error:
        correct = f"{type(error).__name__}: {' '.join(str(error).split())[:300]}"

    if host:
        result = "correct" if correct is None else "incorrect"
        return "NOT_OFFLOADED", f"host: {'; '.join(host)} (result {result})"
    if correct is not None:
        return "WRONG_RESULT", correct
    return "OK", ""


def _child(queue, spec_name, case_id, target, output_dir, check_offload):
    spec = next(s for s in NODES if s.name == spec_name)
    params = dict(dict(spec.cases)[case_id])
    try:
        queue.put(run_case(spec, params, target, Path(output_dir), check_offload))
    except BaseException:
        queue.put(
            ("PIPELINE_ERROR", traceback.format_exc(limit=1).strip().splitlines()[-1])
        )


def run_case_isolated(spec, case_id, target, output_dir: Path, check_offload=True):
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(
        target=_child,
        args=(queue, spec.name, case_id, target, str(output_dir), check_offload),
    )
    proc.start()
    proc.join(CASE_TIMEOUT_S)
    if proc.is_alive():
        proc.kill()
        return "PIPELINE_ERROR", f"timeout after {CASE_TIMEOUT_S}s"
    try:
        return queue.get(timeout=5)
    except queue_module.Empty:
        return "PIPELINE_ERROR", f"crashed with exit code {proc.exitcode}"


def register(namespace, backend):
    """Define the tensor node offloading test, bound to ``backend``, into ``namespace``."""

    @pytest.mark.parametrize(
        "spec,case_id",
        [
            pytest.param(
                spec,
                case_id,
                id=f"{spec.name}-{case_id}",
                marks=(
                    [pytest.mark.skip(reason=BROKEN_ON_CPU[f"{spec.name}-{case_id}"])]
                    if f"{spec.name}-{case_id}" in BROKEN_ON_CPU
                    else []
                ),
            )
            for spec in NODES
            for case_id, _ in spec.cases
        ],
    )
    def test_tensor_node_offload(spec, case_id, tmp_path):
        status, message = run_case_isolated(
            spec, case_id, backend.target, tmp_path / spec.name
        )
        assert status == "OK", f"{status}: {message}"

    namespace.update(test_tensor_node_offload=test_tensor_node_offload)
