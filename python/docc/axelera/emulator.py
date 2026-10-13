"""CPU stand-in for the AxScript subset docc generates, used to run modules without Axelera hardware.

Loaded modules see this module as ``axelera.compiler.alto.script``. Datapath units run in program order and streams
are FIFOs, which matches the hardware for datapaths whose units push and pop once per grid step. DPU registers keep
float32 precision instead of float18.
"""

import contextlib
import importlib.util
import itertools
import sys
import types
from collections import deque

import numpy as np

DDR = "DDR"
L1 = "L1"
L2 = "L2"
IMC = "IMC"

_outputs: list = []


def _element_type(dtype: str):
    base, lanes = dtype.split("x")
    return (np.float32 if base == "float18" else np.dtype(base)), int(lanes)


class _Stream(deque):
    def push(self, value):
        self.append(np.asarray(value))

    def pop(self):
        return self.popleft()


def ir_module(cls):
    return cls


def prim_func(fn):
    def run(*buffers):
        _outputs.clear()
        fn(*buffers)
        return list(_outputs)

    return run


def Buffer(shape, dtype, scope=DDR):
    return (shape, dtype, scope)


def alloc_buffer(shape, dtype, scope=DDR):
    element, lanes = _element_type(dtype)
    return np.zeros((*shape, lanes), element)


def alloc_constant(values, dtype, scope=DDR):
    element, lanes = _element_type(dtype)
    return np.asarray(values, element).reshape(-1, lanes)


def alloc_imc(blocks):
    """The resident weight matrix, unpacked: (rows, columns) int8 for (row, column) blocks of 64x64."""
    return np.zeros((blocks[0] * 64, blocks[1] * 64), np.int8)


def Ramp(base, stride, lanes):
    return slice(base, base + stride * lanes, stride)


# Atomics run synchronously, so ids and waits only order what already is in order; `where=False` skips them.
def _copy(source, scope=None, output=None, where=True, **kwargs):
    if where:
        np.copyto(output, source.astype(output.dtype))


def _load_imc(buffer=None, imc=None, where=True, **kwargs):
    if where:
        np.copyto(imc, buffer.reshape(buffer.shape[0], -1))


def _matmul(
    matrix_ifd=None, matrix_imc=None, output=None, partial=None, where=True, **kwargs
):
    if not where:
        return
    rows = matrix_ifd.shape[0]
    product = matrix_ifd.reshape(rows, -1).astype(np.int32) @ matrix_imc.astype(
        np.int32
    )
    result = product.astype(np.float32)
    if partial is not None:
        result += partial.reshape(rows, -1)
    np.copyto(output, result.reshape(output.shape).astype(output.dtype))


# The axol operators docc emits, as `from axelera.compiler.alto.axol import atomic` sees them.
atomic = types.SimpleNamespace(copy=_copy, load_imc=_load_imc, matmul=_matmul)


def waits(*ids):
    pass


def And(a, b):  # noqa: N802
    return a and b


def Or(a, b):  # noqa: N802
    return a or b


def Not(a):  # noqa: N802
    return not a


def grid(*extents):
    if len(extents) == 1:
        return iter(range(extents[0]))
    return itertools.product(*(range(extent) for extent in extents))


serial = grid


def decl_stream(dtype):
    return _Stream()


def array(dtype, size=None):
    return {}


def cast(value, dtype):
    element = _element_type(dtype)[0]
    value = np.asarray(value)
    # The DPU rounds to nearest and saturates when it narrows to integers.
    if np.issubdtype(element, np.integer) and not np.issubdtype(
        value.dtype, np.integer
    ):
        limits = np.iinfo(element)
        value = np.clip(np.rint(value), limits.min, limits.max)
    return value.astype(element)


# Lane-wise binary DPU operations; these shadow the builtins only inside this module.
def max(a, b):  # noqa: A001
    return np.maximum(a, b)


def min(a, b):  # noqa: A001
    return np.minimum(a, b)


def _lane_reduction(reduce):
    return lambda value: np.full_like(value, reduce(value))


# Cross-lane reductions broadcast their result back to every lane.
dpu = types.SimpleNamespace(
    ops=types.SimpleNamespace(
        add_reduce=_lane_reduction(np.sum),
        max_reduce=_lane_reduction(np.max),
        min_reduce=_lane_reduction(np.min),
        fma=lambda a, b, c: a * b + c,
        neg=np.negative,
        recip=np.reciprocal,
        sqrt=np.sqrt,
        sin=np.sin,
        cos=np.cos,
        log2=np.log2,
        exp2=np.exp2,
    )
)


def output(*buffers):
    _outputs.extend(buffers)


def barrier(*cores):
    pass


def _unit(**kwargs):
    return contextlib.nullcontext()


# Core regions run one after another and barriers are no-ops: datapaths on different cores touch disjoint chunks,
# and the generated code reads other cores' results only on the last core.
Datapath = IFD = DPU = ODR = AICore = _unit


@contextlib.contextmanager
def _as_axscript():
    names = [
        "axelera",
        "axelera.compiler",
        "axelera.compiler.alto",
        "axelera.compiler.alto.axol",
        "axelera.compiler.alto.script",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    for name in names[:-1]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["axelera.compiler.alto.axol"].atomic = atomic
    sys.modules[names[-1]] = sys.modules[__name__]
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load(path: str, name: str, lanes: int = 64):
    """Load the AxScript module at ``path`` as a function taking the SDFG's arguments as flat NumPy arrays."""
    spec = importlib.util.spec_from_file_location("_docc_axscript_" + name, path)
    module = importlib.util.module_from_spec(spec)
    with _as_axscript():
        spec.loader.exec_module(module)
    kernel = getattr(module, name).main
    arguments = [
        (direction, count, shape)
        for _, direction, count, shape in module._docc_arguments
    ]

    def pwords(arg, count, shape):
        padded = np.zeros(int(np.prod(shape)) * lanes, arg.dtype)
        padded[:count] = arg.reshape(-1)[:count]
        return padded.reshape(*shape, lanes)

    def run(*args):
        inputs = [
            pwords(arg, count, shape)
            for arg, (d, count, shape) in zip(args, arguments)
            if d == "in"
        ]
        outputs = [
            (arg, count) for arg, (d, count, _) in zip(args, arguments) if d == "out"
        ]
        for (arg, count), result in zip(outputs, kernel(*inputs)):
            np.copyto(arg.reshape(-1)[:count], result.reshape(-1)[:count])

    return run
