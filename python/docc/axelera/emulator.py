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


def grid(*extents):
    if len(extents) == 1:
        return iter(range(extents[0]))
    return itertools.product(*(range(extent) for extent in extents))


def decl_stream(dtype):
    return _Stream()


def array(dtype):
    return {}


def cast(value, dtype):
    return np.asarray(value).astype(_element_type(dtype)[0])


def output(*buffers):
    _outputs.extend(buffers)


def _unit(**kwargs):
    return contextlib.nullcontext()


Datapath = IFD = DPU = ODR = _unit


@contextlib.contextmanager
def _as_axscript():
    names = [
        "axelera",
        "axelera.compiler",
        "axelera.compiler.alto",
        "axelera.compiler.alto.script",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    for name in names[:-1]:
        sys.modules[name] = types.ModuleType(name)
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
    directions = [direction for _, direction in module._docc_arguments]

    def run(*args):
        inputs = [
            arg.reshape(-1, lanes) for arg, d in zip(args, directions) if d == "in"
        ]
        outputs = [arg for arg, d in zip(args, directions) if d == "out"]
        for arg, result in zip(outputs, kernel(*inputs)):
            np.copyto(arg, result.reshape(-1))

    return run
