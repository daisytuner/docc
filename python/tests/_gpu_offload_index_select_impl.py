"""
Backend-agnostic check that a standalone ``aten.index_select`` (``ml::Index``) node is offloaded.

The node is built directly through ``StructuredSDFGBuilder.add_index_op`` with a
single 1-D index tensor (no PyTorch frontend), run through the regular
expand -> simplify -> normalize -> schedule pipeline and compiled. The test
requires a device kernel in the generated sources and a bit-exact NumPy match.
"""

from pathlib import Path

import numpy as np
import pytest

from docc.sdfg import Pointer, Scalar, StructuredSDFGBuilder, TargetOptions, Tensor
from docc.compiler.compiled_sdfg import CompiledSDFG

from _gpu_offload_layer_impl import _PRIMITIVE, _assert_output

# (id, x_shape, dim, num_indices, index dtype)
CASES = (
    ("2d_dim0", (8, 5), 0, 3, np.int64),
    ("2d_dim1", (4, 7), 1, 5, np.int64),
    ("3d_dim1_repeated", (2, 6, 3), 1, 9, np.int64),
    ("2d_dim0_i32", (16, 4), 0, 6, np.int32),
    ("2d_dim0_large", (1000, 64), 0, 256, np.int64),
)


def _output_shape(x_shape, dim, num_indices):
    return x_shape[:dim] + (num_indices,) + x_shape[dim + 1 :]


def build_index_select_sdfg(x_shape, dim, num_indices, index_dtype):
    builder = StructuredSDFGBuilder("offload_index_select")
    value = Scalar(_PRIMITIVE[np.dtype(np.float32)])
    index = Scalar(_PRIMITIVE[np.dtype(index_dtype)])
    builder.add_container("X", Pointer(value), is_argument=True)
    builder.add_container("Idx", Pointer(index), is_argument=True)
    builder.add_container("Y", Pointer(value), is_argument=True)

    def tensor(element, shape):
        return Tensor(element, [str(d) for d in shape])

    builder.add_index_op(
        "Y",
        tensor(value, _output_shape(x_shape, dim, num_indices)),
        "X",
        tensor(value, x_shape),
        ["Idx"],
        [tensor(index, (num_indices,))],
        [dim],
    )
    return builder.move()


def _compile(backend, sdfg, output_dir: Path):
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
    generated = "\n".join(p.read_text() for p in sources)
    assert (
        "__global__" in generated
    ), f"index_select was not offloaded to a {backend.name} kernel"
    return CompiledSDFG(lib_path, sdfg)


def register(namespace, backend):
    """Define the index_select offloading test, bound to ``backend``, into ``namespace``."""

    @pytest.mark.parametrize(
        "x_shape,dim,num_indices,index_dtype",
        [pytest.param(*case[1:], id=case[0]) for case in CASES],
    )
    def test_offload_index_select(x_shape, dim, num_indices, index_dtype, tmp_path):
        sdfg = build_index_select_sdfg(x_shape, dim, num_indices, index_dtype)
        compiled = _compile(backend, sdfg, tmp_path / "index_select")

        rng = np.random.default_rng(0)
        x = rng.standard_normal(x_shape).astype(np.float32)
        idx = rng.integers(0, x_shape[dim], size=num_indices, dtype=index_dtype)
        y = np.zeros(_output_shape(x_shape, dim, num_indices), dtype=np.float32)

        compiled(x, idx, y)

        _assert_output(y, np.take(x, idx, axis=dim), 0.0, 0.0)

    namespace.update(test_offload_index_select=test_offload_index_select)
