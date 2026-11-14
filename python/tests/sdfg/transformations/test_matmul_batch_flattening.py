import numpy as np
import pytest

from docc.compiler.compiled_sdfg import CompiledSDFG
from docc.sdfg import Pointer, PrimitiveType, Scalar, StructuredSDFGBuilder, Tensor


@pytest.mark.parametrize(
    "a_layout,b_layout,y_layout,flattened",
    [
        (((2, 3, 4), None, 0), ((4, 5), None, 0), ((2, 3, 5), None, 0), True),
        (
            ((2, 3, 1, 4), (12, 4, 0, 1), 3),
            ((4, 5), (1, 4), 2),
            ((2, 3, 1, 5), (15, 5, 0, 1), 7),
            True,
        ),
        (
            ((2, 3, 4), (24, 8, 2), 0),
            ((4, 5), None, 0),
            ((2, 3, 5), None, 0),
            False,
        ),
        (
            ((2, 3, 4), None, 0),
            ((4, 5), None, 0),
            ((2, 3, 5), (30, 10, 2), 0),
            False,
        ),
        (((2, 3, 4), None, 0), ((2, 4, 5), None, 0), ((2, 3, 5), None, 0), False),
    ],
    ids=["shared", "singleton-offset-transpose", "strided-a", "strided-y", "batched-b"],
)
def test_matmul_batch_flattening(a_layout, b_layout, y_layout, flattened, tmp_path):
    builder = StructuredSDFGBuilder("matmul_batch_flattening")
    element = Scalar(PrimitiveType.Double)
    rng = np.random.default_rng(42)
    tensors, buffers, views = [], [], []
    for name, (shape, strides, offset) in zip(
        ("a", "b", "y"), (a_layout, b_layout, y_layout)
    ):
        if strides is None:
            strides = tuple(s // 8 for s in np.empty(shape).strides)
        extent = offset + sum((n - 1) * s for n, s in zip(shape, strides)) + 1
        buffer = rng.standard_normal(extent)
        view = np.ndarray(
            shape,
            dtype=np.float64,
            buffer=buffer,
            offset=offset * 8,
            strides=tuple(s * 8 for s in strides),
        )
        builder.add_container(name, Pointer(element), is_argument=True)
        tensors.append(
            Tensor(element, list(map(str, shape)), list(map(str, strides)), str(offset))
        )
        buffers.append(buffer)
        views.append(view)
    builder.add_matmul_op("a", tensors[0], "b", tensors[1], "y", tensors[2])
    expected_buffer = buffers[2].copy()
    expected = np.ndarray(
        views[2].shape,
        dtype=np.float64,
        buffer=expected_buffer,
        offset=y_layout[2] * 8,
        strides=views[2].strides,
    )
    expected[:] = views[0] @ views[1]

    sdfg = builder.move()
    sdfg.expand("none", "none")
    sdfg.validate()
    report = sdfg.loop_report()
    assert report["MAP"] == (4 if flattened else 2 * len(y_layout[0]))
    assert report["FOR"] == 1
    assert report.get("Malloc", 0) == 0
    assert report.get("Memcpy", 0) == 0
    CompiledSDFG(sdfg._compile(str(tmp_path), "sequential"), sdfg)(*buffers)
    np.testing.assert_allclose(buffers[2], expected_buffer, rtol=1e-12, atol=1e-12)
