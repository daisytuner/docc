"""
Integration test for the CUDA softmax dispatcher.

Creates an SDFG with a SoftmaxNode using StructuredSDFGBuilder,
compiles it targeting CUDA, runs it, and verifies correctness
against a NumPy reference implementation.
"""

from pathlib import Path

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


pytestmark = pytest.mark.cuda()


def numpy_softmax(x, axis=-1):
    """Numerically stable softmax reference implementation."""
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def build_softmax_sdfg(shape, axes):
    """Build an SDFG containing a single SoftmaxNode."""
    builder = StructuredSDFGBuilder("softmax_test")

    float_scalar = Scalar(PrimitiveType.Float)
    builder.add_container("X", Pointer(float_scalar), is_argument=True)
    builder.add_container("Y", Pointer(float_scalar), is_argument=True)

    shape_strs = [str(d) for d in shape]
    input_tensor = Tensor(float_scalar, shape_strs)
    output_tensor = Tensor(float_scalar, shape_strs)

    builder.add_reduce_op("softmax", "X", input_tensor, "Y", output_tensor, axes, False)

    return builder.move()


def compile_and_run_softmax(shape, axes, output_root: Path):
    """Compile a softmax SDFG for CUDA and execute it."""
    sdfg = build_softmax_sdfg(shape, axes)
    sdfg.validate()

    opts = TargetOptions("cuda", "server")
    sdfg.expand(opts)
    sdfg.validate()

    # After expand: SoftmaxNode should have CUDAWithTransfers impl type
    json_after_expand = sdfg.to_json()
    assert "ml::Softmax" in json_after_expand, (
        "SoftmaxNode was expanded away during expand(cuda); "
        "CudaExpansionPass should preserve it with CUDAWithTransfers"
    )
    assert (
        "CUDAWithTransfers" in json_after_expand
    ), "SoftmaxNode does not have CUDAWithTransfers after expand(cuda)"

    sdfg.simplify()
    sdfg.validate()

    # After simplify: SoftmaxNode should still be present
    json_after_simplify = sdfg.to_json()
    assert (
        "ml::Softmax" in json_after_simplify
    ), "SoftmaxNode was destroyed during simplify()"

    sdfg.normalize()
    sdfg.validate()

    sdfg.schedule(opts)
    sdfg.validate()

    # Verify SoftmaxNode survived the pipeline (present at codegen time)
    json_str = sdfg.to_json()
    assert "ml::Softmax" in json_str, (
        "SoftmaxNode was expanded away during the CUDA pipeline; "
        "it should be preserved with CUDAWithTransfers implementation type"
    )
    assert (
        "CUDAWithTransfers" in json_str
    ), "SoftmaxNode does not have CUDAWithTransfers implementation type after schedule"

    shape_str = "x".join(str(s) for s in shape)
    output_dir = output_root / f"softmax_test_{shape_str}_axis{axes[0]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_path = sdfg._compile(str(output_dir), "cuda")
    compiled = CompiledSDFG(lib_path, sdfg)

    rng = np.random.default_rng(42)
    X = rng.standard_normal(shape).astype(np.float32)
    Y = np.zeros(shape, dtype=np.float32)

    compiled(X, Y)

    Y_ref = numpy_softmax(X, axis=axes[0])
    np.testing.assert_allclose(Y, Y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((64, 128), [1]),  # 2D, last axis
        ((16, 256), [1]),  # 2D, last axis (wider rows)
        ((8, 32, 64), [2]),  # 3D, last axis
        ((4, 16, 128), [2]),  # 3D, last axis (wider rows)
        ((2, 64, 256), [2]),  # 3D, last axis (large)
        ((1, 8, 256, 256), [3]),  # 4D segformer block0 (batch=1)
        ((4, 8, 256, 256), [3]),  # 4D segformer block0 (batch=4)
        ((1, 8, 64, 64), [3]),  # 4D segformer block1 (batch=1)
        ((16, 5, 16, 16), [3]),  # 4D segformer block3 (batch=16)
    ],
    ids=[
        "2d_64x128_axis1",
        "2d_16x256_axis1",
        "3d_8x32x64_axis2",
        "3d_4x16x128_axis2",
        "3d_2x64x256_axis2",
        "4d_block0_b1",
        "4d_block0_b4",
        "4d_block1_b1",
        "4d_block3_b16",
    ],
)
@pytest.mark.cuda()
def test_softmax_cuda(shape, axes, tmp_path):
    compile_and_run_softmax(shape, axes, tmp_path)
