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
        ((2, 3, 12, 13), [1]),  # Softmax2d example
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
        "softmax2d_example",
    ],
)
@pytest.mark.cuda()
def test_softmax_cuda(shape, axes, tmp_path):
    compile_and_run_softmax(shape, axes, tmp_path)


@pytest.mark.parametrize(
    "shape,axes",
    [
        # 2D, reduce over the outer (non-innermost) axis: inner stride == row width
        ((128, 64), [0]),  # inner=64
        ((256, 16), [0]),  # inner=16
        ((37, 71), [0]),  # non-power-of-two rows and stride
        # 3D, reduce over axis 0 or 1 (never the innermost)
        ((8, 32, 64), [0]),  # inner = 32*64 = 2048
        ((8, 32, 64), [1]),  # inner = 64
        ((4, 16, 128), [1]),  # inner = 128
        ((2, 64, 256), [0]),  # inner = 64*256 = 16384
        ((5, 7, 11), [1]),  # non-power-of-two, inner=11
        ((3, 129, 5), [1]),  # wide reduced axis, small inner=5
        # 4D, reduce over each non-innermost axis (segformer-like plus odd sizes)
        ((1, 8, 256, 256), [1]),  # inner = 256*256
        ((4, 8, 256, 256), [2]),  # inner = 256
        ((2, 3, 12, 13), [0]),  # inner = 3*12*13
        ((2, 3, 12, 13), [2]),  # inner = 13
        ((3, 5, 7, 9), [0]),  # all non-power-of-two
        ((3, 5, 7, 9), [1]),  # inner = 7*9 = 63
        ((3, 5, 7, 9), [2]),  # inner = 9
        # Reduced axis of size 1 (degenerate) over a non-innermost axis
        ((4, 1, 8, 8), [1]),  # single-element softmax, inner=64
        # Large reduced axis over a non-innermost dimension (exercises multi-warp reduction)
        ((2, 512, 8), [1]),  # row_size=512, inner=8
    ],
    ids=[
        "2d_128x64_axis0",
        "2d_256x16_axis0",
        "2d_37x71_axis0",
        "3d_8x32x64_axis0",
        "3d_8x32x64_axis1",
        "3d_4x16x128_axis1",
        "3d_2x64x256_axis0",
        "3d_5x7x11_axis1",
        "3d_3x129x5_axis1",
        "4d_1x8x256x256_axis1",
        "4d_4x8x256x256_axis2",
        "4d_2x3x12x13_axis0",
        "4d_2x3x12x13_axis2",
        "4d_3x5x7x9_axis0",
        "4d_3x5x7x9_axis1",
        "4d_3x5x7x9_axis2",
        "4d_reduce_size1_axis1",
        "3d_large_reduce_axis1",
    ],
)
@pytest.mark.cuda()
def test_softmax_cuda_non_innermost_axis(shape, axes, tmp_path):
    """Softmax reducing over a non-innermost axis (strided memory access)."""
    compile_and_run_softmax(shape, axes, tmp_path)
