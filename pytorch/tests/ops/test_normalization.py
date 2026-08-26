import torch
import torch.nn as nn

from tests import check

# --- BatchNorm1d ---


def test_batchnorm1d_eval_simple(target: str) -> None:
    class BatchNorm1dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batchnorm1d: nn.BatchNorm1d = nn.BatchNorm1d(3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.batchnorm1d(input)

    check(BatchNorm1dSimpleNet().eval(), torch.randn(2, 3, 16), target=target)


def test_batchnorm1d_eval_eps(target: str) -> None:
    class BatchNorm1dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batchnorm1d: nn.BatchNorm1d = nn.BatchNorm1d(3, eps=1e-04)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.batchnorm1d(input)

    check(BatchNorm1dSimpleNet().eval(), torch.randn(2, 3, 16), target=target)


# --- BatchNorm2d ---


def test_batchnorm2d_eval_simple(target: str) -> None:
    class BatchNorm2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batchnorm2d: nn.BatchNorm2d = nn.BatchNorm2d(3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.batchnorm2d(input)

    check(BatchNorm2dSimpleNet().eval(), torch.randn(2, 3, 16, 16), target=target)


def test_batchnorm2d_eval_eps(target: str) -> None:
    class BatchNorm2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batchnorm2d: nn.BatchNorm2d = nn.BatchNorm2d(3, eps=1e-04)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.batchnorm2d(input)

    check(BatchNorm2dSimpleNet().eval(), torch.randn(2, 3, 16, 16), target=target)


# --- BatchNorm3d ---


def test_batchnorm3d_eval_simple(target: str) -> None:
    class BatchNorm3dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batchnorm3d: nn.BatchNorm3d = nn.BatchNorm3d(3)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.batchnorm3d(input)

    check(BatchNorm3dSimpleNet().eval(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_batchnorm3d_eval_eps(target: str) -> None:
    class BatchNorm3dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batchnorm3d: nn.BatchNorm3d = nn.BatchNorm3d(3, eps=1e-04)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.batchnorm3d(input)

    check(BatchNorm3dSimpleNet().eval(), torch.randn(2, 3, 16, 16, 16), target=target)


# --- LayerNorm ---


def test_layernorm_int_shape(target: str) -> None:
    class LayerNormIntShapeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(16)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormIntShapeNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_list_shape(target: str) -> None:
    class LayerNormListShapeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm([3, 16])

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormListShapeNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_tuple_shape(target: str) -> None:
    class LayerNormTupleShapeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm((3, 16))  # type: ignore

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormTupleShapeNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_full_shape(target: str) -> None:
    class LayerNormFullShapeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm([2, 3, 16])

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormFullShapeNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_eps(target: str) -> None:
    class LayerNormEpsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(16, eps=1e-03)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormEpsNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_no_affine(target: str) -> None:
    class LayerNormNoAffineNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(16, elementwise_affine=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormNoAffineNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_affine_no_bias(target: str) -> None:
    class LayerNormAffineNoBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                16, elementwise_affine=True, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormAffineNoBiasNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_affine_bias(target: str) -> None:
    class LayerNormAffineBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                16, elementwise_affine=True, bias=True
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormAffineBiasNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_no_affine_eps(target: str) -> None:
    class LayerNormNoAffineEpsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                16, eps=1e-03, elementwise_affine=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormNoAffineEpsNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_affine_bias_eps(target: str) -> None:
    class LayerNormAffineBiasEpsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                16, eps=1e-03, elementwise_affine=True, bias=True
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormAffineBiasEpsNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_affine_no_bias_eps(target: str) -> None:
    class LayerNormAffineNoBiasEpsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                16, eps=1e-03, elementwise_affine=True, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormAffineNoBiasEpsNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_list_shape_no_affine(target: str) -> None:
    class LayerNormListShapeNoAffineNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                [3, 16], elementwise_affine=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNormListShapeNoAffineNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm_list_shape_affine_no_bias(target: str) -> None:
    class LayerNormListShapeAffineNoBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                [3, 16], elementwise_affine=True, bias=False
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(
        LayerNormListShapeAffineNoBiasNet().eval(),
        torch.randn(2, 3, 16),
        target=target,
    )


def test_layernorm_dtype(target: str) -> None:
    class LayerNormDtypeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(16, dtype=torch.float64)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(
        LayerNormDtypeNet().eval(),
        torch.randn(2, 3, 16, dtype=torch.float64),
        target=target,
    )


def test_layernorm_dtype32(target: str) -> None:
    class LayerNormDtypeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(16, dtype=torch.float32)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(
        LayerNormDtypeNet().eval(),
        torch.randn(2, 3, 16, dtype=torch.float32),
        target=target,
    )


def test_layernorm_dtype_no_affine(target: str) -> None:
    class LayerNormDtypeNoAffineNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                16, elementwise_affine=False, dtype=torch.float64
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(
        LayerNormDtypeNoAffineNet().eval(),
        torch.randn(2, 3, 16, dtype=torch.float64),
        target=target,
    )


def test_layernorm_dtype_no_bias(target: str) -> None:
    class LayerNormDtypeNoBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                16, elementwise_affine=True, bias=False, dtype=torch.float64
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(
        LayerNormDtypeNoBiasNet().eval(),
        torch.randn(2, 3, 16, dtype=torch.float64),
        target=target,
    )


def test_layernorm_all_params(target: str) -> None:
    class LayerNormAllParamsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(
                [3, 16],
                eps=1e-03,
                elementwise_affine=True,
                bias=False,
                dtype=torch.float64,
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(
        LayerNormAllParamsNet().eval(),
        torch.randn(2, 3, 16, dtype=torch.float64),
        target=target,
    )


# --- LayerNorm1d ---


def test_layernorm1d_simple(target: str) -> None:
    class LayerNorm1dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(16)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNorm1dSimpleNet().eval(), torch.randn(2, 3, 16), target=target)


def test_layernorm1d_eps(target: str) -> None:
    class LayerNorm1dEpsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm(16, eps=1e-04)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNorm1dEpsNet().eval(), torch.randn(2, 3, 16), target=target)


# --- LayerNorm2d ---


def test_layernorm2d_simple(target: str) -> None:
    class LayerNorm2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm([16, 16])

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNorm2dSimpleNet().eval(), torch.randn(2, 3, 16, 16), target=target)


def test_layernorm2d_eps(target: str) -> None:
    class LayerNorm2dEpsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm([16, 16], eps=1e-04)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNorm2dEpsNet().eval(), torch.randn(2, 3, 16, 16), target=target)


# --- LayerNorm3d ---


def test_layernorm3d_simple(target: str) -> None:
    class LayerNorm3dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm([16, 16, 16])

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNorm3dSimpleNet().eval(), torch.randn(2, 3, 16, 16, 16), target=target)


def test_layernorm3d_eps(target: str) -> None:
    class LayerNorm3dEpsNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layernorm: nn.LayerNorm = nn.LayerNorm([16, 16, 16], eps=1e-04)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.layernorm(input)

    check(LayerNorm3dEpsNet().eval(), torch.randn(2, 3, 16, 16, 16), target=target)
