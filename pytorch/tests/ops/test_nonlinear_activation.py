import torch
import torch.nn as nn

from tests import check

# --- ReLU ---


def test_relu_simple(target: str) -> None:
    class ReLUSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu: nn.ReLU = nn.ReLU()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.relu(input)

    check(ReLUSimpleNet(), torch.randn(2), target=target)


# --- GELU ---


def test_gelu_simple(target: str) -> None:
    class GELUSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gelu: nn.GELU = nn.GELU()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.gelu(input)

    check(GELUSimpleNet(), torch.randn(2), target=target)


def test_gelu_tanh_approx(target: str) -> None:
    class GELUSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gelu: nn.GELU = nn.GELU(approximate="tanh")

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.gelu(input)

    check(GELUSimpleNet(), torch.randn(2), target=target)


# --- ELU ---


class EluNet(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.elu: nn.ELU = nn.ELU(alpha)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.elu(input)


def _elu_special_values() -> torch.Tensor:
    return torch.tensor(
        [torch.nan, torch.inf, -torch.inf, 0.0, -0.0, 1.0, -1.0, 1e-30, -1e-30]
    )


def test_elu_simple(target: str) -> None:
    check(EluNet(), torch.randn(8), target=target)


def test_elu_alpha(target: str) -> None:
    check(EluNet(0.5), torch.randn(8), target=target)


def test_elu_alpha_int(target: str) -> None:
    class EluIntAlphaNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return nn.functional.elu(input, 2)

    check(EluIntAlphaNet(), torch.randn(8), target=target)


def test_elu_alpha_inf(target: str) -> None:
    check(EluNet(float("inf")), _elu_special_values(), target=target, equal_nan=True)


def test_elu_special_values(target: str) -> None:
    check(EluNet(), _elu_special_values(), target=target, equal_nan=True)


def test_elu_inplace(target: str) -> None:
    class EluInplaceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.elu: nn.ELU = nn.ELU(inplace=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # check() feeds the same input to the reference and the compiled model.
            return self.elu(input.clone())

    check(EluInplaceNet(), torch.randn(8), target=target)


def test_elu_multidim(target: str) -> None:
    check(EluNet(), torch.randn(2, 3, 4, 5), target=target)


def test_elu_float64(target: str) -> None:
    check(EluNet(), torch.randn(3, 5, dtype=torch.float64), target=target)


def test_selu(target: str) -> None:
    class SELUNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.selu: nn.SELU = nn.SELU()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.selu(input)

    check(SELUNet(), torch.randn(8), target=target)
    check(SELUNet(), _elu_special_values(), target=target, equal_nan=True)


def test_elu_input_scale(target: str) -> None:
    class EluInputScaleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.elu.default(input, 1.5, 2.0, 0.5)

    check(EluInputScaleNet(), torch.randn(8), target=target)


# --- Softmax ---


def test_softmax_simple(target: str) -> None:
    class SoftmaxSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.softmax: nn.Softmax = nn.Softmax(dim=1)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.softmax(input)

    check(SoftmaxSimpleNet(), torch.randn(2, 3), target=target)


# --- Softmax2d ---


def test_softmax2d_simple(target: str) -> None:
    class Softmax2dSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.softmax2d: nn.Softmax2d = nn.Softmax2d()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.softmax2d(input)

    check(Softmax2dSimpleNet(), torch.randn(2, 3, 12, 13), target=target)


# --- Sigmoid ---


def test_sigmoid_simple(target: str) -> None:
    class SigmoidSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sigmoid: nn.Sigmoid = nn.Sigmoid()

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.sigmoid(input)

    check(SigmoidSimpleNet(), torch.randn(4), target=target)
