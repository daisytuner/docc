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
