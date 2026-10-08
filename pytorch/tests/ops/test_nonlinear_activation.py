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


# --- Clamp ---


class ClampNet(nn.Module):
    def __init__(self, min: float | int | None, max: float | int | None) -> None:
        super().__init__()
        self.min: float | int | None = min
        self.max: float | int | None = max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.clamp(input, self.min, self.max)


def _clamp_special_values() -> torch.Tensor:
    return torch.tensor(
        [torch.nan, torch.inf, -torch.inf, 0.0, -0.0, 0.5, -1.0, 2.0, 1e30, -1e30]
    )


def test_clamp_simple(target: str) -> None:
    check(ClampNet(-0.5, 0.5), torch.randn(8), target=target)


def test_clamp_min_only(target: str) -> None:
    check(ClampNet(0.0, None), torch.randn(8), target=target)


def test_clamp_max_only(target: str) -> None:
    class ClampMaxNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.clamp(input, max=0.25)

    check(ClampMaxNet(), torch.randn(8), target=target)


def test_clamp_int_bounds_float_input(target: str) -> None:
    check(ClampNet(-1, 1), torch.randn(8), target=target)


def test_clamp_min_greater_than_max(target: str) -> None:
    check(ClampNet(2.0, 1.0), _clamp_special_values(), target=target, equal_nan=True)


def test_clamp_special_values(target: str) -> None:
    check(ClampNet(0.0, 1.0), _clamp_special_values(), target=target, equal_nan=True)


def test_clamp_nan_min(target: str) -> None:
    check(
        ClampNet(float("nan"), None),
        _clamp_special_values(),
        target=target,
        equal_nan=True,
    )


def test_clamp_nan_max(target: str) -> None:
    check(
        ClampNet(None, float("nan")),
        _clamp_special_values(),
        target=target,
        equal_nan=True,
    )


def test_clamp_inf_bounds(target: str) -> None:
    check(
        ClampNet(-float("inf"), float("inf")),
        _clamp_special_values(),
        target=target,
        equal_nan=True,
    )


def test_clamp_inplace(target: str) -> None:
    class ClampInplaceNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # check() feeds the same input to the reference and the compiled model.
            return input.clone().clamp_(-0.5, 0.5)

    check(ClampInplaceNet(), torch.randn(8), target=target)


def test_clamp_multidim(target: str) -> None:
    check(ClampNet(-0.5, 0.5), torch.randn(2, 3, 4, 5), target=target)


def test_clamp_float64(target: str) -> None:
    check(ClampNet(-0.5, 0.5), torch.randn(3, 5, dtype=torch.float64), target=target)


def test_clamp_int32(target: str) -> None:
    check(
        ClampNet(-3, 4),
        torch.randint(-10, 10, (16,), dtype=torch.int32),
        target=target,
    )


def test_clamp_int64_min_only(target: str) -> None:
    check(ClampNet(0, None), torch.randint(-10, 10, (16,)), target=target)


def test_clamp_uint8(target: str) -> None:
    check(
        ClampNet(10, 200),
        torch.randint(0, 256, (16,), dtype=torch.uint8),
        target=target,
    )


# --- Hardsigmoid ---


class HardsigmoidNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hardsigmoid: nn.Hardsigmoid = nn.Hardsigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.hardsigmoid(input)


def test_hardsigmoid_simple(target: str) -> None:
    check(HardsigmoidNet(), torch.randn(8) * 4, target=target)


def test_hardsigmoid_boundaries(target: str) -> None:
    x = torch.tensor([-4.0, -3.0, -2.999, -1.0, 0.0, 1.0, 2.999, 3.0, 4.0])
    check(HardsigmoidNet(), x, target=target)


def test_hardsigmoid_special_values(target: str) -> None:
    x = torch.tensor([torch.nan, torch.inf, -torch.inf, 0.0, -0.0, 1e30, -1e30])
    check(HardsigmoidNet(), x, target=target, equal_nan=True)


def test_hardsigmoid_inplace(target: str) -> None:
    class HardsigmoidInplaceNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hardsigmoid: nn.Hardsigmoid = nn.Hardsigmoid(inplace=True)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            # check() feeds the same input to the reference and the compiled model.
            return self.hardsigmoid(input.clone())

    check(HardsigmoidInplaceNet(), torch.randn(8) * 4, target=target)


def test_hardsigmoid_multidim(target: str) -> None:
    check(HardsigmoidNet(), torch.randn(2, 3, 4, 5) * 4, target=target)


def test_hardsigmoid_float64(target: str) -> None:
    check(HardsigmoidNet(), torch.randn(3, 5, dtype=torch.float64) * 4, target=target)


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
