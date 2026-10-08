import torch
import torch.nn as nn

from tests import check

# --- Linear ---


def test_linear_simple(target: str) -> None:
    class LinearSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(20, 30, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    check(LinearSimpleNet(), torch.randn(128, 20), target=target)


def test_linear_bias(target: str) -> None:
    class LinearBiasNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(20, 30)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    check(LinearBiasNet(), torch.randn(128, 20), target=target)


def test_linear_half(target: str) -> None:
    class LinearHalfNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(20, 30, bias=False)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    # fp16 GEMM accumulates in low precision, so compare with relaxed tolerances.
    check(
        LinearHalfNet().half(),
        torch.randn(128, 20, dtype=torch.float16),
        rtol=1e-2,
        atol=1e-2,
        target=target,
    )


def test_linear_bias_half(target: str) -> None:
    class LinearBiasHalfNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(20, 30)

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return self.linear(input)

    # fp16 GEMM accumulates in low precision, so compare with relaxed tolerances.
    check(
        LinearBiasHalfNet().half(),
        torch.randn(128, 20, dtype=torch.float16),
        rtol=1e-2,
        atol=1e-2,
        target=target,
    )

