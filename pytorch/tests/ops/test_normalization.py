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
