import torch
import torch.nn as nn

from tests import check

# --- clone ---


def test_clone_simple(target: str) -> None:
    class CloneSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.clone()

    check(CloneSimpleNet(), torch.randn(2, 3), target=target)


def test_clone_memory_format_contiguous(target: str) -> None:
    class CloneMemoryFormatContiguousNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.clone(memory_format=torch.contiguous_format)

    check(CloneMemoryFormatContiguousNet(), torch.randn(2, 3), target=target)


# --- view ---


def test_view_shape_simple(target: str) -> None:
    class TensorMutatingViewShapeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.view(16)

    check(TensorMutatingViewShapeSimpleNet(), torch.randn(4, 4), target=target)


def test_view_shape_neg(target: str) -> None:
    class TensorMutatingViewShapeNegNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.view(-1, 8)

    check(TensorMutatingViewShapeNegNet(), torch.randn(4, 4), target=target)


def test_view_shape_identity(target: str) -> None:
    class TensorMutatingViewShapeIdentityNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.view(1, 3, 2, 4)

    check(TensorMutatingViewShapeIdentityNet(), torch.randn(1, 3, 2, 4), target=target)
