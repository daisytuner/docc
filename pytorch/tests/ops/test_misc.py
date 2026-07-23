import torch
import torch.nn as nn

from tests import check

# --- clone ---


def test_clone_simple(target: str) -> None:
    class CloneSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.clone(input)

    check(CloneSimpleNet(), torch.randn(2, 3), target=target)


def test_clone_memory_format_contiguous(target: str) -> None:
    class CloneMemoryFormatContiguousNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.clone(input, memory_format=torch.contiguous_format)

    check(CloneMemoryFormatContiguousNet(), torch.randn(2, 3), target=target)
