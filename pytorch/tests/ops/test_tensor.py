import torch
import torch.nn as nn

from tests import check

# --- new_full ---


def test_new_full_simple(target: str) -> None:
    class NewFullSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.new_full((3, 4), 3.141592)

    check(
        NewFullSimpleNet(),
        torch.ones((2,)),
        target=target,
    )


def test_new_full_dtype(target: str) -> None:
    class NewFullDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.new_full((3, 4), 3.141592)

    check(
        NewFullDtypeNet(),
        torch.ones((2,), dtype=torch.float64),
        target=target,
    )


def test_new_full_dtype_change(target: str) -> None:
    class NewFullDtypeChangeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.new_full((3, 4), 3.141592, dtype=torch.float32)

    check(
        NewFullDtypeChangeNet(),
        torch.ones((2,), dtype=torch.float64),
        target=target,
    )


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


# --- expand ---


def test_expand_simple(target: str) -> None:
    class ExpandSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.expand(3, 4)

    check(ExpandSimpleNet(), torch.tensor([[1], [2], [3]]), target=target)


def test_expand_neg_dim(target: str) -> None:
    class ExpandNegDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.expand(-1, 4)

    check(ExpandNegDimNet(), torch.tensor([[1], [2], [3]]), target=target)


# --- expand_as ---


def test_expand_as_simple(target: str) -> None:
    class ExpandAsSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
            return input.expand_as(other)

    check(
        ExpandAsSimpleNet(),
        *(torch.tensor([[1], [2], [3]]), torch.randn(3, 4)),
        target=target
    )


# --- fill_ ---


def test_fill__simple(target: str) -> None:
    class Fill_SimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.fill_(3.141592)

    check(Fill_SimpleNet(), torch.ones(2, 3), target=target)


# --- sigmoid ---


def test_sigmoid_simple(target: str) -> None:
    class SigmoidSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.sigmoid()

    check(SigmoidSimpleNet(), torch.randn(4), target=target)


# --- softmax ---


def test_softmax_simple(target: str) -> None:
    class SoftmaxSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.softmax(1)

    check(SoftmaxSimpleNet(), torch.randn(2, 3), target=target)


def test_softmax_dtype(target: str) -> None:
    class SoftmaxDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.softmax(1, dtype=torch.float64)

    check(SoftmaxDtypeNet(), torch.randn(2, 3), target=target)


# --- to ---


def test_to_simple(target: str) -> None:
    class ToSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.to(torch.float64)

    check(ToSimpleNet(), torch.randn(2, 2), target=target)


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
