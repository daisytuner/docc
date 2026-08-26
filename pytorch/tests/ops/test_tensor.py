import torch
import torch.nn as nn
import pytest

from tests import check

# --- alias ---


def test_alias_simple(target: str) -> None:
    class AliasSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.alias.default(input)

    check(AliasSimpleNet(), torch.randn(2, 3), target=target)


def test_alias_chained(target: str) -> None:
    class AliasChainedNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.alias.default(input) + 1.0

    check(AliasChainedNet(), torch.randn(2, 3), target=target)


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


# --- detach ---


def test_detach_simple(target: str) -> None:
    class DetachSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.detach()

    check(DetachSimpleNet(), torch.randn(2, 3), target=target)


def test_detach_chained(target: str) -> None:
    class DetachChainedNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input.detach() + 1.0

    check(DetachChainedNet(), torch.randn(2, 3), target=target)


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


# --- "slicing" ---


def test_slicing_simple(target: str) -> None:
    class SlicingSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[1:7:2]

    check(SlicingSimpleNet(), torch.arange(10), target=target)


def test_slicing_negative_start(target: str) -> None:
    class SlicingNegativeStartNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[-2:10]

    check(SlicingNegativeStartNet(), torch.arange(10), target=target)


def test_slicing_unbound_start(target: str) -> None:
    class SlicingUnboundStartNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:5]

    check(SlicingUnboundStartNet(), torch.arange(10), target=target)


def test_slicing_unbound_end(target: str) -> None:
    class SlicingUnboundEndNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[5:]

    check(SlicingUnboundEndNet(), torch.arange(10), target=target)


def test_slicing_assumed_dim(target: str) -> None:
    class SlicingAssumedDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[1:2]

    check(
        SlicingAssumedDimNet(),
        torch.tensor([[[1], [2], [3]], [[4], [5], [6]]]),
        target=target,
    )


def test_slicing_ellipsis(target: str) -> None:
    class SlicingEllipsisNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[..., 1:4]

    check(SlicingEllipsisNet(), torch.arange(10).reshape(2, 5), target=target)


def test_slicing_colon(target: str) -> None:
    class SlicingColonNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, 1:4]

    check(SlicingColonNet(), torch.arange(10).reshape(2, 5), target=target)


@pytest.mark.skip(reason="Needs support for aten.select.int")
def test_slicing_select_ellipsis(target: str) -> None:
    class SlicingSelectEllipsisNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[..., 0]

    check(
        SlicingSelectEllipsisNet(),
        torch.tensor([[[1], [2], [3]], [[4], [5], [6]]]),
        target=target,
    )


@pytest.mark.skip(reason="Needs support for aten.select.int")
def test_slicing_select_colon(target: str) -> None:
    class SlicingSelectColonNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, :, 0]

    check(
        SlicingSelectColonNet(),
        torch.tensor([[[1], [2], [3]], [[4], [5], [6]]]),
        target=target,
    )


def test_slicing_newaxis(target: str) -> None:
    class SlicingNewaxisNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, torch.newaxis, :, :]

    check(
        SlicingNewaxisNet(),
        torch.tensor([[[1], [2], [3]], [[4], [5], [6]]]),
        target=target,
    )


def test_slicing_None(target: str) -> None:
    class SlicingNoneNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, None, :, :]

    check(
        SlicingNoneNet(),
        torch.tensor([[[1], [2], [3]], [[4], [5], [6]]]),
        target=target,
    )


def test_slicing_multi(target: str) -> None:
    class SlicingMultiNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[1:4, 1:3]

    check(SlicingMultiNet(), torch.arange(20).reshape(5, 4), target=target)
