import torch
import torch.nn as nn
import pytest

from tests import check

# --- argwhere ---


def test_argwhere_simple(target: str) -> None:
    class ArgwhereSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.argwhere(input)

    check(ArgwhereSimpleNet(), torch.tensor([1, 0, 1]), target=target)


def test_argwhere_bigger(target: str) -> None:
    class ArgwhereBiggerNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.argwhere(input)

    check(ArgwhereBiggerNet(), torch.tensor([[1, 0, 1], [0, 1, 1]]), target=target)


# --- cat ---


def test_cat_simple(target: str) -> None:
    class CatSimpleNet(nn.Module):
        def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
            return torch.cat((input1, input2), 0)

    check(CatSimpleNet(), *(torch.randn(2, 3), torch.randn(2, 3)), target=target)


def test_cat_dim_1(target: str) -> None:
    class CatDim1Net(nn.Module):
        def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
            return torch.cat((input1, input2), 1)

    check(CatDim1Net(), *(torch.randn(2, 3), torch.randn(2, 3)), target=target)


def test_cat_dim_neg1(target: str) -> None:
    class CatDimNeg1Net(nn.Module):
        def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
            return torch.cat((input1, input2), -1)

    check(CatDimNeg1Net(), *(torch.randn(2, 3), torch.randn(2, 3)), target=target)


def test_cat_many(target: str) -> None:
    class CatManyNet(nn.Module):
        def forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
            return torch.cat(inputs, 0)

    x = torch.randn(2, 3)
    check(CatManyNet(), (x, x, x, x, x, x, x, x, x, x), target=target)


# --- expand_copy ---


def test_expand_copy_simple(target: str) -> None:
    class ExpandCopySimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.expand_copy(input, (3, 4))

    check(ExpandCopySimpleNet(), torch.tensor([[1], [2], [3]]), target=target)


def test_expand_copy_neg_dim(target: str) -> None:
    class ExpandCopyNegDimNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.expand_copy(input, (-1, 4))

    check(ExpandCopyNegDimNet(), torch.tensor([[1], [2], [3]]), target=target)


def test_expand_copy_implicit(target: str) -> None:
    class ExpandCopyImplicitNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.expand_copy(input, (3, 4), implicit=True)

    check(ExpandCopyImplicitNet(), torch.tensor([[1], [2], [3]]), target=target)


# --- fill ---


@pytest.mark.minimum_pytorch_version((2, 13, 0))
def test_fill_simple(target: str) -> None:
    class FillSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.fill(input, 3.141592)

    check(FillSimpleNet(), torch.ones(2, 3), target=target)


# --- index ---


def test_index_1d(target: str) -> None:
    class Index1dNet(nn.Module):
        def forward(self, input: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return input[idx]

    check(
        Index1dNet(),
        *(torch.randn(5, 4), torch.tensor([0, 2, 3])),
        target=target,
    )


def test_index_repeated(target: str) -> None:
    class IndexRepeatedNet(nn.Module):
        def forward(self, input: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return input[idx]

    check(
        IndexRepeatedNet(),
        *(torch.randn(5, 4), torch.tensor([4, 0, 4, 1, 0])),
        target=target,
    )


def test_index_dim_1(target: str) -> None:
    class IndexDim1Net(nn.Module):
        def forward(self, input: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return input[:, idx]

    check(
        IndexDim1Net(),
        *(torch.randn(5, 4), torch.tensor([0, 2])),
        target=target,
    )


def test_index_two_tensors(target: str) -> None:
    class IndexTwoTensorsNet(nn.Module):
        def forward(
            self, input: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor
        ) -> torch.Tensor:
            return input[rows, cols]

    check(
        IndexTwoTensorsNet(),
        *(torch.randn(5, 4), torch.tensor([0, 1, 4]), torch.tensor([2, 3, 1])),
        target=target,
    )


def test_index_3d(target: str) -> None:
    class Index3dNet(nn.Module):
        def forward(self, input: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return input[idx]

    check(
        Index3dNet(),
        *(torch.randn(4, 3, 5), torch.tensor([0, 2, 3])),
        target=target,
    )


# --- permute ---


def test_permute_simple(target: str) -> None:
    class PermuteSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.permute(input, (2, 0, 1))

    check(PermuteSimpleNet(), torch.randn(2, 3, 5), target=target)


def test_view_permute(target: str) -> None:
    class ViewPermuteNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            input = input.view(2, 3, 5)
            return torch.permute(input, (2, 0, 1))

    check(ViewPermuteNet(), torch.randn(2, 3, 5), target=target)


# --- slice ---


def test_slice_simple(target: str) -> None:
    class SliceSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[1:3]

    check(SliceSimpleNet(), torch.randn(5, 4), target=target)


def test_slice_default(target: str) -> None:
    class SliceDefaultNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:]

    check(SliceDefaultNet(), torch.randn(5, 4), target=target)


def test_slice_dim_1(target: str) -> None:
    class SliceDim1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, 1:3]

    check(SliceDim1Net(), torch.randn(5, 4), target=target)


def test_slice_dim_neg1(target: str) -> None:
    class SliceDimNeg1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.slice_copy(input, -1, 1, 3)

    check(SliceDimNeg1Net(), torch.randn(5, 4), target=target)


def test_slice_start_only(target: str) -> None:
    class SliceStartOnlyNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[2:]

    check(SliceStartOnlyNet(), torch.randn(5, 4), target=target)


def test_slice_end_only(target: str) -> None:
    class SliceEndOnlyNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:3]

    check(SliceEndOnlyNet(), torch.randn(5, 4), target=target)


def test_slice_neg_start(target: str) -> None:
    class SliceNegStartNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[-3:]

    check(SliceNegStartNet(), torch.randn(5, 4), target=target)


def test_slice_neg_end(target: str) -> None:
    class SliceNegEndNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:-1]

    check(SliceNegEndNet(), torch.randn(5, 4), target=target)


def test_slice_step_2(target: str) -> None:
    class SliceStep2Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[0:5:2]

    check(SliceStep2Net(), torch.randn(5, 4), target=target)


def test_slice_full_range_step(target: str) -> None:
    class SliceFullRangeStepNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[::2]

    check(SliceFullRangeStepNet(), torch.randn(6, 4), target=target)


def test_slice_3d_dim_0(target: str) -> None:
    class Slice3dDim0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[1:3]

    check(Slice3dDim0Net(), torch.randn(4, 3, 5), target=target)


def test_slice_3d_dim_1(target: str) -> None:
    class Slice3dDim1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, 1:3]

    check(Slice3dDim1Net(), torch.randn(4, 3, 5), target=target)


def test_slice_3d_dim_2(target: str) -> None:
    class Slice3dDim2Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, :, 1:4]

    check(Slice3dDim2Net(), torch.randn(4, 3, 5), target=target)


def test_slice_4d_dim_0(target: str) -> None:
    class Slice4dDim0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[1:3]

    check(Slice4dDim0Net(), torch.randn(4, 3, 5, 6), target=target)


def test_slice_4d_dim_1(target: str) -> None:
    class Slice4dDim1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, 1:3]

    check(Slice4dDim1Net(), torch.randn(4, 3, 5, 6), target=target)


def test_slice_4d_dim_2(target: str) -> None:
    class Slice4dDim2Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, :, 1:4]

    check(Slice4dDim2Net(), torch.randn(4, 3, 5, 6), target=target)


def test_slice_4d_dim_3(target: str) -> None:
    class Slice4dDim3Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input[:, :, :, 2:5]

    check(Slice4dDim3Net(), torch.randn(4, 3, 5, 6), target=target)


# --- squeeze ---


def test_squeeze_simple(target: str) -> None:
    class SqueezeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input)

    check(SqueezeSimpleNet(), torch.randn(2, 1, 2, 1, 2), target=target)


def test_squeeze_0(target: str) -> None:
    class Squeeze0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input, 0)

    check(Squeeze0Net(), torch.randn(2, 1, 2, 1, 2), target=target)


def test_squeeze_1(target: str) -> None:
    class Squeeze1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input, 1)

    check(Squeeze1Net(), torch.randn(2, 1, 2, 1, 2), target=target)


def test_squeeze_tuple(target: str) -> None:
    class SqueezeTupleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.squeeze(input, (1, 2, 3))

    check(SqueezeTupleNet(), torch.randn(2, 1, 2, 1, 2), target=target)


# --- unsqueeze ---


def test_unsqueeze_0(target: str) -> None:
    class Unsqueeze0Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.unsqueeze(input, 0)

    check(Unsqueeze0Net(), torch.tensor([1, 2, 3, 4]), target=target)


def test_unsqueeze_1(target: str) -> None:
    class Unsqueeze1Net(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.unsqueeze(input, 1)

    check(Unsqueeze1Net(), torch.tensor([1, 2, 3, 4]), target=target)


# --- where ---


def test_where_simple(target: str) -> None:
    class WhereSimpleNet(nn.Module):
        def forward(
            self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor
        ) -> torch.Tensor:
            return torch.where(condition, input, other)

    check(
        WhereSimpleNet(),
        *(
            torch.tensor([True, False]),
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        ),
        target=target,
    )


def test_where_constants(target: str) -> None:
    class WhereConstantsNet(nn.Module):
        def forward(
            self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor
        ) -> torch.Tensor:
            return torch.where(condition, input, other)

    check(
        WhereConstantsNet(),
        *(
            torch.tensor([[True, False], [False, True]]),
            torch.tensor(1.0),
            torch.tensor(0.0),
        ),
        target=target,
    )


def test_where_condition_broadcast(target: str) -> None:
    class WhereConditionBroadcastNet(nn.Module):
        def forward(
            self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor
        ) -> torch.Tensor:
            return torch.where(condition, input, other)

    check(
        WhereConditionBroadcastNet(),
        *(
            torch.tensor([[True, False]]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ),
        target=target,
    )


def test_where_input_broadcast(target: str) -> None:
    class WhereConditionBroadcastNet(nn.Module):
        def forward(
            self, condition: torch.Tensor, input: torch.Tensor, other: torch.Tensor
        ) -> torch.Tensor:
            return torch.where(condition, input, other)

    check(
        WhereConditionBroadcastNet(),
        *(
            torch.tensor([[True, False], [False, True]]),
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[5.0, 6.0]]),
        ),
        target=target,
    )
