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


# --- embedding ---


def test_embedding_simple(target: str) -> None:
    class EmbeddingSimpleNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return torch.embedding(weight, indices)

    check(
        EmbeddingSimpleNet(),
        *(torch.randn(10, 4), torch.tensor([1, 3, 5, 0])),
        target=target,
    )


def test_embedding_padding_idx(target: str) -> None:
    class EmbeddingPaddingIdxNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return torch.embedding(weight, indices, padding_idx=0)

    check(
        EmbeddingPaddingIdxNet(),
        *(torch.randn(10, 4), torch.tensor([1, 0, 5, 0])),
        target=target,
    )


# --- embedding_renorm_ ---


def test_embedding_renorm__simple(target: str) -> None:
    class EmbeddingRenorm_SimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            weight: torch.Tensor = torch.view_copy(input, input.shape)
            return torch.embedding_renorm_(weight, indices, 1.0, 2.0)

    check(
        EmbeddingRenorm_SimpleNet(),
        *(torch.randn(10, 4), torch.tensor([2, 3, 6, 0])),
        target=target,
    )


def test_embedding_renorm__norm_type_1(target: str) -> None:
    class EmbeddingRenorm_NormType1Net(nn.Module):
        def forward(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            weight: torch.Tensor = torch.view_copy(input, input.shape)
            return torch.embedding_renorm_(weight, indices, 1.0, 1.0)

    check(
        EmbeddingRenorm_NormType1Net(),
        *(torch.randn(10, 4), torch.tensor([2, 3, 6, 0])),
        target=target,
    )


def test_embedding_renorm__norm_type_3(target: str) -> None:
    class EmbeddingRenorm_NormType3Net(nn.Module):
        def forward(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            weight: torch.Tensor = torch.view_copy(input, input.shape)
            return torch.embedding_renorm_(weight, indices, 1.0, 3.0)

    check(
        EmbeddingRenorm_NormType3Net(),
        *(torch.randn(10, 4), torch.tensor([2, 3, 6, 0])),
        target=target,
    )


def test_embedding_renorm__norm_type_half(target: str) -> None:
    class EmbeddingRenorm_NormTypeHalfNet(nn.Module):
        def forward(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            weight: torch.Tensor = torch.view_copy(input, input.shape)
            return torch.embedding_renorm_(weight, indices, 1.0, 0.5)

    check(
        EmbeddingRenorm_NormTypeHalfNet(),
        *(torch.randn(10, 4), torch.tensor([2, 3, 6, 0])),
        target=target,
    )


def test_embedding_renorm__norm_type_inf(target: str) -> None:
    class EmbeddingRenorm_NormTypeInf(nn.Module):
        def forward(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            weight: torch.Tensor = torch.view_copy(input, input.shape)
            return torch.embedding_renorm_(weight, indices, 1.0, torch.inf)

    check(
        EmbeddingRenorm_NormTypeInf(),
        *(torch.randn(10, 4), torch.tensor([2, 3, 6, 0])),
        target=target,
    )


def test_embedding_renorm__2d_indices(target: str) -> None:
    class EmbeddingRenorm_SimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            weight: torch.Tensor = torch.view_copy(input, input.shape)
            return torch.embedding_renorm_(weight, indices, 1.0, 2.0)

    check(
        EmbeddingRenorm_SimpleNet(),
        *(torch.randn(10, 4), torch.tensor([[2, 3, 6, 0], [8, 7, 4, 9]])),
        target=target,
    )
