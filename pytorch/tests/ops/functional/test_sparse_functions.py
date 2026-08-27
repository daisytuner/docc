import torch
import torch.nn as nn
import torch.nn.functional as F

from tests import check

# --- embedding ---


def test_embedding_functional(target: str) -> None:
    class EmbeddingFunctionalNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(indices, weight)

    check(
        EmbeddingFunctionalNet(),
        *(torch.randn(10, 4), torch.tensor([1, 3, 5, 0])),
        target=target,
    )


def test_embedding_2d_indices(target: str) -> None:
    class Embedding2dIndicesNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(indices, weight)

    check(
        Embedding2dIndicesNet(),
        *(torch.randn(10, 4), torch.tensor([[1, 3], [5, 0], [2, 7]])),
        target=target,
    )


def test_embedding_padding_idx(target: str) -> None:
    class EmbeddingPaddingIdxNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(indices, weight, padding_idx=0)

    check(
        EmbeddingPaddingIdxNet(),
        *(torch.randn(10, 4), torch.tensor([1, 0, 5, 0])),
        target=target,
    )


def test_embedding_padding_idx_positive(target: str) -> None:
    class EmbeddingPaddingIdxPositiveNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(indices, weight, padding_idx=2)

    check(
        EmbeddingPaddingIdxPositiveNet(),
        *(torch.randn(10, 4), torch.tensor([1, 2, 5, 0])),
        target=target,
    )


def test_embedding_padding_idx_negative(target: str) -> None:
    class EmbeddingPaddingIdxNegativeNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(indices, weight, padding_idx=-1)

    check(
        EmbeddingPaddingIdxNegativeNet(),
        *(torch.randn(10, 4), torch.tensor([1, 3, 5, 0])),
        target=target,
    )


def test_embedding_scale_grad_by_freq(target: str) -> None:
    class EmbeddingScaleGradByFreqNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(indices, weight, scale_grad_by_freq=True)

    check(
        EmbeddingScaleGradByFreqNet(),
        *(torch.randn(10, 4), torch.tensor([1, 3, 5, 0])),
        target=target,
    )


def test_embedding_sparse(target: str) -> None:
    class EmbeddingSparseNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(indices, weight, sparse=True)

    check(
        EmbeddingSparseNet(),
        *(torch.randn(10, 4), torch.tensor([1, 3, 5, 0])),
        target=target,
    )


def test_embedding_all_params(target: str) -> None:
    class EmbeddingAllParamsNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            return F.embedding(
                indices,
                weight,
                padding_idx=2,
                scale_grad_by_freq=True,
                sparse=True,
            )

    check(
        EmbeddingAllParamsNet(),
        *(torch.randn(10, 4), torch.tensor([1, 2, 5, 0])),
        target=target,
    )


def test_embedding_functional_max_norm(target: str) -> None:
    class EmbeddingFunctionalMaxNormNet(nn.Module):
        def forward(self, weight: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            # Clone so the in-place renorm does not mutate the shared input.
            return F.embedding(indices, weight.clone(), max_norm=1.0)

    check(
        EmbeddingFunctionalMaxNormNet(),
        *(torch.randn(10, 4), torch.tensor([1, 3, 5, 0])),
        target=target,
    )
