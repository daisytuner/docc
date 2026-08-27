import torch
import torch.nn as nn

from tests import check

# --- Embedding ---


def test_embedding_simple(target: str) -> None:
    class EmbeddingSimpleNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(EmbeddingSimpleNet(), torch.tensor([1, 3, 5, 0]), target=target)


def test_embedding_padding_idx(target: str) -> None:
    class EmbeddingPaddingIdxNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, padding_idx=0)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingPaddingIdxNet(),
        torch.tensor([1, 0, 5, 0]),
        target=target,
    )


def test_embedding_max_norm(target: str) -> None:
    class EmbeddingMaxNormNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, max_norm=1.0)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNormNet(),
        torch.tensor([1, 3, 5, 0]),
        target=target,
    )


def test_embedding_max_norm_norm_type(target: str) -> None:
    class EmbeddingMaxNormNormTypeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, max_norm=1.5, norm_type=1.0)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNormNormTypeNet(),
        torch.tensor([1, 3, 5, 0]),
        target=target,
    )


def test_embedding_all_params_with_norm(target: str) -> None:
    class EmbeddingAllParamsWithNormNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(
                10,
                4,
                padding_idx=2,
                max_norm=1.0,
                norm_type=2.0,
                scale_grad_by_freq=True,
                sparse=True,
            )

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingAllParamsWithNormNet(),
        torch.tensor([1, 2, 5, 0]),
        target=target,
    )


def test_embedding_max_norm_norm_type_3(target: str) -> None:
    class EmbeddingMaxNormNormType3Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, max_norm=1.0, norm_type=3.0)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNormNormType3Net(),
        torch.tensor([1, 3, 5, 0]),
        target=target,
    )


def test_embedding_max_norm_norm_type_inf(target: str) -> None:
    class EmbeddingMaxNormNormTypeInfNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, max_norm=1.0, norm_type=float("inf"))

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNormNormTypeInfNet(),
        torch.tensor([1, 3, 5, 0]),
        target=target,
    )


def test_embedding_max_norm_norm_type_fractional(target: str) -> None:
    class EmbeddingMaxNormNormTypeFractionalNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, max_norm=1.0, norm_type=0.5)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNormNormTypeFractionalNet(),
        torch.tensor([1, 3, 5, 0]),
        target=target,
    )


def test_embedding_max_norm_no_renorm(target: str) -> None:
    class EmbeddingMaxNormNoRenormNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # max_norm is large enough that no row is renormalized.
            self.embedding = nn.Embedding(10, 4, max_norm=100.0)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNormNoRenormNet(),
        torch.tensor([1, 3, 5, 0]),
        target=target,
    )


def test_embedding_max_norm_repeated_indices(target: str) -> None:
    class EmbeddingMaxNormRepeatedIndicesNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, max_norm=1.0)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNormRepeatedIndicesNet(),
        torch.tensor([3, 1, 3, 3, 1]),
        target=target,
    )


def test_embedding_max_norm_2d_indices(target: str) -> None:
    class EmbeddingMaxNorm2dIndicesNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(10, 4, max_norm=1.0)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.embedding(indices)

    check(
        EmbeddingMaxNorm2dIndicesNet(),
        torch.tensor([[1, 3], [5, 0], [2, 7]]),
        target=target,
    )
