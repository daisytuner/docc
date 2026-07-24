import torch
import torch.nn as nn

from tests import check

# --- addmm ---


def test_addmm_simple(target: str) -> None:
    class AddMMSimpleNet(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2)

    check(
        AddMMSimpleNet(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_broadcast(target: str) -> None:
    class AddMMBroadcastNet(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2)

    check(
        AddMMBroadcastNet(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_alpha(target: str) -> None:
    class AddMMAlphaNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            alpha: float,
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=alpha)

    check(
        AddMMAlphaNet(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3), 2),
        target=target
    )


def test_addmm_alpha_constant_float(target: str) -> None:
    class AddMMAlphaConstantFloat(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=2.0)

    check(
        AddMMAlphaConstantFloat(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_alpha_constant_int(target: str) -> None:
    class AddMMAlphaConstantInt(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=2)

    check(
        AddMMAlphaConstantInt(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_broadcast_alpha(target: str) -> None:
    class AddMMBroadcastAlphaNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            alpha: float,
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=alpha)

    check(
        AddMMBroadcastAlphaNet(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3), 2),
        target=target
    )


def test_addmm_broadcast_alpha_constant_float(target: str) -> None:
    class AddMMBroadcastAlphaConstantFloat(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=2.0)

    check(
        AddMMBroadcastAlphaConstantFloat(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_broadcast_alpha_constant_int(target: str) -> None:
    class AddMMBroadcastAlphaConstantInt(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=2)

    check(
        AddMMBroadcastAlphaConstantInt(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_beta(target: str) -> None:
    class AddMMBetaNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            beta: float,
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, beta=beta)

    check(
        AddMMBetaNet(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3), 2),
        target=target
    )


def test_addmm_beta_constant_float(target: str) -> None:
    class AddMMBetaConstantFloat(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, beta=2.0)

    check(
        AddMMBetaConstantFloat(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_beta_constant_int(target: str) -> None:
    class AddMMBetaConstantInt(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, beta=2)

    check(
        AddMMBetaConstantInt(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_broadcast_beta(target: str) -> None:
    class AddMMBroadcastBetaNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            beta: float,
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, beta=beta)

    check(
        AddMMBroadcastBetaNet(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3), 2),
        target=target
    )


def test_addmm_broadcast_beta_constant_float(target: str) -> None:
    class AddMMBroadcastBetaConstantFloat(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, beta=2.0)

    check(
        AddMMBroadcastBetaConstantFloat(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_broadcast_beta_constant_int(target: str) -> None:
    class AddMMBroadcastBetaConstantInt(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, beta=2)

    check(
        AddMMBroadcastBetaConstantInt(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3)),
        target=target
    )


def test_addmm_alpha_beta(target: str) -> None:
    class AddMMAlphaBetaNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            alpha: float,
            beta: float,
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)

    check(
        AddMMAlphaBetaNet(),
        *(torch.randn(2, 3), torch.randn(2, 3), torch.randn(3, 3), 2, 2),
        target=target
    )


def test_addmm_broadcast_alpha_beta(target: str) -> None:
    class AddMMBroadcastAlphaBetaNet(nn.Module):
        def forward(
            self,
            input: torch.Tensor,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            alpha: float,
            beta: float,
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)

    check(
        AddMMBroadcastAlphaBetaNet(),
        *(torch.randn(3), torch.randn(2, 3), torch.randn(3, 3), 2, 2),
        target=target
    )


# --- bmm ---


def test_bmm_simple(target: str) -> None:
    class BMMSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.bmm(input, mat2)

    check(
        BMMSimpleNet(), *(torch.randn(10, 3, 4), torch.randn(10, 4, 5)), target=target
    )


# --- mm ---


def test_mm_simple(target: str) -> None:
    class MMSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.mm(input, mat2)

    check(MMSimpleNet(), *(torch.randn(2, 3), torch.randn(3, 3)), target=target)
