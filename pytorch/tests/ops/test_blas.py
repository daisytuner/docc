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
        target=target,
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
        target=target,
    )


def test_addmm_broadcast_square(target: str) -> None:
    class AddMMBroadcastSquareNet(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2)

    # Square output (rows == cols == bias length) forces the bias broadcast to
    # align on the trailing (feature) axis. A leading-axis alignment would
    # silently add bias[row] instead of bias[col] without changing shapes.
    check(
        AddMMBroadcastSquareNet(),
        *(torch.randn(4), torch.randn(4, 3), torch.randn(3, 4)),
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
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
        target=target,
    )


def test_addmm_broadcast_half(target: str) -> None:
    class AddMMBroadcastHalfNet(nn.Module):
        def forward(
            self, input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            return torch.addmm(input, mat1, mat2)

    check(
        AddMMBroadcastHalfNet(),
        *(
            torch.randn(3, dtype=torch.half),
            torch.randn(2, 3, dtype=torch.half),
            torch.randn(3, 3, dtype=torch.half),
        ),
        rtol=1e-2,
        atol=1e-2,
        target=target,
    )


# --- bmm ---


def test_bmm_simple(target: str) -> None:
    class BMMSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.bmm(input, mat2)

    check(
        BMMSimpleNet(), *(torch.randn(10, 3, 4), torch.randn(10, 4, 5)), target=target
    )


def test_bmm_half(target: str) -> None:
    class BMMHalfNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.bmm(input, mat2)

    check(
        BMMHalfNet(),
        *(
            torch.randn(10, 3, 4, dtype=torch.half),
            torch.randn(10, 4, 5, dtype=torch.half),
        ),
        rtol=1e-2,
        atol=1e-2,
        target=target,
    )


def test_bmm_bf16(target: str) -> None:
    class BMMBf16Net(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.bmm(input, mat2)

    check(
        BMMBf16Net(),
        *(
            torch.randn(10, 3, 4, dtype=torch.bfloat16),
            torch.randn(10, 4, 5, dtype=torch.bfloat16),
        ),
        rtol=1e-2,
        atol=1e-2,
        target=target,
    )


def test_bmm_force_expand(target: str) -> None:
    class BMMSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.bmm(input, mat2)

    check(
        BMMSimpleNet(),
        *(torch.randn(10, 3, 4), torch.randn(10, 4, 5)),
        target=target,
        options={"library_node_expansion.force_expand": True},
    )


def test_bmm_strided(target: str) -> None:
    class BMMStridedNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            input_permute = torch.permute(input, [2, 0, 1])
            mat2_permute = torch.permute(mat2, [2, 0, 1])
            res = torch.bmm(input_permute, mat2_permute)
            return torch.permute(res, [1, 2, 0])

    check(
        BMMStridedNet(), *(torch.randn(3, 4, 10), torch.randn(4, 5, 10)), target=target
    )


# --- mm ---


def test_mm_simple(target: str) -> None:
    class MMSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.mm(input, mat2)

    check(MMSimpleNet(), *(torch.randn(2, 3), torch.randn(3, 3)), target=target)


def test_mm_half(target: str) -> None:
    class MMHalfNet(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.mm(input, mat2)

    check(
        MMHalfNet(),
        *(
            torch.randn(2, 3, dtype=torch.half),
            torch.randn(3, 3, dtype=torch.half),
        ),
        rtol=1e-2,
        atol=1e-2,
        target=target,
    )


def test_mm_bf16(target: str) -> None:
    class MMBf16Net(nn.Module):
        def forward(self, input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return torch.mm(input, mat2)

    check(
        MMBf16Net(),
        *(
            torch.randn(2, 3, dtype=torch.bfloat16),
            torch.randn(3, 3, dtype=torch.bfloat16),
        ),
        rtol=1e-2,
        atol=1e-2,
        target=target,
    )
