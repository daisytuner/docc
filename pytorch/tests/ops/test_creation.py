import torch
import torch.nn as nn

from tests import check

# --- full ---


def test_full_simple(target: str) -> None:
    class FullSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.full((2, 3), 3.141592)

    check(FullSimpleNet(), *(), target=target)


def test_full_dtype(target: str) -> None:
    class FullDtypeNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.full((2, 3), 3.141592, dtype=torch.float64)

    check(FullDtypeNet(), *(), target=target)


# --- full_like ---


def test_full_like_simple(target: str) -> None:
    class FullLikeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(input, 3.141592)

    check(FullLikeSimpleNet(), torch.ones(2, 3), target=target)


def test_full_like_dtype(target: str) -> None:
    class FullLikeDtypeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(input, 3.141592)

    check(FullLikeDtypeNet(), torch.ones((2, 3), dtype=torch.float64), target=target)


def test_full_like_dtype_change(target: str) -> None:
    class FullLikeDtypeChangeNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(input, 3.141592, dtype=torch.float32)

    check(
        FullLikeDtypeChangeNet(), torch.ones((2, 3), dtype=torch.float64), target=target
    )


# --- scalar_tensor ---


def test_scalar_tensor(target: str) -> None:
    class ScalarTensorNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.scalar_tensor(3.141592)

    check(ScalarTensorNet(), target=target)


def test_scalar_tensor_dtype(target: str) -> None:
    class ScalarTensorDtypeNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.scalar_tensor(3.141592, dtype=torch.float64)

    check(ScalarTensorDtypeNet(), target=target)


def test_scalar_tensor_device(target: str) -> None:
    class ScalarTensorDeviceNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.scalar_tensor(3.141592, device=torch.device("cpu"))

    check(ScalarTensorDeviceNet(), target=target)


def test_scalar_tensor_layout(target: str) -> None:
    class ScalarTensorLayoutNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.scalar_tensor(3.141592, layout=torch.strided)

    check(ScalarTensorLayoutNet(), target=target)


def test_scalar_tensor_add(target: str) -> None:
    class ScalarTensorAddNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return input + torch.scalar_tensor(3.141592)

    check(ScalarTensorAddNet(), torch.ones(2, 3), target=target)


def test_scalar_tensor_pin_memory(target: str) -> None:
    class ScalarTensorPinMemoryNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.scalar_tensor(3.141592, pin_memory=False)

    check(ScalarTensorPinMemoryNet(), target=target)
