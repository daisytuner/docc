import torch
import torch.nn as nn

import pytest

from tests import check

# --- tensor ---


def test_tensor_simple(target: str) -> None:
    class TensorSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.tensor([[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]])

    check(TensorSimpleNet(), *(), target=target)


def test_tensor_type_inference(target: str) -> None:
    class TensorTypeInferenceNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.tensor([0, 1])

    check(TensorTypeInferenceNet(), *(), target=target)


def test_tensor_dtype(target: str) -> None:
    class TensorDtypeNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.tensor([[0.11111, 0.222222, 0.3333333]], dtype=torch.float64)

    check(TensorDtypeNet(), *(), target=target)


def test_tensor_zero_dim(target: str) -> None:
    class TensorZeroDimNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.tensor(3.14159)

    check(TensorZeroDimNet(), *(), target=target)


def test_tensor_empty(target: str) -> None:
    class TensorEmptyNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.tensor([])

    check(TensorEmptyNet(), *(), target=target)


# Currently disable because of unwanted graph breaks and incompatible CUDA/ROCm versions
# @pytest.mark.supported_targets("cuda", "rocm")
# def test_tensor_gpu(target: str) -> None:
#     class TensorSimpleNet(nn.Module):
#         is_docc = None

#         def forward(self) -> torch.Tensor:
#             return torch.tensor(
#                 [[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]],
#                 device=(
#                     torch.device("cuda") if self.is_docc else torch.get_default_device()
#                 ),
#             )

#     check(TensorSimpleNet(), *(), target=target)


# --- arange ---


def test_arange_simple(target: str) -> None:
    class ArangeSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(5)

    check(ArangeSimpleNet(), *(), target=target)


def test_arange_start(target: str) -> None:
    class ArangeStartNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(4, 10)

    check(ArangeStartNet(), *(), target=target)


def test_arange_start_step(target: str) -> None:
    class ArangeStartStepNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(1, 5, 2)

    check(ArangeStartStepNet(), *(), target=target)


def test_arange_dtype(target: str) -> None:
    class ArangeDtypeNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(1, 6, 2, dtype=torch.float64)

    check(ArangeDtypeNet(), *(), target=target)


def test_arange_float(target: str) -> None:
    class ArangeFloatNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(0.0, 1.5, 0.5)

    check(ArangeFloatNet(), *(), target=target)


def test_arange_default_dtype(target: str) -> None:
    class ArangeDefaultDtypeNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(8, dtype=torch.float32)

    check(ArangeDefaultDtypeNet(), *(), target=target)


def test_arange_default_pin_memory(target: str) -> None:
    class ArangeDefaultPinMemoryNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.arange(8, pin_memory=False)

    check(ArangeDefaultPinMemoryNet(), *(), target=target)


def test_arange_symbolic(target: str) -> None:
    class ArangeSymbolicNet(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.arange(3, x.shape[0], 2)

    check(ArangeSymbolicNet(), torch.ones(5), target=target)


@pytest.mark.supported_targets("cuda", "rocm")
def test_arange_gpu(target: str) -> None:
    class ArangeGPUNet(nn.Module):
        is_docc = None

        def forward(self) -> torch.Tensor:
            return torch.arange(
                5,
                device=(
                    torch.device("cuda") if self.is_docc else torch.get_default_device()
                ),
            )

    check(ArangeGPUNet(), *(), target=target)


# --- empty ---


def test_empty_simple(target: str) -> None:
    class EmptySimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.empty((2, 3)).fill_(0.0)

    check(EmptySimpleNet(), *(), target=target)


def test_empty_dtype(target: str) -> None:
    class EmptyDtypeNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.empty((2, 3), dtype=torch.int64).fill_(0)

    check(EmptyDtypeNet(), *(), target=target)


@pytest.mark.supported_targets("cuda", "rocm")
def test_empty_gpu(target: str) -> None:
    class EmptyGPUNet(nn.Module):
        is_docc = None

        def forward(self) -> torch.Tensor:
            return torch.empty(
                (2, 3),
                device=(
                    torch.device("cuda") if self.is_docc else torch.get_default_device()
                ),
            ).fill_(0.0)

    check(EmptyGPUNet(), *(), target=target)


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


def test_full_bools(target: str) -> None:
    class FullBoolsNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.full((2,), True)

    check(FullBoolsNet(), *(), target=target)


@pytest.mark.supported_targets("cuda", "rocm")
def test_full_gpu(target: str) -> None:
    class FullGPUNet(nn.Module):
        is_docc = None

        def forward(self) -> torch.Tensor:
            return torch.full(
                (2, 3),
                3.141592,
                device=(
                    torch.device("cuda") if self.is_docc else torch.get_default_device()
                ),
            )

    check(FullGPUNet(), *(), target=target)


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


@pytest.mark.supported_targets("cuda", "rocm")
def test_full_like_gpu(target: str) -> None:
    class FullLikeGPUNet(nn.Module):
        is_docc = None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(
                input,
                3.141592,
                device=(
                    torch.device("cuda") if self.is_docc else torch.get_default_device()
                ),
            )

    check(FullLikeGPUNet(), torch.ones(2, 3), target=target)


# --- scalar_tensor ---


def test_scalar_tensor_simple(target: str) -> None:
    class ScalarTensorSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.scalar_tensor(3.141592)

    check(ScalarTensorSimpleNet(), target=target)


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


@pytest.mark.supported_targets("cuda", "rocm")
def test_scalar_tensor_gpu(target: str) -> None:
    class ScalarTensorGPUNet(nn.Module):
        is_docc = None

        def forward(self) -> torch.Tensor:
            return torch.scalar_tensor(
                3.141592,
                device=(
                    torch.device("cuda") if self.is_docc else torch.get_default_device()
                ),
            )

    check(ScalarTensorGPUNet(), target=target)
