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


# --- zeros ---


def test_zeros_simple(target: str) -> None:
    class ZerosSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.zeros(5)

    check(ZerosSimpleNet(), *(), target=target)


def test_zeros_multi_dim(target: str) -> None:
    class ZerosMultiDimNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.zeros(2, 3)

    check(ZerosMultiDimNet(), *(), target=target)


# --- zeros_like ---


def test_zeros_like_simple(target: str) -> None:
    class ZerosLikeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(input)

    check(ZerosLikeSimpleNet(), torch.empty(2, 3), target=target)


# --- ones ---


def test_ones_simple(target: str) -> None:
    class OnesSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.ones(5)

    check(OnesSimpleNet(), *(), target=target)


def test_ones_multi_dim(target: str) -> None:
    class OnesMultiDimNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.ones(2, 3)

    check(OnesMultiDimNet(), *(), target=target)


# --- ones_like ---


def test_ones_like_simple(target: str) -> None:
    class OnesLikeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(input)

    check(OnesLikeSimpleNet(), torch.empty(2, 3), target=target)


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


# --- linspace ---


def test_linspace_simple(target: str) -> None:
    class LinspaceSimpleNet(nn.Module):
        def forward(self):
            return torch.linspace(3, 10, steps=5)

    check(LinspaceSimpleNet(), *(), target=target)


def test_linspace_neg_start(target: str) -> None:
    class LinspaceNegStartNet(nn.Module):
        def forward(self):
            return torch.linspace(-10, 10, steps=5)

    check(LinspaceNegStartNet(), *(), target=target)


def test_linspace_steps_one(target: str) -> None:
    class LinspaceStepsOneNet(nn.Module):
        def forward(self):
            return torch.linspace(3, 10, steps=1)

    check(LinspaceStepsOneNet(), *(), target=target)


# --- logspace ---


def test_logspace_simple(target: str) -> None:
    class LogspaceSimpleNet(nn.Module):
        def forward(self):
            return torch.logspace(-10, 10, steps=5)

    check(LogspaceSimpleNet(), *(), target=target)


def test_logspace_float(target: str) -> None:
    class LogspaceFloatNet(nn.Module):
        def forward(self):
            return torch.logspace(0.1, 1.0, steps=5)

    check(LogspaceFloatNet(), *(), target=target)


def test_logspace_float_step_one(target: str) -> None:
    class LogspaceFloatStepOneNet(nn.Module):
        def forward(self):
            return torch.logspace(0.1, 1.0, steps=1)

    check(LogspaceFloatStepOneNet(), *(), target=target)


def test_logspace_base_two(target: str) -> None:
    class LogspaceBaseTwoNet(nn.Module):
        def forward(self):
            return torch.logspace(2, 2, steps=1, base=2)

    check(LogspaceBaseTwoNet(), *(), target=target)


# --- eye ---


def test_eye_simple(target: str) -> None:
    class EyeSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.eye(3)

    check(EyeSimpleNet(), *(), target=target)


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


# --- empty_like ---


def test_empty_like_simple(target: str) -> None:
    class EmptyLikeSimpleNet(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(input).fill_(0)

    check(EmptyLikeSimpleNet(), torch.empty((2, 3), dtype=torch.int32), target=target)


@pytest.mark.supported_targets("cuda", "rocm")
def test_empty_like_gpu(target: str) -> None:
    class EmptyLikeGPUNet(nn.Module):
        is_docc = None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.empty_like(
                input,
                device=(
                    torch.device("cuda") if self.is_docc else torch.get_default_device()
                ),
            ).fill_(0)

    check(EmptyLikeGPUNet(), torch.empty((2, 3), dtype=torch.int32), target=target)


# --- empty_permuted ---


def test_empty_permuted_simple(target: str) -> None:
    class EmptyPermutedSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.empty_permuted((2, 3, 5, 7), (0, 1, 2, 3)).fill_(0.0)

    check(EmptyPermutedSimpleNet(), *(), target=target)


def test_empty_permuted_permuted(target: str) -> None:
    class EmptyPermutedPermutedNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.empty_permuted((2, 3, 5, 7), (0, 2, 3, 1)).fill_(0.0)

    check(EmptyPermutedPermutedNet(), *(), target=target)


# --- empty_strided ---


def test_empty_strided_simple(target: str) -> None:
    class EmptyStridedSimpleNet(nn.Module):
        def forward(self) -> torch.Tensor:
            return torch.empty_strided((2, 3), (1, 2)).fill_(0.0)

    check(EmptyStridedSimpleNet(), *(), target=target)


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
