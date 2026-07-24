import torch
import torch.nn as nn
import pytest

from integration.torch.check import check_backend, check_compile

# --- dtype == torch.float32 ---


def test_float32_compile() -> None:
    class DataTypesFloat32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float32)
    y = torch.randn(2, 4, dtype=torch.float32)
    check_compile(DataTypesFloat32Net().eval(), *(x, y))


def test_float32_backend() -> None:
    class DataTypesFloat32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float32)
    y = torch.randn(2, 4, dtype=torch.float32)
    check_backend(DataTypesFloat32Net().eval(), *(x, y))


# --- dtype == torch.float ---


def test_float_compile() -> None:
    class DataTypesFloatNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float)
    y = torch.randn(2, 4, dtype=torch.float)
    check_compile(DataTypesFloatNet().eval(), *(x, y))


def test_float_backend() -> None:
    class DataTypesFloatNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float)
    y = torch.randn(2, 4, dtype=torch.float)
    check_backend(DataTypesFloatNet().eval(), *(x, y))


# --- dtype == torch.float64 ---


def test_float64_compile() -> None:
    class DataTypesFloat64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float64)
    y = torch.randn(2, 4, dtype=torch.float64)
    check_compile(DataTypesFloat64Net().eval(), *(x, y))


def test_float64_backend() -> None:
    class DataTypesFloat64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float64)
    y = torch.randn(2, 4, dtype=torch.float64)
    check_backend(DataTypesFloat64Net().eval(), *(x, y))


# --- dtype == torch.double ---


def test_double_compile() -> None:
    class DataTypesDoubleNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.double)
    y = torch.randn(2, 4, dtype=torch.double)
    check_compile(DataTypesDoubleNet().eval(), *(x, y))


def test_double_backend() -> None:
    class DataTypesDoubleNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.double)
    y = torch.randn(2, 4, dtype=torch.double)
    check_backend(DataTypesDoubleNet().eval(), *(x, y))


# --- dtype == torch.float16 ---


def test_float16_compile() -> None:
    class DataTypesFloat16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float16)
    y = torch.randn(2, 4, dtype=torch.float16)
    check_compile(DataTypesFloat16Net().eval(), *(x, y))


def test_float16_backend() -> None:
    class DataTypesFloat16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float16)
    y = torch.randn(2, 4, dtype=torch.float16)
    check_backend(DataTypesFloat16Net().eval(), *(x, y))


# --- dtype == torch.half ---


def test_half_compile() -> None:
    class DataTypesHalfNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.half)
    y = torch.randn(2, 4, dtype=torch.half)
    check_compile(DataTypesHalfNet().eval(), *(x, y))


def test_half_backend() -> None:
    class DataTypesHalfNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.half)
    y = torch.randn(2, 4, dtype=torch.half)
    check_backend(DataTypesHalfNet().eval(), *(x, y))


# --- dtype == torch.bfloat16 ---


def test_bfloat16_compile() -> None:
    class DataTypesBFloat16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = torch.randn(2, 4, dtype=torch.bfloat16)
    check_compile(DataTypesBFloat16Net().eval(), *(x, y))


def test_bfloat16_backend() -> None:
    class DataTypesBFloat16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = torch.randn(2, 4, dtype=torch.bfloat16)
    check_backend(DataTypesBFloat16Net().eval(), *(x, y))


# --- dtype == torch.uint8 ---


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint8_compile() -> None:
    class DataTypesUInt8Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint8)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint8)
    check_compile(DataTypesUInt8Net().eval(), *(x, y))


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint8_backend() -> None:
    class DataTypesUInt8Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint8)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint8)
    check_backend(DataTypesUInt8Net().eval(), *(x, y))


# --- dtype == torch.int8 ---


def test_int8_compile() -> None:
    class DataTypesInt8Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int8)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int8)
    check_compile(DataTypesInt8Net().eval(), *(x, y))


def test_int8_backend() -> None:
    class DataTypesInt8Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int8)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int8)
    check_backend(DataTypesInt8Net().eval(), *(x, y))


# --- dtype == torch.uint16 ---


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint16_compile() -> None:
    class DataTypesUInt16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint16)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint16)
    check_compile(DataTypesUInt16Net().eval(), *(x, y))


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint16_backend() -> None:
    class DataTypesUInt16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint16)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint16)
    check_backend(DataTypesUInt16Net().eval(), *(x, y))


# --- dtype == torch.int16 ---


def test_int16_compile() -> None:
    class DataTypesInt16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int16)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int16)
    check_compile(DataTypesInt16Net().eval(), *(x, y))


def test_int16_backend() -> None:
    class DataTypesInt16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int16)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int16)
    check_backend(DataTypesInt16Net().eval(), *(x, y))


# --- dtype == torch.short ---


def test_short_compile() -> None:
    class DataTypesShortNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.short)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.short)
    check_compile(DataTypesShortNet().eval(), *(x, y))


def test_short_backend() -> None:
    class DataTypesShortNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.short)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.short)
    check_backend(DataTypesShortNet().eval(), *(x, y))


# --- dtype == torch.uint32 ---


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint32_compile() -> None:
    class DataTypesUInt32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint32)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint32)
    check_compile(DataTypesUInt32Net().eval(), *(x, y))


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint32_backend() -> None:
    class DataTypesUInt32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint32)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint32)
    check_backend(DataTypesUInt32Net().eval(), *(x, y))


# --- dtype == torch.int32 ---


def test_int32_compile() -> None:
    class DataTypesInt32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int32)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int32)
    check_compile(DataTypesInt32Net().eval(), *(x, y))


def test_int32_backend() -> None:
    class DataTypesInt32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int32)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int32)
    check_backend(DataTypesInt32Net().eval(), *(x, y))


# --- dtype == torch.int ---


def test_int_compile() -> None:
    class DataTypesIntNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int)
    check_compile(DataTypesIntNet().eval(), *(x, y))


def test_int_backend() -> None:
    class DataTypesIntNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int)
    check_backend(DataTypesIntNet().eval(), *(x, y))


# --- dtype == torch.uint64 ---


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint64_compile() -> None:
    class DataTypesUInt64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint64)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint64)
    check_compile(DataTypesUInt64Net().eval(), *(x, y))


@pytest.mark.skip(reason="Unsupported because of torch_mlir")
def test_uint64_backend() -> None:
    class DataTypesUInt64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint64)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint64)
    check_backend(DataTypesUInt64Net().eval(), *(x, y))


# --- dtype == torch.int64 ---


def test_int64_compile() -> None:
    class DataTypesInt64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int64)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int64)
    check_compile(DataTypesInt64Net().eval(), *(x, y))


def test_int64_backend() -> None:
    class DataTypesInt64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int64)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int64)
    check_backend(DataTypesInt64Net().eval(), *(x, y))


# --- dtype == torch.long ---


def test_long_compile() -> None:
    class DataTypesLongNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.long)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.long)
    check_compile(DataTypesLongNet().eval(), *(x, y))


def test_long_backend() -> None:
    class DataTypesLongNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.long)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.long)
    check_backend(DataTypesLongNet().eval(), *(x, y))


# --- dtype == torch.bool ---


def test_bool_compile() -> None:
    class DataTypesBoolNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.bitwise_and(x, y)

    x = torch.randint(0, 1, (2, 4), dtype=torch.bool)
    y = torch.randint(0, 1, (2, 4), dtype=torch.bool)
    check_compile(DataTypesBoolNet().eval(), *(x, y))


def test_bool_backend() -> None:
    class DataTypesBoolNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.bitwise_and(x, y)

    x = torch.randint(0, 1, (2, 4), dtype=torch.bool)
    y = torch.randint(0, 1, (2, 4), dtype=torch.bool)
    check_backend(DataTypesBoolNet().eval(), *(x, y))
