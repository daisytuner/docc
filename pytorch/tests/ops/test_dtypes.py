import torch
import torch.nn as nn

from tests import check

# --- dtype == torch.float32 ---


def test_float32(target: str) -> None:
    class DataTypesFloat32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float32)
    y = torch.randn(2, 4, dtype=torch.float32)
    check(DataTypesFloat32Net().eval(), *(x, y), target=target)


# --- dtype == torch.float ---


def test_float(target: str) -> None:
    class DataTypesFloatNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float)
    y = torch.randn(2, 4, dtype=torch.float)
    check(DataTypesFloatNet().eval(), *(x, y), target=target)


# --- dtype == torch.float64 ---


def test_float64(target: str) -> None:
    class DataTypesFloat64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float64)
    y = torch.randn(2, 4, dtype=torch.float64)
    check(DataTypesFloat64Net().eval(), *(x, y), target=target)


# --- dtype == torch.double ---


def test_double(target: str) -> None:
    class DataTypesDoubleNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.double)
    y = torch.randn(2, 4, dtype=torch.double)
    check(DataTypesDoubleNet().eval(), *(x, y), target=target)


# --- dtype == torch.float16 ---


def test_float16(target: str) -> None:
    class DataTypesFloat16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.float16)
    y = torch.randn(2, 4, dtype=torch.float16)
    check(DataTypesFloat16Net().eval(), *(x, y), rtol=1e-2, atol=1e-2, target=target)


# --- dtype == torch.half ---


def test_half(target: str) -> None:
    class DataTypesHalfNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.half)
    y = torch.randn(2, 4, dtype=torch.half)
    check(DataTypesHalfNet().eval(), *(x, y), rtol=1e-2, atol=1e-2, target=target)


# --- dtype == torch.bfloat16 ---


def test_bfloat16(target: str) -> None:
    class DataTypesBFloat16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randn(2, 4, dtype=torch.bfloat16)
    y = torch.randn(2, 4, dtype=torch.bfloat16)
    check(DataTypesBFloat16Net().eval(), *(x, y), rtol=1e-2, atol=1e-2, target=target)


# --- dtype == torch.uint8 ---


def test_uint8(target: str) -> None:
    class DataTypesUInt8Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint8)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint8)
    check(DataTypesUInt8Net().eval(), *(x, y), target=target)


# --- dtype == torch.int8 ---


def test_int8(target: str) -> None:
    class DataTypesInt8Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int8)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int8)
    check(DataTypesInt8Net().eval(), *(x, y), target=target)


# --- dtype == torch.uint16 ---


def test_uint16(target: str) -> None:
    class DataTypesUInt16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint16)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint16)
    check(DataTypesUInt16Net().eval(), *(x, y), target=target)


# --- dtype == torch.int16 ---


def test_int16(target: str) -> None:
    class DataTypesInt16Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int16)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int16)
    check(DataTypesInt16Net().eval(), *(x, y), target=target)


# --- dtype == torch.short ---


def test_short(target: str) -> None:
    class DataTypesShortNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.short)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.short)
    check(DataTypesShortNet().eval(), *(x, y), target=target)


# --- dtype == torch.uint32 ---


def test_uint32(target: str) -> None:
    class DataTypesUInt32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint32)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint32)
    check(DataTypesUInt32Net().eval(), *(x, y), target=target)


# --- dtype == torch.int32 ---


def test_int32(target: str) -> None:
    class DataTypesInt32Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int32)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int32)
    check(DataTypesInt32Net().eval(), *(x, y), target=target)


# --- dtype == torch.int ---


def test_int(target: str) -> None:
    class DataTypesIntNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int)
    check(DataTypesIntNet().eval(), *(x, y), target=target)


# --- dtype == torch.uint64 ---


def test_uint64(target: str) -> None:
    class DataTypesUInt64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(0, 100, (2, 4), dtype=torch.uint64)
    y = torch.randint(0, 100, (2, 4), dtype=torch.uint64)
    check(DataTypesUInt64Net().eval(), *(x, y), target=target)


# --- dtype == torch.int64 ---


def test_int64(target: str) -> None:
    class DataTypesInt64Net(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.int64)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.int64)
    check(DataTypesInt64Net().eval(), *(x, y), target=target)


# --- dtype == torch.long ---


def test_long(target: str) -> None:
    class DataTypesLongNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.add(x, y)

    x = torch.randint(-100, 100, (2, 4), dtype=torch.long)
    y = torch.randint(-100, 100, (2, 4), dtype=torch.long)
    check(DataTypesLongNet().eval(), *(x, y), target=target)


# --- dtype == torch.bool ---


def test_bool(target: str) -> None:
    class DataTypesBoolNet(nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return torch.bitwise_and(x, y)

    x = torch.randint(0, 1, (2, 4), dtype=torch.bool)
    y = torch.randint(0, 1, (2, 4), dtype=torch.bool)
    check(DataTypesBoolNet().eval(), *(x, y), target=target)
