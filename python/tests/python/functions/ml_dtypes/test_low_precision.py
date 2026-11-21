"""Low-precision floats of ml_dtypes and NumPy: the SDFG keeps their types, FP8 (which every target computes in Float)
rounds as ml_dtypes does, matrix products are accurate to the operands' precision natively and with Ozaki slices, and a
target widening the types it does not store returns the caller's types.
"""

import json

import ml_dtypes
import numpy as np
import pytest

from docc.python import native
from docc.sdfg import PrimitiveType, StructuredSDFG

DTYPES = {
    "float16": (np.float16, 11),
    "bfloat16": (ml_dtypes.bfloat16, 8),
    "float8_e4m3fn": (ml_dtypes.float8_e4m3fn, 4),
    "float8_e5m2": (ml_dtypes.float8_e5m2, 3),
}

PRIMITIVE_TYPES = {
    "float16": PrimitiveType.Half,
    "bfloat16": PrimitiveType.BFloat,
    "float8_e4m3fn": PrimitiveType.Float8E4M3,
    "float8_e5m2": PrimitiveType.Float8E5M2,
}

FLOAT8 = {
    "float8_e4m3fn": (ml_dtypes.float8_e4m3fn, PrimitiveType.Float8E4M3),
    "float8_e5m2": (ml_dtypes.float8_e5m2, PrimitiveType.Float8E5M2),
}


def _matmul(a, b):
    return a @ b


def _operands(dtype):
    rng = np.random.default_rng(0)
    # Rows and columns of different magnitudes; the emulation scales each separately.
    a = rng.standard_normal((24, 40)) * np.logspace(-1, 1, 24)[:, None]
    b = rng.standard_normal((40, 16)) * np.logspace(1, -1, 16)[None, :]
    return a.astype(dtype), b.astype(dtype)


def _add(a, b):
    return a + b


def _subtract(a, b):
    return a - b


def _multiply(a, b):
    return a * b


def _divide(a, b):
    return a / b


class TestTypes:
    @pytest.mark.parametrize("name", DTYPES)
    def test_arguments(self, name):
        dtype, _ = DTYPES[name]
        a, b = _operands(dtype)
        sdfg = native(_add, symbolic_shapes=False).to_sdfg(a[:, :16], b[:24])
        arguments = [sdfg.type(argument).primitive_type for argument in ("a", "b")]
        assert arguments == [PRIMITIVE_TYPES[name]] * 2


class TestElementwise:
    @pytest.mark.parametrize("operation", [_add, _subtract, _multiply, _divide])
    @pytest.mark.parametrize("name", FLOAT8)
    def test_float8(self, name, operation):
        """Each operation computes in Float, as every target computes FP8, and its result rounds back at the boundary
        to nearest, ties to even, subnormals and overflow alike, as ml_dtypes rounds each operation's: E4M3 overflows to
        NaN, E5M2 to infinity."""
        dtype, _ = FLOAT8[name]
        finfo = ml_dtypes.finfo(dtype)
        values = np.array(
            [0.0, -0.0, 1.0, -1.5, 3.25, float(finfo.max), float(finfo.tiny)]
            + [float(finfo.smallest_subnormal) * k for k in (1, 2, 3)]
            + list(np.linspace(-20, 20, 22)),
            np.float32,
        )
        a = values.astype(dtype)
        b = np.roll(values, 3).astype(dtype)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = operation(a, b)
        result = native(operation)(a, b)
        assert result.dtype == dtype
        np.testing.assert_array_equal(
            result.astype(np.float32), expected.astype(np.float32)
        )


class TestMatmul:
    @pytest.mark.parametrize("precision", ["native", "ozaki1"])
    @pytest.mark.parametrize("name", DTYPES)
    def test_accuracy(self, name, precision):
        """The result's type is NumPy's (Float for ml_dtypes' types), and its error is below the operands' unit
        roundoff: without a slice count, Ozaki slices each operand as finely as its type needs, two slices for 16-bit
        floats and one for FP8, where one slice would err by about 1e-2."""
        dtype, significand = DTYPES[name]
        a, b = _operands(dtype)
        result = native(
            _matmul, symbolic_shapes=False, contraction_precision=precision
        )(a, b)
        assert result.dtype == (a @ b).dtype
        exact_a, exact_b = a.astype(np.longdouble), b.astype(np.longdouble)
        error = np.max(
            np.abs(result.astype(np.longdouble) - exact_a @ exact_b)
            / (np.abs(exact_a) @ np.abs(exact_b))
        )
        # Native float16 rounds its sums to float16, within one unit in the last place.
        bound = 2.0 ** (1 - significand) if precision == "native" else 2.0**-significand
        assert error <= bound


@pytest.fixture
def widening(monkeypatch):
    """A CPU target that does not store the low-precision floats, as a target declares in C++
    (TargetPrecision::widened_float_types): expansion widens them to Float."""
    expand = StructuredSDFG.expand

    def expand_and_widen(sdfg, *args):
        expand(sdfg, *args)
        sdfg.widen_floats(list(PRIMITIVE_TYPES.values()))

    monkeypatch.setattr(StructuredSDFG, "expand", expand_and_widen)


def _scale_into(a, b):
    b[:] = a * 2.0


def _sum_and_difference(a, b):
    return a + b, a - b


# Functions of their own: a program's build folder is named by its function and options, and the process keeps the
# library it loaded first from it.
def _product(a, b):
    return a * b


def _matrix_product(a, b):
    return a @ b


@pytest.mark.usefixtures("widening")
class TestWidening:
    """The runtime converts at the boundary: callers pass and get back their own types, computed in Float."""

    @pytest.mark.parametrize("name", PRIMITIVE_TYPES)
    def test_result(self, name):
        dtype, _ = DTYPES[name]
        a, b = _operands(dtype)
        a, b = a[:, :16], b[:24]
        result = native(_product, symbolic_shapes=False)(a, b)
        assert result.dtype == dtype
        np.testing.assert_array_equal(
            result.astype(np.float32), (a * b).astype(np.float32)
        )

    @pytest.mark.parametrize("name", PRIMITIVE_TYPES)
    def test_in_place(self, name):
        dtype, _ = DTYPES[name]
        a, _ = _operands(dtype)
        b = np.zeros_like(a)
        native(_scale_into, symbolic_shapes=False)(a, b)
        assert b.dtype == dtype
        np.testing.assert_array_equal(
            b.astype(np.float32), (a * 2.0).astype(np.float32)
        )

    def test_results(self):
        a, b = _operands(ml_dtypes.bfloat16)
        a, b = a[:, :16], b[:24]
        total, difference = native(_sum_and_difference, symbolic_shapes=False)(a, b)
        assert total.dtype == difference.dtype == ml_dtypes.bfloat16
        np.testing.assert_array_equal(total, a + b)
        np.testing.assert_array_equal(difference, a - b)

    def test_float32(self):
        """Wider results keep their type: FP8 products are float32 either way."""
        a, b = _operands(ml_dtypes.float8_e4m3fn)
        result = native(_matrix_product, symbolic_shapes=False)(a, b)
        assert result.dtype == np.float32
        np.testing.assert_allclose(
            result, a.astype(np.float32) @ b.astype(np.float32), rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize("name", FLOAT8)
    def test_device_resident(self, name, monkeypatch):
        """A device-resident program, to which PyTorch passes its tensors as they are, gets them in Float too: CuPy,
        which copies them to the device, has no FP8."""
        # Skips where torch is missing or does not load, as a build whose BLAS lacks a symbol.
        try:
            import torch
        except ImportError as error:
            pytest.skip(f"torch does not import: {error}")
        dtype, _ = FLOAT8[name]
        a, b = _operands(dtype)
        a, b = np.ascontiguousarray(a[:, :16]), b[:24]
        compiled = native(_product, symbolic_shapes=False).compile(a, b)
        received = []

        def call_device(*args):
            received.extend(arg.dtype for arg in args)
            return compiled._call_host(*(arg.numpy() for arg in args))

        monkeypatch.setattr(compiled, "device_resident", True)
        monkeypatch.setattr(compiled, "_call_device", call_device)
        torch_dtype = getattr(torch, name)
        result = compiled(
            torch.from_numpy(a.astype(np.float32)).to(torch_dtype),
            torch.from_numpy(b.astype(np.float32)).to(torch_dtype),
        )
        assert received == [torch.float32] * 2
        # Host callers get host arrays back, in the caller's type.
        assert result.dtype == dtype
        np.testing.assert_array_equal(
            result.astype(np.float32), (a * b).astype(np.float32)
        )

    def test_arguments(self):
        """The compiled program takes Float and records the caller's types."""
        a, b = _operands(ml_dtypes.bfloat16)
        compiled = native(_product, symbolic_shapes=False).compile(a[:, :16], b[:24])
        assert compiled.sdfg.type("a").primitive_type == PrimitiveType.Float
        recorded = json.loads(compiled.sdfg.metadata("widened_argument_types"))
        assert recorded == {"a": "BFloat", "b": "BFloat", "_docc_ret_0": "BFloat"}
