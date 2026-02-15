import pytest
from docc.python import native
import math
import numpy as np

# =============================================================================
# Trigonometric Functions
# =============================================================================


def test_highway_sin():
    @native(target="sequential")
    def highway_sin(A, B):
        for i in range(A.shape[0]):
            B[i] = math.sin(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)

    highway_sin(A, B)
    assert np.allclose(B, np.sin(A))

    sdfg = highway_sin.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_cos():
    @native(target="sequential")
    def highway_cos(A, B):
        for i in range(A.shape[0]):
            B[i] = math.cos(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)

    highway_cos(A, B)
    assert np.allclose(B, np.cos(A))

    sdfg = highway_cos.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


@pytest.mark.skip(reason="No HIGHWAY implementation for tan yet")
def test_highway_tan():
    @native(target="sequential")
    def highway_tan(A, B):
        for i in range(A.shape[0]):
            B[i] = math.tan(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 1.5 - 0.75  # Avoid tan singularities
    B = np.random.rand(N).astype(np.float64)

    highway_tan(A, B)
    assert np.allclose(B, np.tan(A))

    sdfg = highway_tan.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_asin():
    @native(target="sequential")
    def highway_asin(A, B):
        for i in range(A.shape[0]):
            B[i] = math.asin(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 2 - 1  # Range [-1, 1]
    B = np.random.rand(N).astype(np.float64)

    highway_asin(A, B)
    assert np.allclose(B, np.arcsin(A))

    sdfg = highway_asin.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_acos():
    @native(target="sequential")
    def highway_acos(A, B):
        for i in range(A.shape[0]):
            B[i] = math.acos(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 2 - 1  # Range [-1, 1]
    B = np.random.rand(N).astype(np.float64)

    highway_acos(A, B)
    assert np.allclose(B, np.arccos(A))

    sdfg = highway_acos.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_atan():
    @native(target="sequential")
    def highway_atan(A, B):
        for i in range(A.shape[0]):
            B[i] = math.atan(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10 - 5
    B = np.random.rand(N).astype(np.float64)

    highway_atan(A, B)
    assert np.allclose(B, np.arctan(A))

    sdfg = highway_atan.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_atan2():
    @native(target="sequential")
    def highway_atan2(A, B, C):
        for i in range(A.shape[0]):
            C[i] = math.atan2(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 2 - 1
    B = np.random.rand(N).astype(np.float64) * 2 - 1
    C = np.random.rand(N).astype(np.float64)

    highway_atan2(A, B, C)
    assert np.allclose(C, np.arctan2(A, B))

    sdfg = highway_atan2.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


# =============================================================================
# Hyperbolic Functions
# =============================================================================


def test_highway_sinh():
    @native(target="sequential")
    def highway_sinh(A, B):
        for i in range(A.shape[0]):
            B[i] = math.sinh(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 4 - 2  # Reasonable range
    B = np.random.rand(N).astype(np.float64)

    highway_sinh(A, B)
    assert np.allclose(B, np.sinh(A))

    sdfg = highway_sinh.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


@pytest.mark.skip(reason="No HIGHWAY implementation for cosh yet")
def test_highway_cosh():
    @native(target="sequential")
    def highway_cosh(A, B):
        for i in range(A.shape[0]):
            B[i] = math.cosh(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 4 - 2
    B = np.random.rand(N).astype(np.float64)

    highway_cosh(A, B)
    assert np.allclose(B, np.cosh(A))

    sdfg = highway_cosh.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_tanh():
    @native(target="sequential")
    def highway_tanh(A, B):
        for i in range(A.shape[0]):
            B[i] = math.tanh(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 4 - 2
    B = np.random.rand(N).astype(np.float64)

    highway_tanh(A, B)
    assert np.allclose(B, np.tanh(A))

    sdfg = highway_tanh.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_asinh():
    @native(target="sequential")
    def highway_asinh(A, B):
        for i in range(A.shape[0]):
            B[i] = math.asinh(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10 - 5
    B = np.random.rand(N).astype(np.float64)

    highway_asinh(A, B)
    assert np.allclose(B, np.arcsinh(A))

    sdfg = highway_asinh.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_acosh():
    @native(target="sequential")
    def highway_acosh(A, B):
        for i in range(A.shape[0]):
            B[i] = math.acosh(A[i])

    N = 128
    A = (
        np.random.rand(N).astype(np.float64) * 5 + 1
    )  # Range [1, 6] (acosh requires >= 1)
    B = np.random.rand(N).astype(np.float64)

    highway_acosh(A, B)
    assert np.allclose(B, np.arccosh(A))

    sdfg = highway_acosh.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_atanh():
    @native(target="sequential")
    def highway_atanh(A, B):
        for i in range(A.shape[0]):
            B[i] = math.atanh(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 1.8 - 0.9  # Range (-1, 1)
    B = np.random.rand(N).astype(np.float64)

    highway_atanh(A, B)
    assert np.allclose(B, np.arctanh(A))

    sdfg = highway_atanh.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


# =============================================================================
# Exponential and Logarithmic Functions
# =============================================================================


def test_highway_exp():
    @native(target="sequential")
    def highway_exp(A, B):
        for i in range(A.shape[0]):
            B[i] = math.exp(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 4 - 2  # Avoid overflow
    B = np.random.rand(N).astype(np.float64)

    highway_exp(A, B)
    assert np.allclose(B, np.exp(A))

    sdfg = highway_exp.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


@pytest.mark.skip(reason="No HIGHWAY implementation for exp2 yet")
def test_highway_exp2():
    @native(target="sequential")
    def highway_exp2(A, B):
        for i in range(A.shape[0]):
            B[i] = math.exp2(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 8 - 4  # Reasonable range
    B = np.random.rand(N).astype(np.float64)

    highway_exp2(A, B)
    assert np.allclose(B, np.exp2(A))

    sdfg = highway_exp2.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_expm1():
    @native(target="sequential")
    def highway_expm1(A, B):
        for i in range(A.shape[0]):
            B[i] = math.expm1(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 4 - 2
    B = np.random.rand(N).astype(np.float64)

    highway_expm1(A, B)
    assert np.allclose(B, np.expm1(A))

    sdfg = highway_expm1.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_log():
    @native(target="sequential")
    def highway_log(A, B):
        for i in range(A.shape[0]):
            B[i] = math.log(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10 + 0.1  # Positive values
    B = np.random.rand(N).astype(np.float64)

    highway_log(A, B)
    assert np.allclose(B, np.log(A))

    sdfg = highway_log.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_log2():
    @native(target="sequential")
    def highway_log2(A, B):
        for i in range(A.shape[0]):
            B[i] = math.log2(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10 + 0.1
    B = np.random.rand(N).astype(np.float64)

    highway_log2(A, B)
    assert np.allclose(B, np.log2(A))

    sdfg = highway_log2.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_log10():
    @native(target="sequential")
    def highway_log10(A, B):
        for i in range(A.shape[0]):
            B[i] = math.log10(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10 + 0.1
    B = np.random.rand(N).astype(np.float64)

    highway_log10(A, B)
    assert np.allclose(B, np.log10(A))

    sdfg = highway_log10.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_log1p():
    @native(target="sequential")
    def highway_log1p(A, B):
        for i in range(A.shape[0]):
            B[i] = math.log1p(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10  # x > -1
    B = np.random.rand(N).astype(np.float64)

    highway_log1p(A, B)
    assert np.allclose(B, np.log1p(A))

    sdfg = highway_log1p.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


# =============================================================================
# Power Functions
# =============================================================================


def test_highway_pow():
    @native(target="sequential")
    def highway_pow(A, B, C):
        for i in range(A.shape[0]):
            C[i] = math.pow(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 5 + 0.1  # Positive base
    B = np.random.rand(N).astype(np.float64) * 3  # Reasonable exponent
    C = np.random.rand(N).astype(np.float64)

    highway_pow(A, B, C)
    assert np.allclose(C, np.power(A, B))

    sdfg = highway_pow.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_sqrt():
    @native(target="sequential")
    def highway_sqrt(A, B):
        for i in range(A.shape[0]):
            B[i] = math.sqrt(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10  # Non-negative
    B = np.random.rand(N).astype(np.float64)

    highway_sqrt(A, B)
    assert np.allclose(B, np.sqrt(A))

    sdfg = highway_sqrt.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_cbrt():
    @native(target="sequential")
    def highway_cbrt(A, B):
        for i in range(A.shape[0]):
            B[i] = math.cbrt(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10 + 0.1  # Positive for log-based impl
    B = np.random.rand(N).astype(np.float64)

    highway_cbrt(A, B)
    assert np.allclose(B, np.cbrt(A))

    sdfg = highway_cbrt.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_hypot():
    @native(target="sequential")
    def highway_hypot(A, B, C):
        for i in range(A.shape[0]):
            C[i] = math.hypot(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10 - 5
    B = np.random.rand(N).astype(np.float64) * 10 - 5
    C = np.random.rand(N).astype(np.float64)

    highway_hypot(A, B, C)
    assert np.allclose(C, np.hypot(A, B))

    sdfg = highway_hypot.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


# =============================================================================
# Rounding and Remainder Functions
# =============================================================================


def test_highway_fabs():
    @native(target="sequential")
    def highway_fabs(A, B):
        for i in range(A.shape[0]):
            B[i] = math.fabs(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64)

    highway_fabs(A, B)
    assert np.allclose(B, np.fabs(A))

    sdfg = highway_fabs.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_ceil():
    @native(target="sequential")
    def highway_ceil(A, B):
        for i in range(A.shape[0]):
            B[i] = math.ceil(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64)

    highway_ceil(A, B)
    assert np.allclose(B, np.ceil(A))

    sdfg = highway_ceil.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_floor():
    @native(target="sequential")
    def highway_floor(A, B):
        for i in range(A.shape[0]):
            B[i] = math.floor(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64)

    highway_floor(A, B)
    assert np.allclose(B, np.floor(A))

    sdfg = highway_floor.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_trunc():
    @native(target="sequential")
    def highway_trunc(A, B):
        for i in range(A.shape[0]):
            B[i] = math.trunc(A[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64)

    highway_trunc(A, B)
    assert np.allclose(B, np.trunc(A))

    sdfg = highway_trunc.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_fmod():
    @native(target="sequential")
    def highway_fmod(A, B, C):
        for i in range(A.shape[0]):
            C[i] = math.fmod(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64) * 4 + 0.5  # Avoid division by zero
    C = np.random.rand(N).astype(np.float64)

    highway_fmod(A, B, C)
    assert np.allclose(C, np.fmod(A, B))

    sdfg = highway_fmod.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_mod():
    @native(target="sequential")
    def highway_mod(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] % B[i]

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64) * 4 + 0.5  # Avoid division by zero
    C = np.random.rand(N).astype(np.float64)

    highway_mod(A, B, C)
    # Python's % has floored modulo semantics (result has same sign as divisor)
    assert np.allclose(C, A % B)

    sdfg = highway_mod.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_remainder():
    @native(target="sequential")
    def highway_remainder(A, B, C):
        for i in range(A.shape[0]):
            C[i] = math.remainder(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64) * 4 + 0.5
    C = np.random.rand(N).astype(np.float64)

    highway_remainder(A, B, C)
    # math.remainder uses IEEE 754 semantics: x - round(x/y) * y
    expected = np.array([math.remainder(a, b) for a, b in zip(A, B)])
    assert np.allclose(C, expected)

    sdfg = highway_remainder.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


# =============================================================================
# Minimum and Maximum Functions
# =============================================================================


def test_highway_fmax():
    @native(target="sequential")
    def highway_fmax(A, B, C):
        for i in range(A.shape[0]):
            C[i] = max(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64) * 20 - 10
    C = np.random.rand(N).astype(np.float64)

    highway_fmax(A, B, C)
    assert np.allclose(C, np.fmax(A, B))

    sdfg = highway_fmax.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_fmin():
    @native(target="sequential")
    def highway_fmin(A, B, C):
        for i in range(A.shape[0]):
            C[i] = min(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64) * 20 - 10
    C = np.random.rand(N).astype(np.float64)

    highway_fmin(A, B, C)
    assert np.allclose(C, np.fmin(A, B))

    sdfg = highway_fmin.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


# =============================================================================
# Floating-point Manipulation Functions
# =============================================================================


def test_highway_copysign():
    @native(target="sequential")
    def highway_copysign(A, B, C):
        for i in range(A.shape[0]):
            C[i] = math.copysign(A[i], B[i])

    N = 128
    A = np.random.rand(N).astype(np.float64) * 10
    B = np.random.rand(N).astype(np.float64) * 20 - 10  # Mixed signs
    C = np.random.rand(N).astype(np.float64)

    highway_copysign(A, B, C)
    assert np.allclose(C, np.copysign(A, B))

    sdfg = highway_copysign.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


# =============================================================================
# Arithmetic Operations (Tasklets)
# =============================================================================


def test_highway_add():
    @native(target="sequential")
    def highway_add(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] + B[i]

    N = 128
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.random.rand(N).astype(np.float64)

    highway_add(A, B, C)
    assert np.allclose(C, A + B)

    sdfg = highway_add.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_sub():
    @native(target="sequential")
    def highway_sub(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] - B[i]

    N = 128
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.random.rand(N).astype(np.float64)

    highway_sub(A, B, C)
    assert np.allclose(C, A - B)

    sdfg = highway_sub.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_mul():
    @native(target="sequential")
    def highway_mul(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] * B[i]

    N = 128
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64)
    C = np.random.rand(N).astype(np.float64)

    highway_mul(A, B, C)
    assert np.allclose(C, A * B)

    sdfg = highway_mul.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_div():
    @native(target="sequential")
    def highway_div(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] / B[i]

    N = 128
    A = np.random.rand(N).astype(np.float64)
    B = np.random.rand(N).astype(np.float64) + 0.1  # Avoid division by zero
    C = np.random.rand(N).astype(np.float64)

    highway_div(A, B, C)
    assert np.allclose(C, A / B)

    sdfg = highway_div.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1


def test_highway_neg():
    @native(target="sequential")
    def highway_neg(A, B):
        for i in range(A.shape[0]):
            B[i] = -A[i]

    N = 128
    A = np.random.rand(N).astype(np.float64) * 20 - 10
    B = np.random.rand(N).astype(np.float64)

    highway_neg(A, B)
    assert np.allclose(B, -A)

    sdfg = highway_neg.last_sdfg
    stats = sdfg.loop_report()
    assert stats["HIGHWAY"] == 1
