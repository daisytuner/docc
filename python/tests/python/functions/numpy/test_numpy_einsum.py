from docc.python import native
import json
import pytest
import numpy as np
import math

from docc.compiler.docc_program import DoccOptions


@pytest.mark.parametrize(
    "default,override,expected",
    [
        (True, None, True),
        (True, False, False),
        (False, None, False),
        (False, True, True),
    ],
)
def test_target_einsum_default_can_be_overridden(
    monkeypatch, default, override, expected
):
    monkeypatch.setattr(
        "docc.compiler.docc_program.target_default_einsum", lambda target: default
    )
    assert DoccOptions(einsum=override).einsum is expected


def _library_codes(value):
    if isinstance(value, dict):
        if "code" in value:
            yield value["code"]
        for item in value.values():
            yield from _library_codes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _library_codes(item)


@pytest.mark.parametrize("precision", ["native", "ozaki1"])
def test_cluster_detection_preserves_scaled_gemm_accumulation(precision):
    @native(einsum=True, symbolic_shapes=False, contraction_precision=precision)
    def accumulate(alpha, a, b, out):
        for i in range(3):
            for j in range(5):
                for k in range(4):
                    out[i, j] += alpha * a[i, k] * b[k, j]

    a = np.arange(12.0).reshape(3, 4) / 8
    b = np.arange(20.0).reshape(4, 5) / 16
    out = np.full((3, 5), 2.5)
    sdfg = accumulate.to_sdfg(1.5, a, b, out)
    sdfg.einsum()
    sdfg.validate()
    assert "GEMM" in list(_library_codes(json.loads(sdfg.to_json())))
    sdfg.einsum()
    sdfg.validate()
    assert list(_library_codes(json.loads(sdfg.to_json()))) == ["GEMM"]
    accumulate(1.5, a, b, out)
    np.testing.assert_allclose(out, 2.5 + 1.5 * (a @ b))


@pytest.mark.parametrize("subtract", [False, True])
def test_cluster_detection_preserves_reduction_initial_value(subtract):
    @native(einsum=True)
    def add(a, b):
        total = 2.5
        for i in range(7):
            total += a[i] * b[i]
        return total

    @native(einsum=True)
    def sub(a, b):
        total = 2.5
        for i in range(7):
            total -= a[i] * b[i]
        return total

    a, b = np.arange(7.0) / 8, np.arange(7.0) / 16
    program = sub if subtract else add
    expected = 2.5 + (-1 if subtract else 1) * (a @ b)
    assert program(a, b) == pytest.approx(expected)


def test_cluster_detection_keeps_shared_products_and_neighboring_operations():
    @native(einsum=True)
    def reductions(a, b, out):
        left = 2.0
        right = 3.0
        for i in range(7):
            product = a[i] * b[i]
            left += product
            right -= product
            out[i] = product + 1.0
        return left + right

    a, b = np.arange(7.0) / 8, np.arange(7.0) / 16
    out = np.zeros(7)
    assert reductions(a, b, out) == pytest.approx(5.0)
    np.testing.assert_allclose(out, a * b + 1.0)


@pytest.mark.parametrize("detect", [False, True])
def test_einsum_dot_product(detect):

    @native(einsum=detect)
    def dot_product(a, b) -> float:
        return np.einsum("i,i->", a, b)

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    # (1.0 * 4.0) + (2.0 * 5.0) + (3.0 * 6.0) = 4.0 + 10.0 + 18.0 = 32.0
    assert dot_product(a, b) == 32.0


def test_einsum_1d_sum():

    @native
    def array_sum(a) -> float:
        return np.einsum("i->", a)

    a = np.array([1.0, 2.0, 3.0, 4.0])
    # 1.0 + 2.0 + 3.0 + 4.0 = 10.0
    assert array_sum(a) == 10.0


def test_einsum_outer_product():

    @native
    def outer_product(a, b):
        return np.einsum("i,j->ij", a, b)

    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    result = outer_product(a, b)
    expected = np.array([[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]])
    assert np.allclose(result, expected)


def test_einsum_matrix_transpose():

    @native
    def transpose(m):
        return np.einsum("ij->ji", m)

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = transpose(m)
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert np.allclose(result, expected)


def test_einsum_matrix_trace():

    @native
    def trace(m) -> float:
        return np.einsum("ii->", m)

    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    # 1.0 + 5.0 + 9.0 = 15.0
    assert trace(m) == 15.0


def test_einsum_matrix_vector_mul():

    @native
    def matvec(m, v):
        return np.einsum("ij,j->i", m, v)

    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    v = np.array([5.0, 6.0])
    result = matvec(m, v)
    # [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    expected = np.array([17.0, 39.0])
    assert np.allclose(result, expected)


@pytest.mark.parametrize("detect", [False, True])
def test_einsum_matrix_mul(detect):

    @native(einsum=detect)
    def matmul(a, b):
        return np.einsum("ij,jk->ik", a, b)

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    result = matmul(a, b)
    expected = np.array([[19.0, 22.0], [43.0, 50.0]])
    assert np.allclose(result, expected)


def test_einsum_3d_sum_all():

    @native
    def sum_all(t) -> float:
        return np.einsum("ijk->", t)

    t = np.arange(1.0, 9.0).reshape(2, 2, 2)
    # 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36.0
    assert sum_all(t) == 36.0


def test_einsum_3d_sum_axis():

    @native
    def sum_last_axis(t):
        return np.einsum("ijk->ij", t)

    t = np.arange(1.0, 13.0).reshape(2, 2, 3)
    result = sum_last_axis(t)
    # Sum along last axis: [[1+2+3, 4+5+6], [7+8+9, 10+11+12]] = [[6, 15], [24, 33]]
    expected = np.array([[6.0, 15.0], [24.0, 33.0]])
    assert np.allclose(result, expected)


def test_einsum_3d_permute():

    @native
    def permute(t):
        return np.einsum("ijk->kji", t)

    t = np.arange(1.0, 25.0).reshape(2, 3, 4)
    result = permute(t)
    expected = np.transpose(t, (2, 1, 0))
    assert np.allclose(result, expected)


def test_einsum_3d_diagonal():

    @native
    def diagonal_sum(t) -> float:
        return np.einsum("iii->", t)

    t = np.arange(1.0, 28.0).reshape(3, 3, 3)
    # Diagonal elements: t[0,0,0]=1, t[1,1,1]=14, t[2,2,2]=27
    # 1 + 14 + 27 = 42
    assert diagonal_sum(t) == 42.0


def test_einsum_batch_matmul():

    @native
    def batch_matmul(a, b):
        return np.einsum("bij,bjk->bik", a, b)

    # Batch of 2 matrices: 2x2 @ 2x3
    a = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    b = np.array(
        [[[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    )
    result = batch_matmul(a, b)
    expected = np.array(
        [[[1.0, 2.0, 1.0], [3.0, 4.0, 3.0]], [[11.0, 11.0, 11.0], [15.0, 15.0, 15.0]]]
    )
    assert np.allclose(result, expected)


@pytest.mark.parametrize("detect", [False, True])
def test_einsum_3d_contract(detect):

    @native(einsum=detect)
    def contract_middle(a, b):
        return np.einsum("ijk,jl->ikl", a, b)

    a = np.ones((2, 3, 4))
    b = np.ones((3, 5))
    result = contract_middle(a, b)
    # Contracting over j (size 3), result shape is (2, 4, 5)
    expected = np.full((2, 4, 5), 3.0)
    assert np.allclose(result, expected)


def test_einsum_batched_transpose_matvec_reserved_indices():
    """Batched transpose-matvec using index letters that collide with symbolic
    constants (`e` = Euler's number, `i` = imaginary unit). LULESH hourglass
    pattern `einsum('eji,ej->ei', ...)`.
    """

    @native
    def es_ei(hourgam, xd, out):
        out[:] = np.einsum("eji,ej->ei", hourgam, xd)

    ne = 5
    hg = np.arange(ne * 8 * 4, dtype=np.float64).reshape(ne, 8, 4)
    xd = np.arange(ne * 8, dtype=np.float64).reshape(ne, 8)
    out = np.zeros((ne, 4), dtype=np.float64)
    es_ei(hg.copy(), xd.copy(), out)
    expected = np.einsum("eji,ej->ei", hg, xd)
    assert np.allclose(out, expected)


def test_einsum_batched_matvec_reserved_indices_transpose_back():
    """The reverse contraction `einsum('eji,ei->ej', ...)` (LULESH)."""

    @native
    def es_ej(hourgam, hxx, out):
        out[:] = np.einsum("eji,ei->ej", hourgam, hxx)

    ne = 4
    hg = np.arange(ne * 8 * 4, dtype=np.float64).reshape(ne, 8, 4)
    hxx = np.arange(ne * 4, dtype=np.float64).reshape(ne, 4)
    out = np.zeros((ne, 8), dtype=np.float64)
    es_ej(hg.copy(), hxx.copy(), out)
    expected = np.einsum("eji,ei->ej", hg, hxx)
    assert np.allclose(out, expected)
