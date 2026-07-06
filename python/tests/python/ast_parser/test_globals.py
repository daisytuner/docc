from docc.python import native
import numpy as np
import sys
import pytest

# Module-level constants
GLOBAL_FLOAT = 0.5
GLOBAL_INT = 42
SCALE_FACTOR = 0.25
BET_M = 0.5
BET_P = 0.5


def test_global_float_constant():
    """Test that global float constants are captured."""

    @native
    def use_global_float(arr) -> float:
        return arr[0] * GLOBAL_FLOAT

    arr = np.array([2.0], dtype=np.float64)
    result = use_global_float(arr)
    assert result == 1.0  # 2.0 * 0.5


def test_global_int_constant():
    """Test that global int constants are captured."""

    @native
    def use_global_int(arr) -> float:
        return arr[0] + GLOBAL_INT

    arr = np.array([8.0], dtype=np.float64)
    result = use_global_int(arr)
    assert result == 50.0  # 8.0 + 42


def test_multiple_global_constants():
    """Test using multiple global constants in one function."""

    @native
    def use_multiple_globals(arr) -> float:
        x = arr[0] * BET_M
        y = arr[1] * BET_P
        return x + y

    arr = np.array([4.0, 6.0], dtype=np.float64)
    result = use_multiple_globals(arr)
    assert result == 5.0  # 4.0 * 0.5 + 6.0 * 0.5


def test_global_in_expression():
    """Test global constant used in complex expressions."""

    @native
    def global_in_expr(arr) -> float:
        result = SCALE_FACTOR * (arr[0] + arr[1])
        return result

    arr = np.array([3.0, 5.0], dtype=np.float64)
    result = global_in_expr(arr)
    assert result == 2.0  # 0.25 * (3.0 + 5.0)


@pytest.mark.skipif(sys.platform == "darwin", reason="Segfault on macOS")
def test_global_with_array_slice():
    """Test global constant with array slicing (vadv pattern)."""

    @native
    def global_with_slice(wcon, k) -> float:
        gcv = SCALE_FACTOR * (wcon[1:, :, k + 1] + wcon[:-1, :, k + 1])
        cs = gcv * BET_M
        return cs[0, 0]

    wcon = np.ones((4, 3, 5), dtype=np.float64)
    result = global_with_slice(wcon, 1)
    # 0.25 * (1 + 1) * 0.5 = 0.25
    assert result == 0.25


def test_global_constant_not_overwritten_by_local():
    """Test that global constant is used when no local variable exists."""

    @native
    def global_vs_local(arr) -> float:
        # Use global SCALE_FACTOR (0.25)
        return arr[0] * SCALE_FACTOR

    arr = np.array([8.0], dtype=np.float64)
    result = global_vs_local(arr)
    assert result == 2.0  # 8.0 * 0.25


# Module-level constant array (e.g. LULESH's `gamma`).
GLOBAL_MATRIX = np.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
    dtype=np.float64,
)
GLOBAL_IVEC = np.array([10, 20, 30, 40], dtype=np.int64)


def test_global_array_element_access():
    """A module-global constant array can be indexed element-wise."""

    @native
    def copy_matrix(out):
        for i in range(4):
            for j in range(3):
                out[i, j] = GLOBAL_MATRIX[i, j]

    out = np.zeros((4, 3), dtype=np.float64)
    copy_matrix(out)
    assert np.array_equal(out, GLOBAL_MATRIX)


def test_global_array_partial_index_row():
    """Partial indexing of a global 2-D array yields a row (A[i] == A[i, :])."""

    @native
    def copy_rows(out):
        for i in range(4):
            out[i, :] = GLOBAL_MATRIX[i]

    out = np.zeros((4, 3), dtype=np.float64)
    copy_rows(out)
    assert np.array_equal(out, GLOBAL_MATRIX)


def test_global_int_array():
    """A module-global integer constant array is materialized correctly."""

    @native
    def copy_ivec(out):
        for i in range(4):
            out[i] = GLOBAL_IVEC[i]

    out = np.zeros(4, dtype=np.int64)
    copy_ivec(out)
    assert np.array_equal(out, GLOBAL_IVEC)


def test_global_array_matvec():
    """A matvec against a row of a global matrix (LULESH hourglass pattern).

    Exercises the offset-operand copy in the matmul lowering: `A @ M[i]`.
    """

    @native
    def matvec_rows(a, out):
        for i in range(4):
            out[:, i] = a @ GLOBAL_MATRIX[i]

    a = np.arange(15, dtype=np.float64).reshape(5, 3)
    out = np.zeros((5, 4), dtype=np.float64)
    matvec_rows(a.copy(), out)
    exp = np.zeros((5, 4), dtype=np.float64)
    for i in range(4):
        exp[:, i] = a @ GLOBAL_MATRIX[i]
    assert np.allclose(out, exp)


# Nested dict of integer bit masks (LULESH XI/ETA/ZETA boundary-condition tables)
BC_MASKS = {
    "M": {"mask": 0x007, "SYMM": 0x001, "FREE": 0x002, "COMM": 0x004},
    "P": {"mask": 0x038, "SYMM": 0x008, "FREE": 0x010, "COMM": 0x020},
}


def test_global_dict_constant_fold():
    """A subscript chain into a nested global dict of ints folds to a literal
    at compile time (LULESH XI["M"]["mask"] boundary-condition pattern)."""

    @native
    def mask_and(bc_mask, out):
        out[:] = bc_mask & BC_MASKS["M"]["mask"]

    bc_mask = np.array([0x7, 0x3, 0x38, 0x1], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    mask_and(bc_mask.copy(), out)
    assert np.array_equal(out, bc_mask & 0x007)


def test_global_dict_constant_scalar():
    """A folded dict constant is usable as a plain scalar operand."""

    @native
    def add_symm(arr) -> int:
        return arr[0] + BC_MASKS["P"]["SYMM"]

    arr = np.array([1], dtype=np.int64)
    assert add_symm(arr) == 1 + 0x008


def _region_mask(bc, bc_mask):
    # `bc` is bound to a global dict constant via inline substitution.
    return bc_mask & bc["P"]["mask"]


def test_global_dict_constant_inlined():
    """A global dict constant passed as an argument to an inlined helper is
    substituted into the body and folded (LULESH _calc_*_region_bc pattern)."""

    @native
    def use_region(bc_mask, out):
        out[:] = _region_mask(BC_MASKS, bc_mask)

    bc_mask = np.array([0x7, 0x3, 0x38, 0x1], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    use_region(bc_mask.copy(), out)
    assert np.array_equal(out, bc_mask & 0x038)
