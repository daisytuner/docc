from docc.python import native
import numpy as np


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Vec:
    """A struct with a single 1D array member plus a scalar length."""

    def __init__(self, a):
        self.a = a
        self.n = a.shape[0]


class TwoVecs:
    """A struct with several 1D array members and a scalar member."""

    def __init__(self, x, y, out, scale):
        self.x = x
        self.y = y
        self.out = out
        self.scale = scale


class Grid2D:
    """A struct with a 2D array member."""

    def __init__(self, a):
        self.a = a


def test_type_inference_from_object():
    from docc.sdfg import Structure, Pointer
    from docc.python import PythonProgram

    point = Point2D(3.0, 4.0)
    program = PythonProgram(lambda p: p, target="none")

    inferred_type = program._infer_type(point)
    assert isinstance(inferred_type, Pointer)
    assert inferred_type.has_pointee_type()
    assert isinstance(inferred_type.pointee_type, Structure)
    assert inferred_type.pointee_type.name == "Point2D"


def test_structure_member_zero():
    @native
    def get_x(p: Point2D) -> float:
        return p.x

    point = Point2D(3.0, 4.0)
    result = get_x(point)
    assert result == 3.0


def test_structure_member_one():
    @native
    def get_y(p: Point2D) -> float:
        return p.y

    point = Point2D(5.0, 7.0)
    result = get_y(point)
    assert result == 7.0


def test_structure_member_two():
    @native
    def get_z(p: Point3D) -> float:
        return p.z

    point = Point3D(1.0, 2.0, 3.0)
    result = get_z(point)
    assert result == 3.0


def test_structure_members():
    @native
    def distance_from_origin(p: Point2D) -> float:
        return (p.x * p.x + p.y * p.y) ** 0.5

    point = Point2D(3.0, 4.0)
    result = distance_from_origin(point)
    assert abs(result - 5.0) < 1e-10


# =============================================================================
# ARRAY MEMBERS (struct-of-arrays)
# =============================================================================


def test_array_member_read_element():
    """Read a single element from a 1D array member."""

    @native
    def read_first(v: Vec) -> float:
        return v.a[0]

    v = Vec(np.array([7.0, 8.0, 9.0], dtype=np.float64))
    assert read_first(v) == 7.0


def test_array_member_inplace_scale():
    """In-place elementwise write through an array member."""

    @native
    def scale(v: Vec):
        v.a[:] = v.a * 2.0

    a = np.arange(5, dtype=np.float64)
    v = Vec(a.copy())
    scale(v)
    np.testing.assert_array_equal(v.a, a * 2.0)


def test_array_member_element_write():
    """Write a single element of an array member in place."""

    @native
    def set_second(v: Vec):
        v.a[1] = 42.0

    a = np.arange(4, dtype=np.float64)
    v = Vec(a.copy())
    set_second(v)
    expected = a.copy()
    expected[1] = 42.0
    np.testing.assert_array_equal(v.a, expected)


def test_array_member_reduction():
    """Reduce over an array member."""

    @native
    def total(v: Vec) -> float:
        return np.sum(v.a)

    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    v = Vec(a)
    assert abs(total(v) - 10.0) < 1e-10


def test_array_member_with_scalar_member():
    """Combine an array member with a scalar member."""

    @native
    def add_scale(c: TwoVecs):
        c.out[:] = (c.x + c.y) * c.scale

    n = 6
    x = np.random.rand(n)
    y = np.random.rand(n)
    out = np.zeros(n)
    c = TwoVecs(x.copy(), y.copy(), out.copy(), 3.0)
    add_scale(c)
    np.testing.assert_allclose(c.out, (x + y) * 3.0)


def test_multiple_array_members():
    """Read and write across several array members of a struct."""

    @native
    def axpy(c: TwoVecs):
        c.out[:] = c.scale * c.x + c.y

    n = 8
    x = np.random.rand(n)
    y = np.random.rand(n)
    out = np.zeros(n)
    c = TwoVecs(x.copy(), y.copy(), out.copy(), 2.5)
    axpy(c)
    np.testing.assert_allclose(c.out, 2.5 * x + y)


def test_array_member_2d():
    """In-place update of a 2D array member."""

    @native
    def add_one(g: Grid2D):
        g.a[:] = g.a + 1.0

    a = np.arange(12, dtype=np.float64).reshape(3, 4)
    g = Grid2D(a.copy())
    add_one(g)
    np.testing.assert_array_equal(g.a, a + 1.0)


def test_array_member_augassign():
    """Augmented assignment (+=) directly on an array member attribute."""

    @native
    def accumulate(c: TwoVecs):
        c.out += c.x * c.scale

    n = 7
    x = np.random.rand(n)
    out = np.random.rand(n)
    c = TwoVecs(x.copy(), np.zeros(n), out.copy(), 2.0)
    accumulate(c)
    np.testing.assert_allclose(c.out, out + x * 2.0)


def test_array_member_augassign_2d():
    """Augmented assignment (+=) on a 2D array member attribute."""

    @native
    def bump(g: Grid2D):
        g.a += 5.0

    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    g = Grid2D(a.copy())
    bump(g)
    np.testing.assert_array_equal(g.a, a + 5.0)


class IndexedMesh:
    """A struct whose array member is indexed by another (integer) member."""

    def __init__(self, vals, idx):
        self.vals = vals
        self.idx = idx


def test_gather_with_member_index():
    """Gather where the index array is a struct member: obj.vals[obj.idx]."""

    @native
    def gather_member_index(m: IndexedMesh, out):
        out[:] = m.vals[m.idx]

    vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    idx = np.array([[0, 2], [4, 1], [3, 0]], dtype=np.int32)
    m = IndexedMesh(vals, idx)
    out = np.zeros((3, 2))
    gather_member_index(m, out)
    np.testing.assert_array_equal(out, vals[idx])


def test_array_member_attribute_assignment():
    """Whole-member attribute assignment: obj.attr = value (no subscript)."""

    @native
    def assign_members(c: TwoVecs, src):
        c.x = src[:, 0]
        c.out = src[:, 1] + 1.0

    src = np.random.rand(4, 3)
    c = TwoVecs(np.zeros(4), np.zeros(4), np.zeros(4), 1.0)
    assign_members(c, src)
    np.testing.assert_allclose(c.x, src[:, 0])
    np.testing.assert_allclose(c.out, src[:, 1] + 1.0)


def test_scalar_member_writeback():
    """A scalar struct member written in the kernel round-trips back to the
    Python object (LULESH `domain.dtcourant = ...` pattern)."""

    @native
    def set_scale(c: TwoVecs):
        c.scale = 42.0

    c = TwoVecs(np.zeros(4), np.zeros(4), np.zeros(4), 1.0)
    set_scale(c)
    assert c.scale == 42.0


def test_scalar_member_conditional_update():
    """A scalar member conditionally updated (min-reduction guard) round-trips."""

    @native
    def clamp_scale(c: TwoVecs, cand):
        c.scale = 100.0
        if cand < c.scale:
            c.scale = cand

    c = TwoVecs(np.zeros(4), np.zeros(4), np.zeros(4), 1.0)
    clamp_scale(c, 7.0)
    assert c.scale == 7.0

    c2 = TwoVecs(np.zeros(4), np.zeros(4), np.zeros(4), 1.0)
    clamp_scale(c2, 250.0)
    assert c2.scale == 100.0
