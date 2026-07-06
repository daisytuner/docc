from docc.python import native
import numpy as np


def test_tuple_unpacking():
    @native
    def unpack_two_values() -> int:
        a, b = 1, 2
        return a + b

    assert unpack_two_values() == 3

    @native
    def unpack_three_values() -> int:
        a, b, c = 1, 2, 3
        return a + b + c

    assert unpack_three_values() == 6

    @native
    def unpack_expressions(a, b) -> int:
        d, e = a + 1, b * 2
        return d + e

    assert unpack_expressions(2, 3) == 9


def test_multiple_assignment():
    @native
    def double_assign(in_val: int) -> int:
        a = b = in_val
        return a + b

    assert double_assign(5) == 10

    @native
    def double_assign_expression(in_val: int) -> int:
        a = b = in_val + 2
        return a * b

    assert double_assign_expression(3) == 25

    @native
    def triple_assign(val: int) -> int:
        a = b = c = val
        return a + b + c

    assert triple_assign(10) == 30


def _make_pair(a, b):
    return a + b, a - b


def _make_triple(a):
    return a * 2.0, a + 1.0, a - 1.0


def test_multi_return_unpacking():
    """Tuple unpacking from an inlined function that returns multiple values."""

    @native
    def use_pair(a, b, out):
        s, d = _make_pair(a, b)
        out[:] = s * d

    n = 5
    a = np.random.rand(n)
    b = np.random.rand(n)
    out = np.zeros(n)
    use_pair(a, b, out)
    np.testing.assert_allclose(out, (a + b) * (a - b))

    @native
    def use_triple(a, out):
        x, y, z = _make_triple(a)
        out[:] = x + y + z

    a2 = np.random.rand(n)
    out2 = np.zeros(n)
    use_triple(a2, out2)
    np.testing.assert_allclose(out2, a2 * 2.0 + (a2 + 1.0) + (a2 - 1.0))


def _face(pf, nodes):
    a, b, c, d = nodes
    pf[:, a] += 1.0
    pf[:, b] += 2.0
    pf[:, c] += 3.0
    pf[:, d] += 4.0


def test_tuple_literal_inline_arg():
    """Pass a compile-time tuple literal to an inlined function and unpack it."""

    @native
    def use_faces(pf):
        _face(pf, (0, 1, 2, 3))
        _face(pf, (3, 2, 1, 0))

    pf = np.zeros((5, 4))
    use_faces(pf)

    expected = np.zeros((5, 4))
    for nodes in [(0, 1, 2, 3), (3, 2, 1, 0)]:
        a, b, c, d = nodes
        expected[:, a] += 1.0
        expected[:, b] += 2.0
        expected[:, c] += 3.0
        expected[:, d] += 4.0

    np.testing.assert_allclose(pf, expected)


def test_conditional_expression():
    """Conditional (ternary) expression: `x if cond else y`."""

    @native
    def pick(a, out):
        out[0] = 1.0 if a[0] != 0 else -1.0

    out = np.zeros(1)
    pick(np.array([5.0]), out)
    assert out[0] == 1.0

    out2 = np.zeros(1)
    pick(np.array([0.0]), out2)
    assert out2[0] == -1.0

    # Ternary feeding another operation (lulesh clip-bound pattern).
    @native
    def clip_bounds(a, out):
        lower = a[1] if a[1] != 0 else -100.0
        upper = a[2] if a[2] != 0 else 100.0
        out[:] = np.clip(a, lower, upper)

    a = np.array([5.0, 0.0, 2.0, 9.0])
    out3 = np.zeros(4)
    clip_bounds(a, out3)
    lower = a[1] if a[1] != 0 else -100.0
    upper = a[2] if a[2] != 0 else 100.0
    np.testing.assert_allclose(out3, np.clip(a, lower, upper))
