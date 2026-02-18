from docc.python import native


def test_int_arithmetic():
    @native
    def int_add(a, b) -> int:
        return a + b

    assert int_add(5, 3) == 8

    @native
    def int_sub(a, b) -> int:
        return a - b

    assert int_sub(5, 3) == 2

    @native
    def int_mul(a, b) -> int:
        return a * b

    assert int_mul(5, 3) == 15

    @native
    def int_div(a, b) -> int:
        return a // b

    assert int_div(10, 3) == 3
    assert int_div(10, 2) == 5

    @native
    def int_neg(a) -> int:
        return -a

    assert int_neg(5) == -5

    @native
    def int_mod(a, b) -> int:
        return a % b

    assert int_mod(10, 3) == 1
    assert int_mod(-10, 3) == 2  # Python behavior
    assert int_mod(10, 5) == 0

    @native
    def int_pow(a, b) -> int:
        return a**b

    assert int_pow(2, 3) == 8
    assert int_pow(3, 2) == 9
    assert int_pow(5, 0) == 1


def test_float_arithmetic():
    @native
    def fp_add(a, b) -> float:
        return a + b

    assert fp_add(5.5, 2.5) == 8.0

    @native
    def fp_sub(a, b) -> float:
        return a - b

    assert fp_sub(5.5, 2.5) == 3.0

    @native
    def fp_mul(a, b) -> float:
        return a * b

    assert fp_mul(5.5, 2.0) == 11.0

    @native
    def fp_div(a, b) -> float:
        return a / b

    assert fp_div(5.0, 2.0) == 2.5

    @native
    def fp_neg(a) -> float:
        return -a

    assert fp_neg(5.5) == -5.5

    @native
    def fp_mod(a, b) -> float:
        return a % b

    # Python's floored modulo (result has same sign as divisor)
    assert fp_mod(5.5, 2.0) == 1.5
    assert fp_mod(-5.5, 2.0) == 0.5  # Python: -5.5 % 2.0 = 0.5 (not -1.5 like fmod)
    assert fp_mod(5.5, -2.0) == -0.5  # Python: 5.5 % -2.0 = -0.5 (not 1.5 like fmod)
    assert fp_mod(-5.5, -2.0) == -1.5


def test_bitwise_ops():
    @native
    def bit_or(a, b) -> int:
        return a | b

    assert bit_or(5, 3) == 7  # 101 | 011 = 111 (7)

    @native
    def bit_and(a, b) -> int:
        return a & b

    assert bit_and(5, 3) == 1  # 101 & 011 = 001 (1)

    @native
    def bit_xor(a, b) -> int:
        return a ^ b

    assert bit_xor(5, 3) == 6  # 101 ^ 011 = 110 (6)

    @native
    def bit_not(a) -> int:
        return ~a

    assert bit_not(5) == -6  # ~5 = -6 in Python (two's complement)


def test_logical_ops():
    # Logical ops use control flow, but verify they work
    @native
    def logical_and(a, b) -> int:
        if a > 0 and b > 0:
            return 1
        return 0

    assert logical_and(1, 1) == 1
    assert logical_and(1, 0) == 0

    @native
    def logical_or(a, b) -> int:
        if a > 0 or b > 0:
            return 1
        return 0

    assert logical_or(1, 0) == 1
    assert logical_or(0, 0) == 0

    @native
    def logical_not(a) -> int:
        if not (a > 0):
            return 1
        return 0

    assert logical_not(0) == 1
    assert logical_not(1) == 0
