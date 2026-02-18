from docc.python import native


class Point2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


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
