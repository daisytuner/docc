import pytest

from docc.sdfg import (
    PrimitiveType,
    Scalar,
    Pointer,
    Structure,
    Tensor,
)
from docc.python.types import sdfg_type_from_type


class Point2D:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


def test_python_class():
    result = sdfg_type_from_type(Point2D)
    assert isinstance(result, Pointer)
    assert result.has_pointee_type()
    assert isinstance(result.pointee_type, Structure)
    assert result.pointee_type.name == "Point2D"
