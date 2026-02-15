import pytest
from docc.sdfg import DebugInfo, StructuredSDFGBuilder, Scalar, PrimitiveType


def test_debug_info_creation():
    di = DebugInfo("test.py", 10, 5, 12, 20)
    assert di.filename == "test.py"
    assert di.start_line == 10
    assert di.start_column == 5
    assert di.end_line == 12
    assert di.end_column == 20
    assert di.function == ""

    di2 = DebugInfo("test.py", "my_func", 10, 5, 12, 20)
    assert di2.filename == "test.py"
    assert di2.function == "my_func"
    assert di2.start_line == 10


def test_builder_with_debug_info():
    builder = StructuredSDFGBuilder("test_sdfg")
    di = DebugInfo("test.py", 1, 1, 1, 10)

    builder.add_container("A", Scalar(PrimitiveType.Int64), False)
    builder.add_container("B", Scalar(PrimitiveType.Int64), False)

    builder.add_transition("A", "1", di)

    sdfg = builder.move()
    assert sdfg is not None


def test_control_flow_debug_info():
    builder = StructuredSDFGBuilder("test_cf")
    di = DebugInfo("test.py", 2, 1, 5, 1)

    builder.add_container("i", Scalar(PrimitiveType.Int64), False)

    builder.begin_for("i", "0", "10", "1", di)
    builder.end_for()

    sdfg = builder.move()
    assert sdfg is not None


def test_if_else_debug_info():
    builder = StructuredSDFGBuilder("test_if")
    di = DebugInfo("test.py", 3, 1, 6, 1)

    builder.begin_if("1 == 1", di)
    builder.end_if()

    sdfg = builder.move()
    assert sdfg is not None
