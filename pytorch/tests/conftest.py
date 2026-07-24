"""Configuration of the pytests for the PyTorch frontend

Each pytest gets an argument called `target` of type string that represents the SDFG target. Pytest
can then be called with ``--target`` option to set it. Default is ``--target=none``.

To make a test ignore targets, it can be marked with ``unsupported_targets``. For example, say you
do not want a test to run on GPUs, put this code before it:
@pytest.mark.unsupported_targets("cuda", "rocm")
def test_foo(target: str) -> None:
    ...
"""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--target",
        action="store",
        choices=["none", "sequential", "openmp", "cuda", "rocm"],
        default="none",
        help="Select the docc target.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "unsupported_targets(*targets): mark test as unsupported for given targets",
    )


@pytest.fixture
def target(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--target")


def pytest_collection_modifyitems(config, items):
    selected_target = config.getoption("--target")

    for item in items:
        marker = item.get_closest_marker("unsupported_targets")
        if marker and selected_target in marker.args:
            item.add_marker(
                pytest.mark.skip(reason=f"Test skipped for target '{selected_target}'")
            )
