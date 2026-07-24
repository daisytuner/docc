"""Configuration of the pytests for the PyTorch frontend

Each pytest gets an argument called `target` of type string that represents the SDFG target. Pytest
can then be called with ``--target`` option to set it. Default is ``--target=none``.

To make a test ignore targets, it can be marked with ``unsupported_targets``. For example, say you
do not want a test to run on GPUs, put this code before it:
@pytest.mark.unsupported_targets("cuda", "rocm")
def test_foo(target: str) -> None:
    ...

Some features are not available in older PyTorch versions. Marking a test with
``minimum_pytorch_version`` and a tuple version skips the test if the currently installed PyTorch
version is lower than the minimum required one:
@pytest.mark.minimum_pytorch_version((2, 9 , 1))
def test_foo(target: str) -> None:
    ...
"""

import re
from typing import cast

import pytest


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    # Keep only the numeric x.y.z prefix and ignore local/build suffixes.
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    if not match:
        return tuple()
    return tuple(int(part) for part in match.group(1).split("."))


def _normalize_for_compare(version: tuple[int, ...], length: int) -> tuple[int, ...]:
    if len(version) >= length:
        return version[:length]
    return version + (0,) * (length - len(version))


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
    config.addinivalue_line(
        "markers",
        "minimum_pytorch_version(version_tuple): mark test to require a minimum PyTorch version",
    )


@pytest.fixture
def target(request: pytest.FixtureRequest) -> str:
    return cast(str, request.config.getoption("--target"))


def pytest_collection_modifyitems(config, items):
    import torch

    selected_target = config.getoption("--target")
    torch_version = _parse_version_tuple(torch.__version__)

    for item in items:
        marker = item.get_closest_marker("unsupported_targets")
        if marker and selected_target in marker.args:
            item.add_marker(
                pytest.mark.skip(reason=f"Test skipped for target '{selected_target}'")
            )

        min_version_marker = item.get_closest_marker("minimum_pytorch_version")
        if min_version_marker:
            if len(min_version_marker.args) != 1 or not isinstance(
                min_version_marker.args[0], tuple
            ):
                raise pytest.UsageError(
                    "minimum_pytorch_version marker expects exactly one tuple argument, "
                    "for example @pytest.mark.minimum_pytorch_version((2, 9, 1))"
                )

            required_version = min_version_marker.args[0]
            if not all(isinstance(part, int) for part in required_version):
                raise pytest.UsageError(
                    "minimum_pytorch_version tuple must contain only integers"
                )

            compare_len = max(len(torch_version), len(required_version))
            normalized_current = _normalize_for_compare(torch_version, compare_len)
            normalized_required = _normalize_for_compare(required_version, compare_len)

            if normalized_current < normalized_required:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            "Test skipped because it requires PyTorch >= "
                            f"{'.'.join(str(x) for x in required_version)} "
                            f"(found {torch.__version__})"
                        )
                    )
                )
