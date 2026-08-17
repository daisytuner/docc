import os
import sys
import re
from typing import cast

import pytest
import torch

# Ensure the pytorch project root is importable so that top-level packages such
# as ``tests`` and ``benchmarks`` resolve when this benchmark module is
# collected directly by pytest (e.g. ``pytest benchmarks/torch/model_zoo/...``).
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


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


@pytest.fixture(autouse=True)
def seed_rng():
    """Set a fixed random seed before each test for reproducibility."""
    torch.manual_seed(815)


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
