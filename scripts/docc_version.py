"""scikit-build-core metadata provider that supplies the version of docc's Python packages.

The docc-compiler, docc-ai and docc-pytorch pyprojects declare ``dynamic = ["version"]`` and point
``[tool.scikit-build.metadata.version]`` at this module (``provider = "docc_version"``,
``provider-path = "../scripts"``). scikit-build-core imports it from ``provider-path`` and calls
:func:`dynamic_metadata` with the remaining keys of that table as ``settings``.

It applies the same rule as docc's top-level CMakeLists.txt, so the wheel version and the version
compiled into the C++ libraries always agree:

* Release builds: the release workflow exports ``DOCC_RELEASE_VERSION`` (``MAJOR.MINOR.PATCH``,
  allocated by the Daisytuner backend), which is returned unchanged.
* Dev builds: ``MAJOR.MINOR`` from ``VERSION_BASE``, marked as a PEP 440 dev release and tagged with
  the commit hash, e.g. ``0.8.dev0+gabc1234`` (CMake: ``0.8+gabc1234``). The backend is never contacted.

Only the standard library is used, because the isolated build environment contains nothing but the
``[build-system] requires`` of the package being built.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

_BASE_PATTERN = re.compile(r"\d+\.\d+")
_VERSION_PATTERN = re.compile(r"\d+\.\d+\.\d+")

# Keys accepted in [tool.scikit-build.metadata.version] besides provider and provider-path.
_SETTINGS_KEYS = {"input"}


def _git_short_hash(cwd: Path) -> str:
    """Returns the short hash of the commit checked out in ``cwd``.

    Args:
        cwd: A directory inside the docc repository.

    Returns:
        The short commit hash, or ``"unknown"`` if git is missing or refuses the checkout (as it does
        in a cibuildwheel container, where the repository belongs to a different user).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def dynamic_metadata(field: str, settings: Mapping[str, Any]) -> str:
    """Computes the package version (scikit-build-core dynamic-metadata hook).

    Args:
        field: The metadata field to provide. Only ``"version"`` is supported.
        settings: The provider settings from pyproject.toml. ``input`` is the path to docc's
            ``VERSION_BASE`` file, relative to the package directory (the build's working directory).

    Returns:
        ``DOCC_RELEASE_VERSION`` for release builds, otherwise ``MAJOR.MINOR.dev0+g<hash>``.

    Raises:
        RuntimeError: If the field or settings are unsupported, or if ``DOCC_RELEASE_VERSION`` or
            ``VERSION_BASE`` is malformed.
    """
    # Validate the configuration first, so a typo in a pyproject fails loudly instead of being ignored.
    if field != "version":
        raise RuntimeError(f"docc_version only provides 'version', not {field!r}")
    unknown = set(settings) - _SETTINGS_KEYS
    if unknown:
        raise RuntimeError(f"Unsupported docc_version settings: {sorted(unknown)}")
    if "input" not in settings:
        raise RuntimeError(
            "docc_version needs the 'input' setting (path to VERSION_BASE)"
        )

    base_file = Path(settings["input"])
    base = base_file.read_text(encoding="utf-8").strip()
    if not _BASE_PATTERN.fullmatch(base):
        raise RuntimeError(
            f"{base_file} must contain MAJOR.MINOR (e.g. 0.8), got {base!r}"
        )

    # Release build: use the version the release workflow got from the backend.
    release = os.environ.get("DOCC_RELEASE_VERSION", "").strip()
    if release:
        if not _VERSION_PATTERN.fullmatch(release):
            raise RuntimeError(
                f"DOCC_RELEASE_VERSION must be MAJOR.MINOR.PATCH (e.g. 0.8.4), got {release!r}"
            )
        # Same guard as in CMake: a stale DOCC_RELEASE_VERSION from another series would mislabel the wheel.
        if not release.startswith(base + "."):
            raise RuntimeError(
                f"DOCC_RELEASE_VERSION {release!r} does not belong to VERSION_BASE {base!r}"
            )
        return release

    # Dev build: ".dev0" sorts below the release of the same number; "+g<hash>" is a PEP 440 local
    # label identifying the commit (PyPI rejects local labels, so dev wheels can't be published by mistake).
    return f"{base}.dev0+g{_git_short_hash(base_file.resolve().parent)}"
