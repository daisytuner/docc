"""Artifacts of targets with their own code generator, as the targets declare them in C++ (``DoccTarget::artifact``)."""

import importlib
from typing import Any, Callable, Dict, Optional

from docc.sdfg import target_artifact


def _artifact(target: Optional[str]) -> Optional[Dict[str, str]]:
    # Not cached: target plugins may register while the process runs.
    return target_artifact(target) if target else None


def artifact_file(target: Optional[str], sdfg_name: str) -> str:
    """File name of what the compile step writes for ``target``: its own artifact, else the C++ shared library."""
    artifact = _artifact(target)
    if artifact is None:
        return f"lib{sdfg_name}.so"
    return artifact["file"].format(name=sdfg_name)


def is_python_module(target: Optional[str]) -> bool:
    """Whether ``target``'s artifact is a Python module rather than a shared library."""
    artifact = _artifact(target)
    return artifact is not None and artifact["kind"] == "python_module"


def _entry_point(spec: str) -> Callable[..., Any]:
    module, _, function = spec.partition(":")
    return getattr(importlib.import_module(module), function)


def build_artifact(target: Optional[str], artifact_path: str, sdfg_name: str) -> None:
    """Run the Python build step ``target``'s artifact declares after compiling, if any."""
    artifact = _artifact(target)
    if artifact is not None and artifact["build"]:
        _entry_point(artifact["build"])(artifact_path, sdfg_name)


def load_artifact(
    target: Optional[str], artifact_path: str, sdfg_name: str
) -> Optional[Callable[..., Any]]:
    """The callable running the artifact, if ``target``'s artifact declares a Python loader; None otherwise."""
    artifact = _artifact(target)
    if artifact is None or not artifact["loader"]:
        return None
    return _entry_point(artifact["loader"])(artifact_path, sdfg_name)
