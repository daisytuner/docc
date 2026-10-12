"""Codegens of docc targets, as the targets declare them in C++ (``DoccTarget::codegens``)."""

import importlib
from typing import Any, Callable, Dict, Optional, Tuple

from docc.sdfg import target_codegens


def _codegens(target: str) -> Dict[str, Dict[str, str]]:
    # Not cached: target plugins may register while the process runs.
    return {codegen["name"]: codegen for codegen in target_codegens(target)}


def get_target_codegens(target: str) -> Tuple[str, ...]:
    """Codegens the target supports, default first."""
    return tuple(codegen["name"] for codegen in target_codegens(target))


def get_target_default_codegen(target: str) -> str:
    """Codegen used when none is selected."""
    return get_target_codegens(target)[0]


def _codegen(target: str, codegen: str) -> Dict[str, str]:
    codegens = _codegens(target)
    if codegen not in codegens:
        raise ValueError(f"Target '{target}' does not support codegen '{codegen}'")
    return codegens[codegen]


def codegen_artifact(target: str, codegen: str, sdfg_name: str) -> str:
    """File name of what the compile step writes for ``codegen`` of ``target``."""
    return _codegen(target, codegen)["artifact"].format(name=sdfg_name)


def is_python_module(target: str, codegen: str) -> bool:
    """Whether the artifact of ``codegen`` is a Python module rather than a shared library."""
    return _codegen(target, codegen)["kind"] == "python_module"


def _entry_point(spec: str) -> Callable[..., Any]:
    module, _, function = spec.partition(":")
    return getattr(importlib.import_module(module), function)


def build_artifact(
    target: str, codegen: str, artifact_path: str, sdfg_name: str
) -> None:
    """Run the Python build step ``codegen`` declares after compiling, if any."""
    build = _codegen(target, codegen)["build"]
    if build:
        _entry_point(build)(artifact_path, sdfg_name)


def load_artifact(
    target: str, codegen: str, artifact_path: str, sdfg_name: str
) -> Optional[Callable[..., Any]]:
    """The callable running the artifact, if ``codegen`` declares a Python loader for it; None otherwise."""
    loader = _codegen(target, codegen)["loader"]
    if not loader:
        return None
    return _entry_point(loader)(artifact_path, sdfg_name)
