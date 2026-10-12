from typing import Any, Callable, Dict, Iterable, Optional
from docc.sdfg import StructuredSDFG

TargetCompileFn = Callable[[StructuredSDFG, str, str, bool, Dict[str, Any]], str]

_target_compile_registry: dict[str, TargetCompileFn] = {}
_target_codegen_registry: dict[str, frozenset[str]] = {}

# Codegens of the built-in targets; any other target supports only "cpp" unless it registers more.
_BUILTIN_TARGET_CODEGENS: dict[str, frozenset[str]] = {
    "none": frozenset({"cpp", "numpy"}),
    "sequential": frozenset({"cpp", "numpy"}),
}


def register_target_overrides(
    name: str,
    compile_fn: Optional[TargetCompileFn] = None,
    codegens: Optional[Iterable[str]] = None,
) -> None:
    """Override the compile step or the codegens of this target.

    Args:
        name: Target name (e.g., "openmp")
        compile_fn: Function that performs compile step; receives the selected codegen as options["codegen"]
        codegens: Codegens the target supports (e.g. {"cpp", "numpy"})
    """
    if compile_fn is not None:
        _target_compile_registry[name] = compile_fn

    if codegens is not None:
        _target_codegen_registry[name] = frozenset(codegens)


def unregister_target(name: str) -> None:
    """Unregister a target's overrides."""
    _target_compile_registry.pop(name, None)
    _target_codegen_registry.pop(name, None)


def get_target_compile_fn(name: str) -> Optional[TargetCompileFn]:
    """Get a registered target compile function, or None if not found."""
    return _target_compile_registry.get(name)


def get_target_codegens(name: str) -> frozenset[str]:
    """Codegens the target supports."""
    if name in _target_codegen_registry:
        return _target_codegen_registry[name]
    return _BUILTIN_TARGET_CODEGENS.get(name, frozenset({"cpp"}))


def reset_target_registry() -> None:
    """Clear all global state of the registry"""
    _target_compile_registry.clear()
    _target_codegen_registry.clear()
