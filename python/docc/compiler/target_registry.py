from typing import Callable, Optional, Dict, Any
from docc.sdfg import StructuredSDFG

TargetScheduleFn = Callable[[StructuredSDFG, str, Dict[str, Any]], None]
LegacyTargetScheduleFn = Callable[[StructuredSDFG, str], None]
TargetCompileFn = Callable[[StructuredSDFG, str, str, bool, Dict[str, Any]], str]
TargetExpandFn = Callable[[StructuredSDFG, str, Dict[str, Any]], None]

_target_expand_registry: dict[str, TargetExpandFn] = {}
_target_schedule_registry: dict[str, TargetScheduleFn] = {}
_target_compile_registry: dict[str, TargetCompileFn] = {}


def register_target(name: str, schedule_fn: Optional[LegacyTargetScheduleFn]) -> None:
    def wrapper(
        sdfg: StructuredSDFG, category: str, options: Optional[Dict[str, Any]] = None
    ) -> None:
        schedule_fn(sdfg, category)  # type: ignore

    register_target_overrides(name, schedule_fn=wrapper, compile_fn=None)


def register_target_overrides(
    name: str,
    schedule_fn: Optional[TargetScheduleFn],
    compile_fn: Optional[TargetCompileFn] = None,
    expand_fn: Optional[TargetExpandFn] = None,
) -> None:
    """Override the scheduling or compile step for this target.

    The schedule function will be called with:
    - sdfg: The StructuredSDFG to schedule (has _native_ptr for native access)
    - category: The target category (e.g., "desktop", "server")
    - options: (Optional) Dictionary of options

    Args:
        name: Target name (e.g., "openmp")
        schedule_fn: Function that performs scheduling transformations
        compile_fn: Function that performs compile step
        expand_fn: Function that performs the expand step
    """
    if schedule_fn is not None:
        _target_schedule_registry[name] = schedule_fn  # type: ignore

    if compile_fn is not None:
        _target_compile_registry[name] = compile_fn

    if expand_fn is not None:
        _target_expand_registry[name] = expand_fn


def unregister_target(name: str) -> None:
    """Unregister a custom target scheduler."""
    _target_schedule_registry.pop(name, None)
    _target_compile_registry.pop(name, None)
    _target_expand_registry.pop(name, None)


def get_target_schedule_fn(name: str) -> Optional[TargetScheduleFn]:
    """Get a registered target scheduler, or None if not found."""
    return _target_schedule_registry.get(name)


def get_target_compile_fn(name: str) -> Optional[TargetCompileFn]:
    """Get a registered target compile function, or None if not found."""
    return _target_compile_registry.get(name)


def get_target_expand_fn(name: str) -> Optional[TargetExpandFn]:
    """Get a registered target expand function, or None if not found."""
    return _target_expand_registry.get(name)


def reset_target_registry() -> None:
    """Clear all global state of the registry"""
    _target_expand_registry.clear()
    _target_schedule_registry.clear()
    _target_compile_registry.clear()
