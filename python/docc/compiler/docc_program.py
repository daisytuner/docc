from abc import ABC, abstractmethod
import sys
from typing import Any, Dict, Optional
import os
import re

from docc.sdfg import StructuredSDFG
from docc.sdfg._sdfg import (
    _enable_statistics,
    _statistics_enabled_by_env,
    _statistics_summary,
)
from docc.compiler.compiled_sdfg import CompiledSDFG
from docc.compiler.target_registry import (
    get_target_schedule_fn,
    get_target_compile_fn,
    get_target_expand_fn,
)


def _cuda_expand_fn(sdfg, category: str, kwargs: Dict[str, Any]) -> None:
    sdfg.expand_cuda()
    sdfg.expand()


def _rocm_expand_fn(sdfg, category: str, kwargs: Dict[str, Any]) -> None:
    sdfg.expand_rocm()
    sdfg.expand()


def _parse_docc_debug() -> dict[str, str]:
    debug_env = os.environ.get("DOCC_DEBUG", "")
    debug_dict = {}
    if debug_env:
        for entry in re.split(r"[;:]", debug_env):
            if not entry:
                continue
            parts = entry.split("=", 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            debug_dict[key] = value
    return debug_dict


def _is_debug_dump(flags: dict[str, str]) -> bool:
    return "dump" in flags


def _is_debug_compile(flags: dict[str, str]) -> bool:
    return "build" in flags


def _get_build_thread_count(flags: dict[str, str]) -> int:
    return int(flags.get("build_threads", "0"))


class DoccProgram(ABC):

    def __init__(
        self,
        name: str,
        target: str = "none",
        category: str = "server",
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
        remote_tuning: bool = False,
    ):
        self.name = name
        self.target = target
        self.category = category
        self.remote_tuning = remote_tuning
        self.last_sdfg: Optional[StructuredSDFG] = None
        self.cache: dict = {}
        debug_flags = _parse_docc_debug()
        self.debug_dump: bool = _is_debug_dump(debug_flags)
        self.debug_build: bool = _is_debug_compile(debug_flags)
        self.build_thread_count: int = _get_build_thread_count(debug_flags)

        # Check environment variable DOCC_CI
        docc_ci = os.environ.get("DOCC_CI", "")
        if docc_ci:
            if docc_ci == "regions":
                if instrumentation_mode is None:
                    instrumentation_mode = "ols"
            elif docc_ci == "arg-capture":
                if capture_args is None:
                    capture_args = True
            else:
                # Full mode (or unknown value treated as full)
                if instrumentation_mode is None:
                    instrumentation_mode = "ols"
                if capture_args is None:
                    capture_args = True

        self.instrumentation_mode = instrumentation_mode
        self.capture_args = capture_args

        if target == "cuda":
            from docc.python import register_target_overrides

            register_target_overrides(
                "cuda",
                schedule_fn=None,
                compile_fn=None,
                expand_fn=_cuda_expand_fn,
            )
        elif target == "rocm":
            from docc.python import register_target_overrides

            register_target_overrides(
                "rocm",
                schedule_fn=None,
                compile_fn=None,
                expand_fn=_rocm_expand_fn,
            )

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def compile(self, *args: Any, output_folder: Optional[str] = None) -> CompiledSDFG:
        pass

    def sdfg_pipe(
        self,
        sdfg: StructuredSDFG,
        output_folder: Optional[str],
        instrumentation_mode: str,
        capture_args: bool,
    ) -> str:

        if self.debug_dump:
            sdfg.dump(output_folder, "py0.parsed", dump_dot=True)

        # Enable statistics if envvar is set
        if _statistics_enabled_by_env():
            _enable_statistics()

        sdfg.validate()

        # Tensor targets keep tensor nodes
        custom_expand_fn = get_target_expand_fn(self.target)
        if custom_expand_fn is not None:
            custom_expand_fn(sdfg, self.category, {})
        else:
            sdfg.expand()
        if self.debug_dump:
            sdfg.dump(output_folder, "py1.expanded", dump_dot=True)

        # Simplify pipelines
        sdfg.simplify()
        if self.debug_dump:
            sdfg.dump(output_folder, "py2.opt", dump_dot=True)

        # Normalization for scheduling
        if self.target != "none":
            sdfg.normalize()

        if self.debug_dump or instrumentation_mode or capture_args:
            sdfg.dump(
                output_folder,
                "py3.norm",
                dump_dot=self.debug_dump,
                dump_json=True,
                record_for_instrumentation=True,
            )

        # Schedule if target is specified

        if self.target != "none":
            # Check for custom registered target first
            custom_schedule_fn = get_target_schedule_fn(self.target)
            if custom_schedule_fn is not None:
                custom_schedule_fn(
                    sdfg, self.category, {"remote_tuning": self.remote_tuning}
                )
            else:
                sdfg.schedule(self.target, self.category, self.remote_tuning)

            if self.debug_dump:
                sdfg.dump(output_folder, "py4.post_sched", dump_dot=True)

        self.last_sdfg = sdfg

        custom_compile_fn = get_target_compile_fn(self.target)
        if custom_compile_fn is not None:
            lib_path = custom_compile_fn(
                sdfg,
                output_folder,
                instrumentation_mode,
                capture_args,
                {"debug_build": self.debug_build, "threads": self.build_thread_count},
            )
        else:
            lib_path = sdfg._compile(
                output_folder=output_folder,
                target=self.target,
                instrumentation_mode=instrumentation_mode,
                capture_args=capture_args,
                debug_build=self.debug_build,
                threads=self.build_thread_count,
            )

        # Dump statistics after compile
        if _statistics_enabled_by_env():
            print(_statistics_summary(), file=sys.stderr)

        return lib_path

    @abstractmethod
    def to_sdfg(self, *args: Any) -> StructuredSDFG:
        pass

    @abstractmethod
    def _convert_inputs(self, args: tuple) -> tuple:
        pass

    @abstractmethod
    def _convert_outputs(self, result: Any, original_args: tuple) -> Any:
        pass

    def _get_cache_key(self, *args: Any) -> str:
        return ""
