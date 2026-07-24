from abc import ABC, abstractmethod
import sys
from typing import Any, Dict, Optional
import json
import os
import re

from docc.sdfg import StructuredSDFG, TargetOptions
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
    register_target_overrides,
)


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
        self._device_resident: bool = False
        self._device_backend: Optional[str] = None
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

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def compile(self, *args: Any, output_folder: Optional[str] = None) -> CompiledSDFG:
        pass

    def _resolve_compile_options(
        self,
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
        remote_tuning: Optional[bool] = None,
    ) -> tuple[str, bool, bool]:
        """Resolve compile-time options, falling back to instance defaults and env vars."""
        if instrumentation_mode is None:
            instrumentation_mode = self.instrumentation_mode
        if capture_args is None:
            capture_args = self.capture_args
        if remote_tuning is None:
            remote_tuning = self.remote_tuning

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

        # Defaults
        if instrumentation_mode is None:
            instrumentation_mode = ""
        if capture_args is None:
            capture_args = False

        return instrumentation_mode, capture_args, remote_tuning

    def sdfg_pipe(
        self,
        sdfg: StructuredSDFG,
        output_folder: Optional[str],
        instrumentation_mode: str,
        capture_args: bool,
        remote_tuning: Optional[bool] = None,
        reuse_sources: bool = False,
    ) -> str:

        if not reuse_sources and output_folder:
            if self.debug_dump:
                sdfg.dump(output_folder, "py0.parsed", dump_dot=True)

            if not output_folder is None:
                sdfg.output_dir = output_folder

            # Enable statistics if envvar is set
            if _statistics_enabled_by_env():
                _enable_statistics()

            sdfg.validate()

            if remote_tuning is None:
                remote_tuning = self.remote_tuning

            target_options = TargetOptions()
            target_options.target = self.target
            target_options.category = self.category
            target_options.remote_tuning = remote_tuning

            # Einsum detection
            sdfg.einsum()
            if self.debug_dump:
                sdfg.dump(output_folder, "py1.einsum", dump_dot=True)

            # Tensor targets keep tensor nodes
            custom_expand_fn = get_target_expand_fn(self.target)
            if custom_expand_fn is not None:
                custom_expand_fn(sdfg, self.category, {})
            else:
                sdfg.expand(target_options)
            if self.debug_dump:
                sdfg.dump(output_folder, "py2.expanded", dump_dot=True)

            # Simplify pipelines
            sdfg.simplify()
            if self.debug_dump:
                sdfg.dump(output_folder, "py3.opt", dump_dot=True)

            # Normalization for scheduling
            if self.target != "none":
                sdfg.normalize()

            if self.debug_dump or instrumentation_mode or capture_args:
                sdfg.dump(
                    output_folder,
                    "py4.norm",
                    dump_dot=self.debug_dump,
                    dump_json=True,
                    record_for_instrumentation=True,
                )

            # Schedule if target is specified

            custom_schedule_fn = get_target_schedule_fn(self.target)
            if custom_schedule_fn is not None:
                custom_schedule_fn(
                    sdfg, self.category, {"remote_tuning": remote_tuning}
                )
            else:
                sdfg.schedule(target_options)

            if self.debug_dump:
                sdfg.dump(output_folder, "py5.post_sched", dump_dot=True)

        # Promote pointer arguments to device residency when the whole program keeps
        # data on device. Communicated explicitly via the pass return value (bool),
        # not through SDFG metadata.
        self._device_resident = False
        self._device_backend = None
        if self.target in ("cuda", "rocm"):
            if sdfg.promote_device_residency(self.target == "rocm"):
                self._device_resident = True
                self._device_backend = self.target

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
                reuse_sources=reuse_sources,
            )

        # Dump statistics after compile
        if _statistics_enabled_by_env():
            print(_statistics_summary(), file=sys.stderr)

        # Record the device-residency decision in the persisted (py4.norm) SDFG
        # metadata. It is computed here (not stored in metadata by the pass) and
        # decides host vs device argument marshalling. Binary-reuse paths
        # (DOCC_REUSE_BINARIES) load only the cached .so + normalized SDFG and
        # never re-run scheduling/promotion, so without this they default to
        # host execution and feed host pointers into a device-resident binary
        # -> heap corruption / double free.
        if output_folder:
            self._persist_device_residency(output_folder, sdfg)

        return lib_path

    def _persist_device_residency(
        self, output_folder: str, sdfg: StructuredSDFG
    ) -> None:
        """Stamp the device-residency decision into the persisted SDFG metadata.

        Patches only the ``metadata`` object of the already-written
        ``py4.norm.json`` (the file the reuse path loads), leaving the SDFG
        structure and element IDs untouched so instrumentation references stay
        valid.
        """
        json_path = os.path.join(output_folder, f"{sdfg.name}.py4.norm.json")
        try:
            with open(json_path) as f:
                data = json.load(f)
            metadata = data.setdefault("metadata", {})
            metadata["device_resident"] = "1" if self._device_resident else "0"
            metadata["device_backend"] = self._device_backend or ""
            with open(json_path, "w") as f:
                json.dump(data, f)
        except (OSError, ValueError):
            pass

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
