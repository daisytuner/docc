from abc import ABC, abstractmethod
from typing import Any, Optional
import os

from docc.sdfg import StructuredSDFG
from docc.compiler.compiled_sdfg import CompiledSDFG
from docc.compiler.target_registry import (
    get_target_schedule_fn,
    get_target_compile_fn,
    get_target_expand_fn,
)


def _is_debug_dump() -> bool:
    return bool(os.environ.get("DOCC_DEBUG"))


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
        self.debug_dump: bool = _is_debug_dump()

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

    def sdfg_pipe(
        self,
        sdfg: StructuredSDFG,
        output_folder: Optional[str],
        instrumentation_mode: str,
        capture_args: bool,
    ) -> str:

        if self.debug_dump:
            sdfg.dump(output_folder, "py0.parsed", dump_dot=True)
        sdfg.validate()

        # Tensor targets keep tensor nodes
        if self.target != "onnx":
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
                sdfg, output_folder, instrumentation_mode, capture_args, {}
            )
        else:
            lib_path = sdfg._compile(
                output_folder=output_folder,
                target=self.target,
                instrumentation_mode=instrumentation_mode,
                capture_args=capture_args,
            )

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
