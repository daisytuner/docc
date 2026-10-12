from __future__ import annotations

import sys
import json
import os
import re
import time
import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, get_args, get_type_hints

from docc.sdfg import (
    StructuredSDFG,
    TargetOptions,
    DoccMetrics,
    registered_options,
    target_default_contraction_precision,
)
from docc.sdfg._sdfg import (
    _enable_statistics,
    _statistics_mode_by_env,
    _statistics_report,
)
from docc.compiler.compiled_sdfg import CompiledSDFG
from docc.compiler.codegens import (
    build_artifact,
    get_target_codegens,
    get_target_default_codegen,
)


def _pass_option_specs() -> Dict[str, dict]:
    """Registered pass options keyed by full key (empty if unavailable)."""
    # Not cached: target plugins register their options when they are loaded.
    return {spec["key"]: spec for spec in registered_options()}


def _coerce_pass_option(spec: dict, value: Any) -> Any:
    """Coerce a value to a registered pass option's declared type."""
    kind = spec["type"]
    if kind == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in ("1", "true", "yes", "on"):
                return True
            if v in ("0", "false", "no", "off", ""):
                return False
        elif isinstance(value, (int, float)):
            return bool(value)
        raise TypeError(f"pass option {spec['key']!r} expects bool, got {value!r}")
    if kind == "int":
        return int(value)
    if kind == "double":
        return float(value)
    return str(value)


# Mirrors fp_math::ContractionPrecision::parse.
_CONTRACTION_PRECISION = re.compile(r"native|int[2-8]|ozaki|ozaki[12](:[1-9][0-9]*)?")


@dataclass
class DoccOptions:
    """Compile options for a :class:`DoccProgram`.

    Precedence (lowest to highest): field default < environment variable
    (``DOCC_CI``, ``DOCC_DEBUG``, ``DOCC_CODEGEN``) < explicit constructor argument.

    Env-settable fields default to ``None`` ("unset") so an explicit value is
    distinguishable from the default and wins over the environment. Everything
    is resolved once in :meth:`__post_init__`.
    """

    # Optimization
    target: Optional[str] = None
    category: Optional[str] = None
    remote_tuning: Optional[bool] = None

    # Code generation (settable via DOCC_CODEGEN): "cpp" builds a shared
    # library, "numpy" a Python module. Must be supported by the target.
    codegen: Optional[str] = None

    # Debug (settable via DOCC_DEBUG)
    debug_dump: Optional[bool] = None
    debug_build: Optional[bool] = None
    build_thread_count: Optional[int] = None

    # Instrumentation (settable via DOCC_CI)
    instrumentation_mode: Optional[str] = None
    capture_args: Optional[bool] = None

    # Reuse (settable via DOCC_REUSE_BINARIES / DOCC_REUSE_SOURCES)
    reuse_binaries: Optional[bool] = None
    reuse_sources: Optional[bool] = None

    # Compiler passes
    einsum: Optional[bool] = None
    normalize: Optional[bool] = None
    device_residency: Optional[bool] = None
    # Fast, approximate device math intrinsics (e.g. __expf). Unset resolves to the
    # frontend default (DoccProgram.default_approximate_math).
    approximate_math: Optional[bool] = None
    # Mirrors -ffp-contract=fast: contract a multiplication feeding an addition
    # into a fused multiply-add during simplify. Defaults to True.
    fp_contract: Optional[bool] = None
    # How Float and Double matmuls compute: "native" (the default of most targets); "int8" (or int2 ... int7)
    # quantizes them with a scale per row and column, losing accuracy; "ozaki", "ozaki1[:s]"
    # (slices) and "ozaki2[:L]" (moduli) emulate them on exact Int8 matmuls at the native
    # precision. Pays off only where Int8 matrix units are much faster than the native type. Unset takes
    # the target's default (DoccTarget::default_contraction_precision).
    contraction_precision: Optional[str] = None

    # Frontend: parse array shapes as symbolic sizes (``_s{i}`` arguments) when
    # True, or bake in the concrete integer sizes from the example inputs when
    # False.
    symbolic_shapes: Optional[bool] = None

    # Overrides for registered options ({full_key: value}); see
    # registered_options(). Forwarded to the SDFG in sdfg_pipe.
    pass_options: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Resolve default values for all unset values."""

        # Environment fills only still-unset fields, so explicit args win over env.
        self._resolve_docc_ci()
        self._resolve_docc_debug()
        self._resolve_docc_reuse()

        # Optimization
        if self.target is None:
            self.target = "none"
        if self.category is None:
            self.category = "server"
        if self.remote_tuning is None:
            self.remote_tuning = False

        # Code generation
        if self.codegen is None:
            self.codegen = os.environ.get("DOCC_CODEGEN") or get_target_default_codegen(
                self.target
            )

        # Debug
        if self.debug_dump is None:
            self.debug_dump = False
        if self.debug_build is None:
            self.debug_build = False
        if self.build_thread_count is None:
            self.build_thread_count = 0

        # Instrumentation
        if self.instrumentation_mode is None:
            self.instrumentation_mode = ""
        if self.capture_args is None:
            self.capture_args = False

        # Reuse
        if self.reuse_binaries is None:
            self.reuse_binaries = False
        if self.reuse_sources is None:
            self.reuse_sources = False

        # Reuse reloads the persisted post-schedule SDFG (py5.post_sched.json),
        # so the build that produces the cache must dump it.
        if self.reuse_binaries or self.reuse_sources:
            self.debug_dump = True

        # Compiler Passes
        if self.normalize is None:
            self.normalize = self.target in ("sequential", "openmp")
        if self.device_residency is None:
            self.device_residency = self.target in ("cuda", "rocm")
        if self.symbolic_shapes is None:
            self.symbolic_shapes = True
        if self.fp_contract is None:
            self.fp_contract = True
        if self.contraction_precision is None:
            self.contraction_precision = target_default_contraction_precision(
                self.target
            )
        if not _CONTRACTION_PRECISION.fullmatch(self.contraction_precision):
            raise ValueError(
                f"Invalid contraction_precision {self.contraction_precision!r}: expected native, "
                "int2 ... int8, ozaki, ozaki1[:s] or ozaki2[:L]"
            )

        if self.pass_options is None:
            self.pass_options = {}

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> DoccOptions:
        """Build from arbitrary user kwargs.

        Field kwargs are coerced to their declared type. Keys matching a
        registered option (see ``registered_options()``) are routed
        into ``pass_options`` and coerced to the option's type. Anything else is
        dropped with a warning. Use at boundaries that accept free-form options
        (the ``@native`` decorator, the torch.compile backend); direct
        construction stays strict.
        """
        known = {f.name for f in fields(cls)}
        pass_specs = _pass_option_specs()

        # Explicit pass_options dict plus any flat registered keys collected below.
        pass_opts: Dict[str, Any] = dict(kwargs.pop("pass_options", None) or {})
        field_kwargs: Dict[str, Any] = {}
        unknown = []
        for key, value in kwargs.items():
            if key in known:
                field_kwargs[key] = cls._coerce(key, value)
            elif key in pass_specs:
                pass_opts[key] = value
            else:
                unknown.append(key)
        if unknown:
            warnings.warn(
                f"Ignoring unknown compile options: {', '.join(sorted(unknown))}"
            )

        resolved: Dict[str, Any] = {}
        for key, value in pass_opts.items():
            spec = pass_specs.get(key)
            if spec is None:
                warnings.warn(f"Ignoring unknown pass option: {key}")
                continue
            resolved[key] = _coerce_pass_option(spec, value)
        field_kwargs["pass_options"] = resolved

        return cls(**field_kwargs)

    @classmethod
    def _field_types(cls) -> Dict[str, type]:
        """Concrete (non-``None``) declared type for each field."""
        hints = get_type_hints(cls)
        types: Dict[str, type] = {}
        for f in fields(cls):
            non_none = [a for a in get_args(hints[f.name]) if a is not type(None)]
            types[f.name] = non_none[0] if non_none else hints[f.name]
        return types

    @classmethod
    def _coerce(cls, name: str, value: Any) -> Any:
        """Validate/convert ``value`` to the declared type of field ``name``."""
        if value is None:
            return None
        target = cls._field_types()[name]
        try:
            if target is bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    v = value.strip().lower()
                    if v in ("1", "true", "yes", "on"):
                        return True
                    if v in ("0", "false", "no", "off", ""):
                        return False
                elif isinstance(value, (int, float)):
                    return bool(value)
                raise ValueError(f"cannot interpret {value!r} as a bool")
            # bool is a subclass of int; don't silently accept True/False as an int.
            if isinstance(value, target) and not (
                target is int and isinstance(value, bool)
            ):
                return value
            return target(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"compile option {name!r} expects {target.__name__}, got {value!r}"
            ) from exc

    def _resolve_docc_ci(self) -> None:
        docc_ci = os.environ.get("DOCC_CI", "")
        if not docc_ci:
            return

        # Fill only unset knobs; an explicit value overrides the environment.
        if self.instrumentation_mode is None and docc_ci != "arg-capture":
            self.instrumentation_mode = "ols"
        if self.capture_args is None and docc_ci != "regions":
            self.capture_args = True

    def _resolve_docc_debug(self) -> None:
        debug_env = os.environ.get("DOCC_DEBUG", "")
        if not debug_env:
            return

        debug_flags = {}
        for entry in re.split(r"[;:]", debug_env):
            if not entry:
                continue
            parts = entry.split("=", 1)
            key = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            debug_flags[key] = value

        if self.debug_dump is None and "dump" in debug_flags:
            self.debug_dump = True
        if self.debug_build is None and "build" in debug_flags:
            self.debug_build = True
        if self.build_thread_count is None and "build_threads" in debug_flags:
            self.build_thread_count = self._coerce(
                "build_thread_count", debug_flags["build_threads"]
            )

    def _resolve_docc_reuse(self) -> None:
        # Proper boolean flags: yes/1/true enable, no/0/false (or empty) disable.
        for field, env_var in (
            ("reuse_binaries", "DOCC_REUSE_BINARIES"),
            ("reuse_sources", "DOCC_REUSE_SOURCES"),
        ):
            raw = os.environ.get(env_var)
            if raw is not None and getattr(self, field) is None:
                setattr(self, field, self._coerce(field, raw))


class DoccProgram(ABC):

    default_approximate_math: bool = False

    def __init__(
        self,
        name: str,
        options: DoccOptions,
    ):
        self.name = name
        # Options are fully resolved at construction (DoccOptions.__post_init__).
        self.options = options
        if self.options.approximate_math is None:
            self.options.approximate_math = self.default_approximate_math

        self.last_sdfg: Optional[StructuredSDFG] = None
        self.cache: dict = {}

        self._device_resident: bool = False
        self._device_backend: Optional[str] = None

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
        reuse_sources: bool = False,
        metrics: Optional[DoccMetrics] = None,
    ) -> str:

        start_time = time.perf_counter()

        if self.options.codegen not in get_target_codegens(self.options.target):
            raise ValueError(
                f"Target '{self.options.target}' does not support codegen "
                f"'{self.options.codegen}'"
            )

        if not reuse_sources and output_folder:
            if self.options.debug_dump:
                sdfg.dump(output_folder, "py0.parsed", dump_dot=True)

            if not output_folder is None:
                sdfg.output_dir = output_folder

            # Enable statistics if envvar is set
            stats_mode = _statistics_mode_by_env()
            if stats_mode > 0:
                _enable_statistics()

            sdfg.validate()

            # Forward user-set pass options before running any pass.
            for key, value in self.options.pass_options.items():
                sdfg.set_option(key, value)

            target_options = TargetOptions()
            target_options.target = self.options.target
            target_options.category = self.options.category
            target_options.remote_tuning = self.options.remote_tuning
            target_options.fp_contract = self.options.fp_contract
            target_options.contraction_precision = self.options.contraction_precision
            target_options.options = sdfg.options
            metrics.add_target_options(target_options)

            # Einsum detection
            if self.options.einsum:
                sdfg.einsum()
                if self.options.debug_dump:
                    sdfg.dump(output_folder, "py1.einsum", dump_dot=True)

            # Tensor targets keep tensor nodes
            sdfg.expand(target_options)
            if self.options.debug_dump:
                sdfg.dump(output_folder, "py2.expanded", dump_dot=True)

            # Simplify pipelines
            sdfg.simplify(target_options)
            if self.options.debug_dump:
                sdfg.dump(output_folder, "py3.opt", dump_dot=True)

            # Normalization for scheduling
            if self.options.normalize:
                sdfg.normalize()
                target_options.already_normalized = True
            if self.options.debug_dump:
                sdfg.dump(
                    output_folder,
                    "py4.norm",
                    dump_dot=True,
                )

            # Schedule if target is specified
            sdfg.schedule(
                target_options,
                not self.options.capture_args,
            )

            # CMath nodes exist only after expansion.
            if self.options.approximate_math:
                sdfg.approximate_math()

            # Promote pointer arguments to device residency when the whole program keeps
            # data on device. Communicated explicitly via the pass return value (bool),
            # not through SDFG metadata.
            self._device_resident = False
            self._device_backend = None
            if self.options.device_residency:
                self._device_resident = sdfg.promote_device_residency(
                    self.options.target == "rocm"
                )

            sdfg.add_metadata("device_resident", "1" if self._device_resident else "0")
            if self._device_resident:
                self._device_backend = self.options.target
                sdfg.add_metadata("device_backend", self.options.target)

            if (
                self.options.debug_dump
                or self.options.instrumentation_mode
                or self.options.capture_args
            ):
                sdfg.dump(
                    output_folder,
                    "py5.post_sched",
                    dump_dot=True,
                    dump_json=True,
                    record_for_instrumentation=True,
                )
        else:
            self._device_resident = sdfg.metadata("device_resident") == "1"
            backend = sdfg.metadata("device_backend")
            self._device_backend = backend or None

        self.last_sdfg = sdfg

        compile_end_time = time.perf_counter()
        sdfg_opt_time = compile_end_time - start_time
        if metrics is not None:
            metrics.add_metric(
                "sdfg_compile_time_ms", round(sdfg_opt_time * 1000), "compile_times"
            )

        lib_path = sdfg._compile(
            output_folder=output_folder,
            target=self.options.target,
            instrumentation_mode=self.options.instrumentation_mode,
            capture_args=self.options.capture_args,
            debug_build=self.options.debug_build,
            threads=self.options.build_thread_count,
            reuse_sources=reuse_sources,
            codegen=self.options.codegen,
        )
        build_artifact(self.options.target, self.options.codegen, lib_path, sdfg.name)

        bin_build_time = time.perf_counter() - compile_end_time
        if metrics is not None:
            metrics.add_metric(
                "bin_build_time_ms", round(bin_build_time * 1000), "compile_times"
            )

        # Dump statistics after compile
        if stats_mode > 0:
            print(_statistics_report(stats_mode), file=sys.stderr)

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
