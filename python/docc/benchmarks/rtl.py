"""Read, validate and query daisy RTL instrumentation traces.

The RTL library (``rtl/src/instrumentation.cpp``) emits a Chrome-trace subset
extended with docc-specific metadata. This module loads such a trace, validates
it against the bundled JSON schema (``daisy_trace.schema.json``) and exposes a
typed, queryable :class:`Trace` object.

Two trace shapes exist, selected at capture time by ``__DAISY_INSTRUMENTATION_MODE``:

* per-invocation (``cat == "region,daisy"``) — one event per region entry/exit,
  with ``metrics`` a flat ``{counter: int}`` map.
* aggregated (``cat == "aggregated_region,daisy"``) — one event per region with
  Welford statistics per counter plus a ``runtime`` stat block.

Example::

    from docc.benchmarks import Trace

    trace = Trace.load("daisy_trace.json")
    for region in trace.regions:
        print(region.name, region.runtime_mean_us)

    hot = trace.filter(target_type="CUDA")
"""

from __future__ import annotations

import copy
import ctypes
import functools
import json
import os
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

__all__ = [
    "Trace",
    "TraceRegion",
    "LoopInfo",
    "SourcePos",
    "SourceRange",
    "MetricStat",
    "TraceValidationError",
    "validate",
    "PER_INVOCATION_CAT",
    "AGGREGATED_CAT",
    "STATIC_PREFIX",
    "reset_instrumentation",
    "total_stats",
    "RtlTotalStats",
]

PER_INVOCATION_CAT = "region,daisy"
AGGREGATED_CAT = "aggregated_region,daisy"
STATIC_PREFIX = "static:::"
_RUNTIME_KEY = "runtime"

_SCHEMA_FILENAME = "daisy_trace.schema.json"


# --- Runtime control of the instrumentation RTL -----------------------------
#
# libdaisy_rtl is now linked as dynamic lib, so it defaults to shared global state per process.

_RTL_RESET_SYMBOL = "__daisy_instrumentation_reset_all"


def _rtl_lib_names() -> tuple[str, str]:
    """(exact filename, glob) for the RTL on the current platform."""
    if sys.platform == "darwin":
        return "libdaisy_rtl.dylib", "libdaisy_rtl*.dylib"
    if sys.platform.startswith("win"):
        return "daisy_rtl.dll", "daisy_rtl*.dll"
    # Match the plain name and, defensively, any soname-versioned file.
    return "libdaisy_rtl.so", "libdaisy_rtl.so*"


def _candidate_lib_dirs() -> List[Path]:
    dirs: List[Path] = []
    # Authoritative: the native driver's own library search paths, reconstructed
    # from the extension module's on-disk location (DefaultDoccPaths). This is the
    # exact set of directories the compiler uses to link against libdaisy_rtl.
    try:
        from docc.sdfg._sdfg import _default_library_paths  # noqa: PLC0415

        for p in _default_library_paths():
            dirs.append(Path(p))
    except Exception:
        pass

    # Only when the native resolver yields nothing do we fall back to the dynamic
    # linker's own search path (LD_LIBRARY_PATH / DYLD_LIBRARY_PATH). When we do
    # have authoritative paths, the loader would search these anyway, so adding
    # them here is redundant.
    if not dirs:
        env_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        for entry in os.environ.get(env_var, "").split(os.pathsep):
            if entry:
                dirs.append(Path(entry))

    # De-duplicate while preserving order.
    seen = set()
    unique: List[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _find_rtl_lib() -> Optional[Path]:
    """Locate the packaged shared RTL, or None when statically linked."""
    override = os.environ.get("DAISY_RTL_LIB")
    if override:
        p = Path(override)
        return p if p.is_file() else None

    exact, pattern = _rtl_lib_names()
    for d in _candidate_lib_dirs():
        if not d.is_dir():
            continue
        exact_path = d / exact
        if exact_path.is_file():
            return exact_path
        matches = sorted(d.glob(pattern))
        if matches:
            return matches[0]
    return None


@functools.lru_cache(maxsize=1)
def _rtl_lib() -> Optional[ctypes.CDLL]:
    """Load the shared RTL once; None if it is not a shared build."""
    path = _find_rtl_lib()
    if path is None:
        return None
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        return None


def reset_instrumentation() -> None:
    """Discard aggregated region stats so only post-reset runs are recorded.

    Harnesses call this after an untimed warmup. With a shared RTL there is one
    process-global instance, so a single reset covers every artifact; otherwise
    it falls back to resetting each statically-linked artifact individually.
    """
    lib = _rtl_lib()
    if lib is not None:
        try:
            # Subscript access bypasses ctypes' dunder-name guard.
            reset = lib[_RTL_RESET_SYMBOL]
        except (AttributeError, KeyError, ValueError):
            reset = None
        if reset is not None:
            reset.restype = None
            reset.argtypes = []
            reset()
            return


@dataclass(frozen=True)
class RtlTotalStats:
    """Aggregated runtime across all live regions in the shared RTL instance."""

    mean_us: float
    variance_us2: float
    #: Minimum invocation count over all regions (0 means some region is empty).
    count: int


def total_stats() -> Optional[RtlTotalStats]:
    """Live aggregate runtime stats from the shared RTL, or None.

    Returns None when the RTL is statically linked (no queryable shared instance)
    or when no region has recorded a run yet.
    """
    lib = _rtl_lib()
    if lib is None:
        return None
    try:
        fn = lib["__daisy_instrumentation_total_stats"]
    except (AttributeError, KeyError, ValueError):
        return None
    fn.restype = ctypes.c_bool
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_longlong),
    ]
    mean = ctypes.c_double(0.0)
    variance = ctypes.c_double(0.0)
    count = ctypes.c_longlong(0)
    if not fn(ctypes.byref(mean), ctypes.byref(variance), ctypes.byref(count)):
        return None
    return RtlTotalStats(
        mean_us=mean.value, variance_us2=variance.value, count=count.value
    )


def _find_schema() -> Path:
    """Locate the daisy-trace JSON schema.

    In an installed/built wheel the schema is shipped as package data next to this
    module (CMake installs ``rtl/schema/daisy_trace.schema.json`` into
    ``docc/benchmarks``). When running from a source checkout that install has not
    happened, so fall back to the canonical copy under ``rtl/schema``.
    """
    bundled = Path(__file__).with_name(_SCHEMA_FILENAME)
    if bundled.is_file():
        return bundled

    # rtl.py -> docc/python/docc/benchmarks/rtl.py ; schema -> docc/rtl/schema/...
    repo_schema = (
        Path(__file__).resolve().parents[3] / "rtl" / "schema" / _SCHEMA_FILENAME
    )
    if repo_schema.is_file():
        return repo_schema

    raise FileNotFoundError(
        f"Could not locate {_SCHEMA_FILENAME}; expected it next to {bundled} "
        f"(installed wheel) or at {repo_schema} (source checkout)."
    )


def load_schema() -> dict:
    """Return the bundled daisy-trace JSON schema as a dict."""
    with open(_find_schema(), "r") as f:
        return json.load(f)


class TraceValidationError(ValueError):
    """Raised when a trace does not conform to the daisy-trace schema."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("Trace failed schema validation:\n" + "\n".join(errors))


def validate(instance: Any, schema: Optional[dict] = None) -> List[str]:
    """Validate a parsed trace object against the schema.

    Returns a list of human-readable error strings (empty when valid). Requires
    the optional ``jsonschema`` dependency.
    """
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - exercised only without dep
        raise ImportError(
            "Validating daisy traces requires the 'jsonschema' package. "
            "Install it with `pip install jsonschema`."
        ) from exc

    if schema is None:
        schema = load_schema()

    validator = Draft202012Validator(schema)
    errors: List[str] = []
    for error in sorted(
        validator.iter_errors(instance), key=lambda e: list(e.absolute_path)
    ):
        location = "/".join(str(p) for p in error.absolute_path) or "<root>"
        errors.append(f"{location}: {error.message}")
    return errors


@dataclass(frozen=True)
class SourcePos:
    line: int
    col: int

    @classmethod
    def from_dict(cls, d: dict) -> "SourcePos":
        return cls(line=d["line"], col=d["col"])


@dataclass(frozen=True)
class SourceRange:
    file: str
    begin: SourcePos
    end: SourcePos

    @classmethod
    def from_dict(cls, d: dict) -> "SourceRange":
        return cls(
            file=d["file"],
            begin=SourcePos.from_dict(d["from"]),
            end=SourcePos.from_dict(d["to"]),
        )


@dataclass(frozen=True)
class LoopInfo:
    loopnest_index: int
    num_loops: int
    num_maps: int
    num_fors: int
    num_whiles: int
    max_depth: int
    is_perfectly_nested: bool
    is_perfectly_parallel: bool
    is_elementwise: bool
    has_side_effects: bool

    @classmethod
    def from_dict(cls, d: dict) -> "LoopInfo":
        return cls(
            loopnest_index=d["loopnest_index"],
            num_loops=d["num_loops"],
            num_maps=d["num_maps"],
            num_fors=d["num_fors"],
            num_whiles=d["num_whiles"],
            max_depth=d["max_depth"],
            is_perfectly_nested=d["is_perfectly_nested"],
            is_perfectly_parallel=d["is_perfectly_parallel"],
            is_elementwise=d["is_elementwise"],
            has_side_effects=d["has_side_effects"],
        )


@dataclass(frozen=True)
class MetricStat:
    """Aggregated statistics for a single counter (aggregated traces).

    ``mean``/``variance``/``min``/``max`` may be ``None`` for provided
    (``static:::``) metrics whose value was non-finite at capture time.
    """

    mean: Optional[float]
    variance: Optional[float]
    count: int
    min: Optional[float]
    max: Optional[float]

    @classmethod
    def from_dict(cls, d: dict) -> "MetricStat":
        return cls(
            mean=d["mean"],
            variance=d["variance"],
            count=d["count"],
            min=d["min"],
            max=d["max"],
        )


class TraceRegion:
    """A single region event, wrapping the raw trace dict with typed accessors."""

    def __init__(self, raw: dict):
        self._raw = raw

    # -- Raw / identity ----------------------------------------------------
    @property
    def raw(self) -> dict:
        return self._raw

    @property
    def category(self) -> str:
        return self._raw["cat"]

    @property
    def is_aggregated(self) -> bool:
        return self.category == AGGREGATED_CAT

    @property
    def name(self) -> str:
        return self._raw["name"]

    @property
    def pid(self) -> int:
        return self._raw["pid"]

    @property
    def tid(self) -> int:
        return self._raw["tid"]

    @property
    def ts_us(self) -> Optional[float]:
        return self._raw.get("ts")

    @property
    def dur_us(self) -> Optional[float]:
        return self._raw.get("dur")

    # -- Source metadata ---------------------------------------------------
    @property
    def _args(self) -> dict:
        return self._raw["args"]

    @property
    def function(self) -> str:
        return self._args["function"]

    @property
    def module(self) -> str:
        return self._args["module"]

    @property
    def target_type(self) -> str:
        return self._args.get("target_type", "")

    @cached_property
    def source_ranges(self) -> List[SourceRange]:
        return [SourceRange.from_dict(r) for r in self._args.get("source_ranges", [])]

    # -- docc metadata -----------------------------------------------------
    @property
    def _docc(self) -> dict:
        return self._args["docc"]

    @property
    def sdfg_name(self) -> str:
        return self._docc.get("sdfg_name", "")

    @property
    def sdfg_file(self) -> str:
        return self._docc.get("sdfg_file", "")

    @property
    def arg_capture_path(self) -> str:
        return self._docc.get("arg_capture_path", "")

    @property
    def features_file(self) -> str:
        return self._docc.get("features_file", "")

    @property
    def opt_report_file(self) -> str:
        return self._docc.get("opt_report_file", "")

    @property
    def has_element(self) -> bool:
        """True when SDFG element metadata (id/type/loop_info) is present."""
        return "element_id" in self._docc

    @property
    def element_id(self) -> Optional[int]:
        return self._docc.get("element_id")

    @property
    def element_type(self) -> Optional[str]:
        return self._docc.get("element_type")

    @property
    def loopnest_index(self) -> Optional[int]:
        return self._docc.get("loopnest_index")

    @cached_property
    def loop_info(self) -> Optional[LoopInfo]:
        li = self._docc.get("loop_info")
        return LoopInfo.from_dict(li) if li is not None else None

    # -- Metrics -----------------------------------------------------------
    @property
    def _metrics(self) -> dict:
        return self._args.get("metrics", {})

    @cached_property
    def counters(self) -> Dict[str, int]:
        """PAPI counter values for a per-invocation region (empty if aggregated)."""
        if self.is_aggregated:
            return {}
        return dict(self._metrics)

    @cached_property
    def counter_stats(self) -> Dict[str, MetricStat]:
        """Aggregated PAPI counter statistics (empty for per-invocation regions).

        Excludes the reserved ``runtime`` block and ``static:::`` provided metrics.
        """
        if not self.is_aggregated:
            return {}
        out: Dict[str, MetricStat] = {}
        for key, val in self._metrics.items():
            if key == _RUNTIME_KEY or key.startswith(STATIC_PREFIX):
                continue
            out[key] = MetricStat.from_dict(val)
        return out

    @cached_property
    def static_metrics(self) -> Dict[str, MetricStat]:
        """User-provided (``static:::``) metric statistics, keyed without the prefix."""
        if not self.is_aggregated:
            return {}
        out: Dict[str, MetricStat] = {}
        for key, val in self._metrics.items():
            if key.startswith(STATIC_PREFIX):
                out[key[len(STATIC_PREFIX) :]] = MetricStat.from_dict(val)
        return out

    @cached_property
    def runtime(self) -> Optional[MetricStat]:
        """Aggregated runtime statistics in microseconds (aggregated traces only)."""
        rt = self._metrics.get(_RUNTIME_KEY) if self.is_aggregated else None
        return MetricStat.from_dict(rt) if rt is not None else None

    @property
    def runtime_mean_us(self) -> Optional[float]:
        """Mean runtime in microseconds.

        For aggregated traces this is ``runtime.mean``; for per-invocation traces
        it is this event's own ``dur``.
        """
        if self.is_aggregated:
            return self.runtime.mean if self.runtime is not None else None
        return self.dur_us

    @property
    def runtime_min_us(self) -> Optional[float]:
        """Best (minimum) runtime in microseconds.

        The min over the aggregated samples is the steady-state estimate.
        """
        if self.is_aggregated:
            return self.runtime.min if self.runtime is not None else None
        return self.dur_us

    def metric(self, name: str) -> Optional[Union[int, float]]:
        """Return a representative value for ``name``.

        For per-invocation regions this is the raw counter value; for aggregated
        regions it is the counter/static metric mean. Returns ``None`` if absent.
        """
        if not self.is_aggregated:
            return self.counters.get(name)
        if name in self.counter_stats:
            return self.counter_stats[name].mean
        if name in self.static_metrics:
            return self.static_metrics[name].mean
        return None

    def __repr__(self) -> str:
        return (
            f"TraceRegion(name={self.name!r}, module={self.module!r}, "
            f"target={self.target_type!r}, aggregated={self.is_aggregated})"
        )


class Trace:
    """A parsed, validated daisy trace: a collection of :class:`TraceRegion`."""

    def __init__(self, raw: dict):
        self._raw = raw
        self._regions = [TraceRegion(e) for e in raw.get("traceEvents", [])]

    # -- Construction ------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict, validate_schema: bool = True) -> "Trace":
        """Build a :class:`Trace` from an already-parsed dict.

        Raises :class:`TraceValidationError` if ``validate_schema`` is set and the
        data does not conform to the schema.
        """
        if validate_schema:
            errors = validate(data)
            if errors:
                raise TraceValidationError(errors)
        return cls(data)

    @classmethod
    def load(cls, path: Union[str, Path], validate_schema: bool = True) -> "Trace":
        """Read a trace JSON file, optionally validate it, and return a :class:`Trace`."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, validate_schema=validate_schema)

    def save(self, path: Union[str, Path], indent: Optional[int] = None) -> None:
        """Write this trace back to a JSON file (the daisy-trace format)."""
        with open(path, "w") as f:
            json.dump(self._raw, f, indent=indent)

    # -- Container protocol ------------------------------------------------
    @property
    def regions(self) -> List[TraceRegion]:
        return self._regions

    @property
    def raw(self) -> dict:
        return self._raw

    def __len__(self) -> int:
        return len(self._regions)

    def __iter__(self) -> Iterator[TraceRegion]:
        return iter(self._regions)

    def __getitem__(self, index: int) -> TraceRegion:
        return self._regions[index]

    @property
    def is_aggregated(self) -> bool:
        """True if the trace was captured in aggregate mode.

        Determined from the first region; a valid trace never mixes modes.
        """
        return bool(self._regions) and self._regions[0].is_aggregated

    # -- Queries -----------------------------------------------------------
    def filter(
        self,
        *,
        module: Optional[str] = None,
        function: Optional[str] = None,
        target_type: Optional[str] = None,
        element_type: Optional[str] = None,
        sdfg_name: Optional[str] = None,
    ) -> List[TraceRegion]:
        """Return regions matching all provided (non-``None``) criteria."""
        out = []
        for r in self._regions:
            if module is not None and r.module != module:
                continue
            if function is not None and r.function != function:
                continue
            if target_type is not None and r.target_type != target_type:
                continue
            if element_type is not None and r.element_type != element_type:
                continue
            if sdfg_name is not None and r.sdfg_name != sdfg_name:
                continue
            out.append(r)
        return out

    def by_element_id(self, element_id: int) -> List[TraceRegion]:
        """Return regions carrying the given SDFG ``element_id``."""
        return [r for r in self._regions if r.element_id == element_id]

    def hottest(self, n: Optional[int] = None) -> List[TraceRegion]:
        """Regions sorted by mean runtime (descending); ``n`` limits the count."""
        ranked = sorted(
            (r for r in self._regions if r.runtime_mean_us is not None),
            key=lambda r: r.runtime_mean_us,
            reverse=True,
        )
        return ranked[:n] if n is not None else ranked

    def total_runtime_us(self) -> float:
        """Sum of mean runtimes across all regions (microseconds)."""
        return sum((r.runtime_mean_us or 0.0) for r in self._regions)

    # -- Combining ---------------------------------------------------------
    @staticmethod
    def _region_key(region: "TraceRegion") -> tuple:
        """Stable identity of a region across separate runs of the same workload.

        Matches on source/docc identity and excludes pid/tid/timestamps, which
        differ between runs.
        """
        return (
            region.name,
            region.function,
            region.module,
            region.sdfg_name,
            region.element_id,
        )

    @classmethod
    def combine(
        cls, traces: Sequence["Trace"], validate_schema: bool = False
    ) -> "Trace":
        """Combine multiple traces of the same workload into one.

        Intended for counter multiplexing: when the hardware can only measure a
        subset of counters at once, the workload is run once per counter group,
        producing several traces that each carry a slice of the ``metrics``. This
        takes the first trace as the reference for regions and timesteps (name,
        ts, dur, source/docc metadata, runtime) and unions the per-region
        ``metrics`` maps from the remaining traces into it.

        Regions are matched across traces by :meth:`_region_key` (source/docc
        identity, ignoring pid/tid/timestamps). On a metric-key collision the
        reference wins, so its ``runtime`` and any ``static:::`` values are kept.
        Returns a new :class:`Trace`; the inputs are not mutated.
        """
        traces = list(traces)
        if not traces:
            raise ValueError("Trace.combine() requires at least one trace")
        if len(traces) == 1:
            return cls.from_dict(
                copy.deepcopy(traces[0].raw), validate_schema=validate_schema
            )

        # Index each additional trace's per-region metrics by region key.
        extra_metrics: Dict[tuple, List[dict]] = {}
        for other in traces[1:]:
            for region in other.regions:
                metrics = region.raw.get("args", {}).get("metrics", {})
                extra_metrics.setdefault(cls._region_key(region), []).append(metrics)

        # Deep-copy the reference and union in the other traces' metrics.
        combined_raw = copy.deepcopy(traces[0].raw)
        for event in combined_raw.get("traceEvents", []):
            key = cls._region_key(TraceRegion(event))
            metrics = event.setdefault("args", {}).setdefault("metrics", {})
            for other_metrics in extra_metrics.get(key, []):
                for name, value in other_metrics.items():
                    metrics.setdefault(name, value)

        return cls.from_dict(combined_raw, validate_schema=validate_schema)
