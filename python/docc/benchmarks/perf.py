"""Runtime measurement control for Daisy benchmark commands.

The launcher selects a measurement mode with ``DAISY_CI_MEASUREMENT_MODE``:

``automatic``
    No-op mode. The command may still wrap measured code in
    :meth:`PerfControl.measure`; the launcher measures the whole command.

``control``
    Perf-gated mode. When a benchmark process is launched under
    ``perf stat --control`` with perf started disabled (``-D -1``), the process
    can turn counting on and off at runtime. This lets the reported counters
    cover only a region of interest -- for example the steady-state loop --
    instead of the whole process, which is otherwise dominated by
    interpreter/library import, graph tracing, and one-time compilation/warmup
    costs.

``manual``
    Manual mode. The benchmark command writes measurement blocks to
    ``DAISY_CI_RUNTIME_OUTPUT``. The benchmark provides explicit metric values,
    including ``duration_time``, with :meth:`write_measurement` or
    :meth:`measure_callback`.

In ``control`` mode, the launcher wires up the control channel via environment
variables, either as fifo paths (``PERF_CTL_FIFO`` / ``PERF_ACK_FIFO``) or
inherited file descriptors (``PERF_CTL_FD`` / ``PERF_ACK_FD``). A matching perf
invocation looks like::

    ctl=/tmp/perf_ctl.fifo; ack=/tmp/perf_ack.fifo
    mkfifo "$ctl" "$ack"
    PERF_CTL_FIFO=$ctl PERF_ACK_FIFO=$ack \\
    perf stat -D -1 --control fifo:$ctl,$ack -e cycles,instructions -- \\
        python my_benchmark.py

Automatic mode usage::

    from docc.benchmarks.perf import PerfControl

    perf = PerfControl.from_env()

    warmup()
    with perf.measure():     # no-op in automatic mode; the launcher wraps all of this command
        for _ in range(n_runs):
            step()

Control mode usage::

    from docc.benchmarks.perf import PerfControl

    perf = PerfControl.from_env()

    warmup()                 # cold-start work, not counted
    with perf.measure():     # counters enabled only inside this block
        for _ in range(n_runs):
            step()

Manual mode usage::

    from docc.benchmarks.perf import PerfControl

    perf = PerfControl.from_env()

    for _ in range(perf.measurements or 3):
        metrics = run_one_measurement()
        perf.write_measurement({
            "duration_time": (metrics.runtime_ms, "ms"),
            "energy-pkg": (metrics.energy_joules, "Joules"),
            "custom_metric": metrics.custom_metric,
        })

Manual mode can also write several marker-separated measurements at once::

    perf.write_measurements([
        {"duration_time": (100, "ms"), "energy-pkg": (0.10, "Joules")},
        {"duration_time": (110, "ms"), "energy-pkg": (0.12, "Joules")},
        {"duration_time": (105, "ms"), "energy-pkg": (0.11, "Joules")},
    ])

Or manually with start / stop / resume::

    perf.start()
    ...
    perf.stop()              # pause
    perf.resume()            # continue
    ...
    perf.stop()

When no measurement environment is configured, every method is a no-op, so the
same code runs unchanged outside Daisy CI.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import IO, Any, Callable, Iterable, Iterator, Mapping, Optional

__all__ = ["PerfControl"]


class PerfControl:
    """Toggle ``perf stat`` counting around a region of interest.

    Instances are usually created with :meth:`from_env`. When the process was
    not launched under ``perf stat --control``, the returned instance is inert
    (:attr:`enabled` is ``False``) and every method does nothing.
    """

    def __init__(
        self,
        ctl_file: Optional[IO[str]] = None,
        ack_file: Optional[IO[str]] = None,
        *,
        mode: str = "automatic",
        output_path: Optional[str] = None,
        measurements: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self._ctl = ctl_file
        self._ack = ack_file
        self._mode = mode
        self._output_path = output_path
        self._measurements = measurements
        self._active = False
        if verbose and self._mode == "control" and self._ctl is not None:
            print(
                "perf control active: measuring only the selected region",
                flush=True,
            )
        elif verbose and self._mode == "manual" and self._output_path is not None:
            print(
                f"perf manual mode active: writing measurements to {self._output_path}",
                flush=True,
            )

    @classmethod
    def from_env(
        cls,
        env: Optional[Mapping[str, str]] = None,
        *,
        verbose: bool = True,
    ) -> "PerfControl":
        """Build a :class:`PerfControl` from the perf control environment.

        Reads ``PERF_CTL_FD`` / ``PERF_ACK_FD`` (inherited file descriptors) or,
        failing that, ``PERF_CTL_FIFO`` / ``PERF_ACK_FIFO`` (named-pipe paths).
        Returns an inert instance when neither is set or the channel cannot be
        opened, so callers never need to special-case the non-perf run.
        If channel setup and cleanup both fail, raise an ``ExceptionGroup``
        containing the setup error followed by each cleanup error.
        """
        # Determine mode from environment
        env = os.environ if env is None else env
        mode = env.get("DAISY_CI_MEASUREMENT_MODE", "").strip().lower()
        ack_fd = env.get("PERF_ACK_FD")
        ctl_fd = env.get("PERF_CTL_FD")
        ctl_fifo = env.get("PERF_CTL_FIFO")
        ack_fifo = env.get("PERF_ACK_FIFO")
        if not mode:
            mode = (
                "control" if ctl_fd is not None or ctl_fifo is not None else "automatic"
            )

        output_path = env.get("DAISY_CI_RUNTIME_OUTPUT")
        measurements = _parse_positive_int(env.get("DAISY_CI_MEASUREMENTS"))

        # PerfControl is a no-op in automatic mode
        if mode == "automatic":
            return cls(
                None,
                None,
                mode="automatic",
                output_path=output_path,
                measurements=measurements,
                verbose=False,
            )

        # PerfControl writes explicit values
        if mode == "manual":
            return cls(
                None,
                None,
                mode="manual",
                output_path=output_path,
                measurements=measurements,
                verbose=verbose,
            )

        # PerfControl starts and stops perf
        if mode != "control":
            print(
                f"unknown DAISY_CI_MEASUREMENT_MODE={mode!r}; measurement control disabled",
                file=sys.stderr,
            )
            return cls(None, None, mode="automatic", verbose=False)

        ctl_file: Optional[IO[str]] = None
        ack_file: Optional[IO[str]] = None
        try:
            if ctl_fd is not None:
                ctl_file = os.fdopen(int(ctl_fd), "w")
                if ack_fd is not None:
                    ack_file = os.fdopen(int(ack_fd), "r")
            elif ctl_fifo is not None:
                # perf opens the ctl fifo for reading and the ack fifo for
                # writing, so these opens do not block once perf is running.
                ctl_file = open(ctl_fifo, "w")
                if ack_fifo is not None:
                    ack_file = open(ack_fifo, "r")
        except (OSError, ValueError, TypeError, OverflowError) as setup_error:
            # Keep the setup error and any cleanup failures together.
            cleanup_errors: list[Exception] = []
            for handle in (ctl_file, ack_file):
                if handle is not None:
                    try:
                        handle.close()
                    except Exception as cleanup_error:
                        cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                raise ExceptionGroup(
                    "Perf control setup failed and cleanup also failed",
                    [setup_error, *cleanup_errors],
                ) from None
            print(
                f"perf control unavailable ({setup_error}); measurement not gated",
                file=sys.stderr,
            )
            return cls(None, None, mode="control", verbose=False)

        return cls(
            ctl_file,
            ack_file,
            mode="control",
            output_path=output_path,
            measurements=measurements,
            verbose=verbose,
        )

    @property
    def enabled(self) -> bool:
        """Whether this instance will actively affect measurement output."""
        return self._ctl is not None or self._mode == "manual"

    @property
    def mode(self) -> str:
        """Measurement mode: ``automatic``, ``control``, or ``manual``."""
        return self._mode

    @property
    def measurements(self) -> Optional[int]:
        """Requested number of measurements from ``DAISY_CI_MEASUREMENTS``."""
        return self._measurements

    @property
    def active(self) -> bool:
        """Whether counting is currently enabled."""
        return self._active

    def _send(self, command: str) -> None:
        if self._ctl is None:
            return
        self._ctl.write(command + "\n")
        self._ctl.flush()
        if self._ack is not None:
            # perf replies with "ack\n" once the command has been applied.
            self._ack.readline()

    def start(self) -> None:
        """Enable perf counting (begin the measured region)."""
        self._send("enable")
        self._active = True

    def resume(self) -> None:
        """Resume perf counting after :meth:`stop` (``continue``)."""
        self._send("enable")
        self._active = True

    def stop(self) -> None:
        """Disable perf counting (pause or end the measured region)."""
        self._send("disable")
        self._active = False

    def record_metric(self, name: str, value: Any, unit: str = "") -> None:
        """Append one manual-mode metric as its own measurement block."""
        self._write_manual_block([(name, unit, value)])

    def measure_callback(
        self,
        callback: Callable[[], Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Run ``callback`` and append the explicit metrics it returns."""
        metrics = callback()
        self.write_measurement(metrics=metrics)
        return metrics

    def write_measurement(
        self,
        metrics: Optional[Mapping[str, Any]] = None,
        *,
        duration_time: Optional[float] = None,
        duration_unit: str = "s",
        energy: Optional[float] = None,
        energy_unit: str = "Joules",
    ) -> None:
        """Append one manual-mode measurement block to the runtime output file."""
        entries: list[tuple[str, str, Any]] = []
        if duration_time is not None:
            entries.append(("duration_time", duration_unit, duration_time))
        if energy is not None:
            entries.append(("energy-pkg", energy_unit, energy))
        if metrics is not None:
            for name, value in metrics.items():
                if isinstance(value, tuple):
                    metric_value, metric_unit = value
                    entries.append((name, str(metric_unit), metric_value))
                else:
                    entries.append((name, "", value))
        self._write_manual_block(entries)

    def write_measurements(self, measurements: Iterable[Mapping[str, Any]]) -> None:
        """Append multiple manual-mode measurement blocks, one marker per block."""
        for metrics in measurements:
            self.write_measurement(metrics)

    def _write_manual_block(self, entries: list[tuple[str, str, Any]]) -> None:
        if self._mode != "manual":
            return
        if self._output_path is None:
            print(
                "manual measurement requested but DAISY_CI_RUNTIME_OUTPUT is unset",
                file=sys.stderr,
            )
            return
        if not entries:
            return
        with open(self._output_path, "a", encoding="utf-8") as output:
            for name, unit, value in entries:
                output.write(f"{name},{unit},{value}\n")
            output.write("#### DAISYTUNER Marker ####\n")

    @contextmanager
    def measure(self) -> Iterator["PerfControl"]:
        """Enable counting on entry and disable it on exit."""
        self.start()
        try:
            yield self
        finally:
            self.stop()

    def close(self) -> None:
        """Close the control/ack channel file objects."""
        for handle in (self._ctl, self._ack):
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
        self._ctl = None
        self._ack = None

    def __enter__(self) -> "PerfControl":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _parse_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None
