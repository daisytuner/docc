"""Runtime control of ``perf stat`` hardware-counter measurement.

When a benchmark process is launched under ``perf stat --control`` with perf
started disabled (``-D -1``), the process can turn counting on and off at
runtime. This lets the reported counters cover only a region of interest -- for
example the steady-state loop -- instead of the whole process, which is
otherwise dominated by interpreter/library import, graph tracing, and one-time
compilation/warmup costs.

The launcher wires up the control channel via environment variables, either as
fifo paths (``PERF_CTL_FIFO`` / ``PERF_ACK_FIFO``) or inherited file descriptors
(``PERF_CTL_FD`` / ``PERF_ACK_FD``). A matching perf invocation looks like::

    ctl=/tmp/perf_ctl.fifo; ack=/tmp/perf_ack.fifo
    mkfifo "$ctl" "$ack"
    PERF_CTL_FIFO=$ctl PERF_ACK_FIFO=$ack \\
    perf stat -D -1 --control fifo:$ctl,$ack -e cycles,instructions -- \\
        python my_benchmark.py

Typical usage inside the benchmark::

    from docc.benchmarks.perf import PerfControl

    perf = PerfControl.from_env()

    warmup()                 # cold-start work, not counted
    with perf.measure():     # counters enabled only inside this block
        for _ in range(n_runs):
            step()

Or manually with start / stop / resume::

    perf.start()
    ...
    perf.stop()              # pause
    perf.resume()            # continue
    ...
    perf.stop()

When no perf control channel is configured (env vars unset) every method is a
no-op, so the same code runs unchanged outside of perf.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import IO, Iterator, Mapping, Optional

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
        verbose: bool = True,
    ) -> None:
        self._ctl = ctl_file
        self._ack = ack_file
        self._active = False
        if verbose and self._ctl is not None:
            print(
                "perf control active: measuring only the selected region",
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
        """
        env = os.environ if env is None else env
        ctl_fd = env.get("PERF_CTL_FD")
        ack_fd = env.get("PERF_ACK_FD")
        ctl_fifo = env.get("PERF_CTL_FIFO")
        ack_fifo = env.get("PERF_ACK_FIFO")

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
        except OSError as exc:
            print(
                f"perf control unavailable ({exc}); measurement not gated",
                file=sys.stderr,
            )
            return cls(None, None, verbose=False)

        return cls(ctl_file, ack_file, verbose=verbose)

    @property
    def enabled(self) -> bool:
        """Whether a perf control channel is wired up."""
        return self._ctl is not None

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
