import errno
import os
import queue
import select
import threading
from io import StringIO

import pytest

from docc.benchmarks import PerfControl
import docc.benchmarks.perf as perf_module

MARKER = "#### DAISYTUNER Marker ####"


def assert_closed(fd):
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF


def close_if_open(fd):
    try:
        os.close(fd)
    except OSError as error:
        assert error.errno == errno.EBADF


def test_automatic_mode_is_noop(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "automatic",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    assert perf.mode == "automatic"
    assert not perf.enabled

    with perf.measure():
        # No-op
        perf.record_metric("energy-pkg", 1.0, "Joules")

    assert not output.exists()


def test_control_mode_sends_enable_disable():
    ctl = StringIO()
    ack = StringIO("ack\nack\n")
    perf = PerfControl(ctl, ack, mode="control", verbose=False)

    try:
        with perf.measure():
            assert perf.active

        assert not perf.active
        assert ctl.getvalue().splitlines() == ["enable", "disable"]
    finally:
        perf.close()


def test_manual_mode_measure_does_not_write_implicit_duration(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
            "DAISY_CI_MEASUREMENTS": "3",
        },
        verbose=False,
    )

    assert perf.mode == "manual"
    assert perf.enabled
    assert perf.measurements == 3

    with perf.measure():
        pass

    assert not output.exists()


def test_manual_mode_measure_callback_writes_provided_metrics(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    result = perf.measure_callback(
        lambda: {
            "duration_time": (125, "ms"),
            "energy-pkg": (0.75, "Joules"),
            "custom_score": 9,
        }
    )

    assert result == {
        "duration_time": (125, "ms"),
        "energy-pkg": (0.75, "Joules"),
        "custom_score": 9,
    }
    assert output.read_text(encoding="utf-8").splitlines() == [
        "duration_time,ms,125",
        "energy-pkg,Joules,0.75",
        "custom_score,,9",
        MARKER,
    ]


def test_manual_mode_write_measurement_accepts_extra_metrics(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    perf.write_measurement(
        duration_time=125,
        duration_unit="ms",
        energy=0.75,
        metrics={"cache_misses": (42, "count"), "custom_score": 9},
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "duration_time,ms,125",
        "energy-pkg,Joules,0.75",
        "cache_misses,count,42",
        "custom_score,,9",
        MARKER,
    ]


def test_manual_mode_write_measurements_separates_each_block(tmp_path):
    output = tmp_path / "runtime.csv"
    perf = PerfControl.from_env(
        {
            "DAISY_CI_MEASUREMENT_MODE": "manual",
            "DAISY_CI_RUNTIME_OUTPUT": str(output),
        },
        verbose=False,
    )

    perf.write_measurements(
        [
            {"duration_time": (100, "ms"), "energy-pkg": (0.10, "Joules")},
            {"duration_time": (110, "ms"), "energy-pkg": (0.12, "Joules")},
            {"duration_time": (105, "ms"), "energy-pkg": (0.11, "Joules")},
        ]
    )

    assert output.read_text(encoding="utf-8").splitlines() == [
        "duration_time,ms,100",
        "energy-pkg,Joules,0.1",
        MARKER,
        "duration_time,ms,110",
        "energy-pkg,Joules,0.12",
        MARKER,
        "duration_time,ms,105",
        "energy-pkg,Joules,0.11",
        MARKER,
    ]


@pytest.mark.parametrize(
    "descriptor", ["not-a-descriptor", "", "-1", str(1 << 100), str(1 << 30)]
)
def test_invalid_control_descriptors_disable_measurement(descriptor, capsys):
    perf = PerfControl.from_env({"PERF_CTL_FD": descriptor}, verbose=False)
    try:
        assert perf.mode == "control"
        assert not perf.enabled
        assert perf._ctl is None and perf._ack is None
        with perf.measure():
            pass
        diagnostic = capsys.readouterr().err
        assert diagnostic.startswith("perf control unavailable (")
        assert diagnostic.endswith("); measurement not gated\n")
    finally:
        perf.close()


@pytest.mark.parametrize(
    "descriptor", ["not-a-descriptor", "", "-1", str(1 << 100), str(1 << 30)]
)
def test_invalid_ack_descriptors_close_control_channel(descriptor, capsys, monkeypatch):
    fdopen = os.fdopen
    handles = []

    def keep_handle(fd, mode):
        handle = fdopen(fd, mode)
        handles.append(handle)
        return handle

    monkeypatch.setattr(perf_module.os, "fdopen", keep_handle)
    read_fd, write_fd = os.pipe()
    try:
        perf = PerfControl.from_env(
            {"PERF_CTL_FD": str(write_fd), "PERF_ACK_FD": descriptor}, verbose=False
        )
        try:
            assert not perf.enabled
            assert perf._ctl is None and perf._ack is None
            assert_closed(write_fd)
            assert os.read(read_fd, 1) == b""
            assert "measurement not gated" in capsys.readouterr().err
        finally:
            perf.close()
    finally:
        for handle in handles:
            handle.close()
        close_if_open(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize(
    "close_error", [OSError(errno.EIO, "close failed"), ValueError("close failed")]
)
def test_setup_and_cleanup_failures_are_both_reported(monkeypatch, close_error, capsys):
    read_fd, write_fd = os.pipe()
    fdopen = os.fdopen
    handles = []

    class FailingClose:
        def __init__(self, handle):
            self.handle = handle
            self.calls = 0

        def close(self):
            self.calls += 1
            self.handle.close()
            raise close_error

    def wrap(fd, mode):
        handle = FailingClose(fdopen(fd, mode))
        handles.append(handle)
        return handle

    monkeypatch.setattr(perf_module.os, "fdopen", wrap)
    try:
        with pytest.raises(ExceptionGroup) as raised:
            PerfControl.from_env(
                {"PERF_CTL_FD": str(write_fd), "PERF_ACK_FD": "invalid-ack"},
                verbose=False,
            )
        group = raised.value
        assert group.message == "Perf control setup failed and cleanup also failed"
        assert len(group.exceptions) == 2
        assert type(group.exceptions[0]) is ValueError
        assert "invalid-ack" in str(group.exceptions[0])
        assert group.exceptions[1] is close_error
        assert handles[0].calls == 1
        assert_closed(write_fd)
        assert os.read(read_fd, 1) == b""
        assert capsys.readouterr().err == ""
    finally:
        close_if_open(write_fd)
        os.close(read_fd)


def test_valid_pipes_wait_for_each_ack_and_close(monkeypatch):
    ctl_read, ctl_write = os.pipe()
    ack_read, ack_write = os.pipe()
    entered = threading.Event()
    finished = threading.Event()
    release_body = threading.Event()
    reading_ack = queue.Queue()
    errors = queue.Queue()
    fdopen = os.fdopen

    class TrackedReader:
        def __init__(self, handle):
            self.handle = handle

        def readline(self):
            reading_ack.put(True)
            return self.handle.readline()

        def close(self):
            self.handle.close()

    def track(fd, mode):
        handle = fdopen(fd, mode)
        return TrackedReader(handle) if mode == "r" else handle

    monkeypatch.setattr(perf_module.os, "fdopen", track)
    perf = PerfControl.from_env(
        {"PERF_CTL_FD": str(ctl_write), "PERF_ACK_FD": str(ack_read)}, verbose=False
    )

    def measure():
        try:
            with perf:
                with perf.measure():
                    entered.set()
                    assert release_body.wait(2)
            finished.set()
        except BaseException as error:
            errors.put(error)

    worker = threading.Thread(target=measure)
    worker.start()
    try:
        reading_ack.get(timeout=2)
        assert select.select([ctl_read], [], [], 2)[0] == [ctl_read]
        assert os.read(ctl_read, 64) == b"enable\n"
        assert not entered.is_set()
        os.write(ack_write, b"ack\n")
        assert entered.wait(2)
        assert perf.active
        release_body.set()
        reading_ack.get(timeout=2)
        assert select.select([ctl_read], [], [], 2)[0] == [ctl_read]
        assert os.read(ctl_read, 64) == b"disable\n"
        assert not finished.is_set()
        os.write(ack_write, b"ack\n")
        assert finished.wait(2)
        worker.join(2)
        assert not worker.is_alive()
        assert errors.empty()
        assert not perf.active
        assert_closed(ctl_write)
        assert_closed(ack_read)
    finally:
        release_body.set()
        os.close(ack_write)
        worker.join(3)
        perf.close()
        os.close(ctl_read)
        assert not worker.is_alive()
