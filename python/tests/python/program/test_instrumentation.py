import os
import pytest
import numpy as np
import tempfile
import json
import sys
import subprocess

from docc.python import native
from docc.benchmarks import Trace
from docc.benchmarks.rtl import _find_rtl_lib

_requires_linux = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="requires the shared RTL (DAISY_RTL_SHARED) build",
)


def _run_instrumented(code: str, trace_file: str):
    """Run `code` in a subprocess with instrumentation enabled (trace flushes at exit)."""
    fd_script, script_file = tempfile.mkstemp(suffix=".py")
    os.close(fd_script)
    with open(script_file, "w") as f:
        f.write(code)

    env = os.environ.copy()
    env["DOCC_CI"] = "regions"
    env["__DAISY_PAPI_VERSION"] = "0x07020000"
    env["__DAISY_INSTRUMENTATION_FILE"] = trace_file
    env["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"
    try:
        return subprocess.run(
            [sys.executable, script_file], env=env, capture_output=True, text=True
        )
    finally:
        if os.path.exists(script_file):
            os.remove(script_file)


def test_instrumentation_compile():
    # Test only capture
    @native(instrumentation_mode="", capture_args=True)
    def vec_add_capture(A, B, C):
        for i in range(A.shape[0]):
            C[i] = A[i] + B[i]

    N = 1024
    A = np.random.rand(N)
    B = np.random.rand(N)
    C = np.zeros(N)
    vec_add_capture(A, B, C)


@pytest.mark.skipif(
    sys.platform == "darwin", reason="Instrumentation not supported on macOS"
)
def test_env_var_instrumentation():
    # Create a temporary file for the trace
    fd, trace_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)

    # Create a temporary python script
    fd_script, script_file = tempfile.mkstemp(suffix=".py")
    os.close(fd_script)

    code = """
from docc.python import native
import numpy as np
import os

@native
def vec_add_env(A, B, C):
    for i in range(A.shape[0]):
        C[i] = A[i] + B[i]

N = 1024
A = np.random.rand(N)
B = np.random.rand(N)
C = np.zeros(N)

vec_add_env(A, B, C)
"""

    with open(script_file, "w") as f:
        f.write(code)

    env = os.environ.copy()
    env["DOCC_CI"] = "ON"
    env["__DAISY_PAPI_VERSION"] = "0x07020000"
    env["__DAISY_INSTRUMENTATION_FILE"] = trace_file
    env["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"

    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, script_file], env=env, capture_output=True, text=True
        )

        if result.returncode != 0:
            print("Subprocess stdout:", result.stdout)
            print("Subprocess stderr:", result.stderr)

        assert result.returncode == 0

        # Verify trace file exists and has content
        assert os.path.exists(trace_file)
        with open(trace_file, "r") as f:
            content = f.read()
            assert len(content) > 0
            try:
                trace = json.loads(content)
                assert "traceEvents" in trace
                events = trace["traceEvents"]
                assert len(events) == 1

                args = events[0]["args"]
                assert args["function"] == "vec_add_env"
                assert args["source_ranges"][0]["from"]["line"] == 8
                assert args["source_ranges"][0]["from"]["col"] == 5
                assert args["source_ranges"][0]["to"]["line"] == 9
                assert args["source_ranges"][0]["to"]["col"] == 27
            except json.JSONDecodeError:
                pass

    finally:
        if os.path.exists(trace_file):
            os.remove(trace_file)
        if os.path.exists(script_file):
            os.remove(script_file)


_TWO_SDFGS = """
from docc.python import native
import numpy as np


@native
def kernel_add(A, B, C):
    for i in range(A.shape[0]):
        C[i] = A[i] + B[i]


@native
def kernel_mul(A, B, C):
    for i in range(A.shape[0]):
        C[i] = A[i] * B[i]


N = 256
A = np.random.rand(N)
B = np.random.rand(N)
C = np.zeros(N)
kernel_add(A, B, C)
kernel_mul(A, B, C)
"""


@_requires_linux
def test_two_sdfgs_contribute_to_trace():
    assert _find_rtl_lib() is not None

    fd, trace_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        result = _run_instrumented(_TWO_SDFGS, trace_file)
        assert result.returncode == 0, result.stderr

        # Both compiled functions feed the one shared RTL instance, so a single
        # trace holds a region from each SDFG.
        trace = Trace.load(trace_file, validate_schema=False)
        functions = {r.function for r in trace.regions}
        assert {"kernel_add", "kernel_mul"} <= functions, functions
    finally:
        if os.path.exists(trace_file):
            os.remove(trace_file)


_STOP_ONE_SDFG = """
from docc.python import native
from docc.benchmarks.rtl import start_instrumentation, stop_instrumentation
import numpy as np


@native
def kernel_add(A, B, C):
    for i in range(A.shape[0]):
        C[i] = A[i] + B[i]


@native
def kernel_mul(A, B, C):
    for i in range(A.shape[0]):
        C[i] = A[i] * B[i]


N = 256
A = np.random.rand(N)
B = np.random.rand(N)
C = np.zeros(N)

# Run kernel_mul with measurement globally stopped, then re-enable for kernel_add.
# Both regions still register with the shared RTL, but only kernel_add records
# samples.
stop_instrumentation()
kernel_mul(A, B, C)
start_instrumentation()
kernel_add(A, B, C)
"""


@_requires_linux
def test_start_stop_excludes_one_sdfg():
    assert _find_rtl_lib() is not None

    fd, trace_file = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        result = _run_instrumented(_STOP_ONE_SDFG, trace_file)
        assert result.returncode == 0, result.stderr

        trace = Trace.load(trace_file, validate_schema=False)
        by_function = {r.function: r for r in trace.regions}

        # kernel_add ran while enabled: it has at least one runtime sample.
        assert "kernel_add" in by_function, by_function.keys()
        add = by_function["kernel_add"]
        assert add.runtime is not None
        assert add.runtime.count >= 1, add.runtime.count

        # kernel_mul ran while stopped: it may register but records no samples.
        mul = by_function.get("kernel_mul")
        if mul is not None:
            assert mul.runtime is None or mul.runtime.count == 0, (
                mul.runtime.count if mul.runtime else None
            )
    finally:
        if os.path.exists(trace_file):
            os.remove(trace_file)
