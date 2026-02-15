import os
import pytest
import numpy as np
import tempfile
import json
import sys

from docc.python import native


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
