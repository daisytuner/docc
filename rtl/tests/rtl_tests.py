import os
import subprocess
import pytest
import json
import numpy as np

from pathlib import Path
import base64


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(""),
        pytest.param("perf::BRANCHES,perf::CYCLES"),
    ],
)
def test_instrumentation(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_test.c"
    output_path = workdir / "instrumentation_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_cpu.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cpu.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = ""

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    event_names = []
    if event:
        event_names = event.split(",")

    result = json.load(open(workdir / "data_cpu.json"))
    events = result["traceEvents"]
    assert len(events) == 10

    for i in range(len(events)):
        # General checks
        assert events[i]["ph"] == "X"
        assert events[i]["cat"] == "region,daisy"
        assert events[i]["name"] == "main [L18-31]"

        # Source Metadata checks
        assert events[i]["args"]["module"] == "instrumentation_test.c"
        assert events[i]["args"]["function"] == "main"
        assert events[i]["args"]["source_ranges"][0]["from"]["line"] == 18
        assert events[i]["args"]["source_ranges"][0]["to"]["line"] == 31
        assert events[i]["args"]["source_ranges"][0]["from"]["col"] == 4
        assert events[i]["args"]["source_ranges"][0]["to"]["col"] == 5

        # DOCC Metadata checks
        docc_metadata = events[i]["args"]["docc"]
        assert docc_metadata["sdfg_name"] == "__daisy_instrumentation_test_0"
        assert docc_metadata["sdfg_file"] == "/tmp/DOCC/0000-0000/123456789/sdfg_0.json"
        assert docc_metadata["arg_capture_path"] == ""
        assert docc_metadata["element_id"] == 10
        assert docc_metadata["loopnest_index"] == 0

        assert events[i]["args"]["target_type"] == "sequential"

        # Event metrics checks
        assert len(events[i]["args"]["metrics"]) == len(event_names)
        for event_name in event_names:
            assert event_name in events[i]["args"]["metrics"]
            assert events[i]["args"]["metrics"][event_name] > 0


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(""),
        pytest.param("perf::BRANCHES,perf::CYCLES"),
    ],
)
def test_instrumentation_aggregate(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_test.c"
    output_path = workdir / "instrumentation_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_cpu.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cpu.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    event_names = []
    if event:
        event_names = event.split(",")

    result = json.load(open(workdir / "data_cpu.json"))
    events = result["traceEvents"]
    assert len(events) == 1

    event = events[0]
    # General checks
    assert event["ph"] == "X"
    assert event["cat"] == "aggregated_region,daisy"
    assert event["name"] == "main [L18-31]"

    # Source Metadata checks
    assert event["args"]["module"] == "instrumentation_test.c"
    assert event["args"]["function"] == "main"
    assert event["args"]["source_ranges"][0]["from"]["line"] == 18
    assert event["args"]["source_ranges"][0]["to"]["line"] == 31
    assert event["args"]["source_ranges"][0]["from"]["col"] == 4
    assert event["args"]["source_ranges"][0]["to"]["col"] == 5

    # DOCC Metadata checks
    docc_metadata = event["args"]["docc"]
    assert docc_metadata["sdfg_name"] == "__daisy_instrumentation_test_0"
    assert docc_metadata["sdfg_file"] == "/tmp/DOCC/0000-0000/123456789/sdfg_0.json"
    assert docc_metadata["element_id"] == 10
    assert docc_metadata["element_type"] == "for"
    assert docc_metadata["loopnest_index"] == 0

    assert event["args"]["target_type"] == "sequential"

    assert len(event["args"]["metrics"]) == len(event_names) + 1
    for event_name in event_names:
        assert event_name in event["args"]["metrics"]
        assert event["args"]["metrics"][event_name]["mean"] > 0
        assert event["args"]["metrics"][event_name]["min"] > 0
        assert event["args"]["metrics"][event_name]["max"] > 0
        assert event["args"]["metrics"][event_name]["count"] == 10
        assert event["args"]["metrics"][event_name]["variance"] >= 0

    assert "runtime" in event["args"]["metrics"]
    assert event["args"]["metrics"]["runtime"]["mean"] > 0
    assert event["args"]["metrics"]["runtime"]["min"] > 0
    assert event["args"]["metrics"]["runtime"]["max"] > 0
    assert event["args"]["metrics"]["runtime"]["variance"] >= 0
    assert event["args"]["metrics"]["runtime"]["count"] == 10

    assert np.allclose(
        event["dur"],
        event["args"]["metrics"]["runtime"]["mean"]
        * event["args"]["metrics"]["runtime"]["count"],
    )


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(""),
        pytest.param(
            "nvml:::NVIDIA_GeForce_RTX_5060_Ti:device_0:gpu_utilization,nvml:::NVIDIA_GeForce_RTX_5060_Ti:device_0:memory_utilization"
        ),
    ],
)
def test_instrumentation_cuda(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_cuda_test.cu"
    output_path = workdir / "instrumentation_cuda_test.out"
    cmd = [
        "nvcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_cuda.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cuda.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS_CUDA"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = ""

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    event_names = []
    if event:
        event_names = event.split(",")

    result = json.load(open(workdir / "data_cuda.json"))
    events = result["traceEvents"]
    assert len(events) > 0

    print(events)
    for i in range(len(events)):
        # General checks
        assert events[i]["ph"] == "X"
        assert events[i]["cat"] == "region,daisy"
        assert events[i]["name"] == "main [L18-31]"

        # Source Metadata checks
        assert events[i]["args"]["module"] == "instrumentation_cuda_test.cu"
        assert events[i]["args"]["function"] == "main"
        assert events[i]["args"]["source_ranges"][0]["from"]["line"] == 18
        assert events[i]["args"]["source_ranges"][0]["to"]["line"] == 31
        assert events[i]["args"]["source_ranges"][0]["from"]["col"] == 4
        assert events[i]["args"]["source_ranges"][0]["to"]["col"] == 5

        # DOCC Metadata checks
        docc_metadata = events[i]["args"]["docc"]
        assert docc_metadata["sdfg_name"] == "__daisy_instrumentation_cuda_test_0"
        assert docc_metadata["sdfg_file"] == "/tmp/DOCC/0000-0000/123456789/sdfg_0.json"
        assert docc_metadata["arg_capture_path"] == ""
        assert docc_metadata["element_id"] == 10
        assert docc_metadata["element_type"] == "for"
        assert docc_metadata["loopnest_index"] == 0

        assert events[i]["args"]["target_type"] == "cuda"

        for event_name in event_names:
            assert event_name in events[i]["args"]["metrics"]


@pytest.mark.parametrize(
    "strat, expected_reports",
    [
        pytest.param("once", "0,-1"),
        pytest.param("all", "0,1"),
        pytest.param("never", "-0,-1"),
        pytest.param("", "0,-1"),
    ],
)
def test_capture_strats(strat, expected_reports):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "capture_test.c"
    output_path = workdir / "capture_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_CAPTURE_STRATEGY_DEFAULT"] = strat

    try:
        # Run benchmark
        process = subprocess.Popen(
            [str(output_path)],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stdout)
            print(stderr)
        assert process.returncode == 0

        invocations = []
        if expected_reports:
            args = expected_reports.split(",")
            invocations = []
            for s in args:
                has_minus = s.startswith("-")
                num = int(s[1:]) if has_minus else int(s)
                invocations.append(
                    (
                        num,
                        f"__daisy_capture_test_function_inv{num}_123.index.json",
                        has_minus,
                    )
                )

        def check_ext_file(cap, expected_array):
            ext_file_path = workdir / "arg_captures" / cap["ext_file"]
            assert (
                ext_file_path.exists()
            ), f"External file {cap['ext_file']} does not exist"
            arr = np.fromfile(ext_file_path, dtype=np.int32)
            np.testing.assert_array_equal(arr, expected_array)

        def check_base64(cap, expected_bytes):
            expected_b64 = base64.b64encode(expected_bytes).decode("ascii")
            assert cap["data"] == expected_b64

        for [inv, inv_file, must_not_exist] in invocations:

            inv_path = workdir / "arg_captures" / inv_file
            if must_not_exist:
                assert (
                    not inv_path.exists()
                ), f"Did not expect capture file {inv_file} to exist"
                continue

            assert inv_path.exists(), f"Expected capture file {inv_file} not found"
            inv_data = json.load(open(inv_path))

            captures = inv_data["captures"]
            assert len(captures) == 4
            # Build a map: (arg_idx, after) -> capture
            capture_map = {}
            for capture in captures:
                key = (capture["arg_idx"], capture["after"])
                capture_map[key] = capture

            assert (0, False) in capture_map
            cap = capture_map[(0, False)]
            assert cap["primitive_type"] == 4
            assert cap["dims"] == [4, 10]

            check_ext_file(cap, np.arange(1, 11))

            assert (1, False) in capture_map
            cap = capture_map[(1, False)]
            assert cap["primitive_type"] == 4
            assert cap["dims"] == [4, 10]
            check_ext_file(cap, np.arange(11, 21))

            assert (2, False) in capture_map
            cap = capture_map[(2, False)]
            assert cap["primitive_type"] == 5
            assert cap["dims"] == [8]
            check_ext_file(
                cap,
                (
                    np.array([-1, -1], dtype=np.int32)
                    if inv == 0
                    else np.zeros(2, dtype=np.int32)
                ),
            )

            assert (3, True) in capture_map
            cap = capture_map[(3, True)]
            assert cap["primitive_type"] == 4
            assert cap["dims"] == [4, 10]
            check_ext_file(
                cap,
                (
                    np.array([12, 14, 16, 18, 20, 22, 24, 26, 28, 30], dtype=np.int32)
                    if inv == 0
                    else np.zeros(10, dtype=np.int32)
                ),
            )

    finally:
        # Cleanup
        subprocess.run(["rm", "-rf", str(workdir / "arg_captures")])


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(""),
    ],
)
def test_instrumentation_static(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_static_test.c"
    output_path = workdir / "instrumentation_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_static.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_static.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    result = json.load(open(workdir / "data_static.json"))
    events = result["traceEvents"]
    assert len(events) == 1

    event = events[0]
    assert "static:::foo" in event["args"]["metrics"]
    assert event["args"]["metrics"]["static:::foo"]["mean"] == 4.5
    assert event["args"]["metrics"]["static:::foo"]["min"] == 0
    assert event["args"]["metrics"]["static:::foo"]["max"] == 9
    assert event["args"]["metrics"]["static:::foo"]["count"] == 10
    assert event["args"]["metrics"]["static:::foo"]["variance"] == 8.25


@pytest.mark.parametrize(
    "event",
    [
        pytest.param(""),
    ],
)
def test_instrumentation_manual(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_manual_test.c"
    output_path = workdir / "instrumentation_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_manual.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_manual.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    result = json.load(open(workdir / "data_static.json"))
    events = result["traceEvents"]
    assert len(events) == 1

    event = events[0]
    assert "static:::foo" in event["args"]["metrics"]
    assert event["args"]["metrics"]["static:::foo"]["mean"] == 4.5
    assert event["args"]["metrics"]["static:::foo"]["min"] == 0
    assert event["args"]["metrics"]["static:::foo"]["max"] == 9
    assert event["args"]["metrics"]["static:::foo"]["count"] == 10
    assert event["args"]["metrics"]["static:::foo"]["variance"] == 8.25


def test_instrumentation_loop_info():
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_loop_info_test.c"
    output_path = workdir / "instrumentation_loop_info_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_cpu.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cpu.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = "perf::CYCLES"
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = ""

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    result = json.load(open(workdir / "data_cpu.json"))
    events = result["traceEvents"]
    assert len(events) == 1

    loop_info = events[0]["args"]["docc"]["loop_info"]
    assert loop_info["num_loops"] == 3
    assert loop_info["num_maps"] == 2
    assert loop_info["num_fors"] == 1
    assert loop_info["num_whiles"] == 0
    assert loop_info["max_depth"] == 3
    assert loop_info["is_perfectly_nested"] == True
    assert loop_info["is_perfectly_parallel"] == False
    assert loop_info["is_elementwise"] == False
    assert loop_info["has_side_effects"] == False


@pytest.mark.parametrize(
    "fixture",
    [
        "sample_per_invocation.json",
        "sample_aggregated.json",
    ],
)
def test_trace_schema_fixtures(fixture):
    from validate_trace import build_validator, validate_trace

    validator = build_validator()
    trace = json.load(open(Path(__file__).parent / "data" / fixture))
    errors = validate_trace(trace, validator)
    assert errors == [], "\n".join(errors)
