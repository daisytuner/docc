import os
import subprocess
import pytest
import json
import numpy as np
import shutil
from datetime import datetime
from tempfile import mkdtemp

from pathlib import Path
import sys

def _prepend_env_path(var_name: str, path: Path) -> None:
    if not path.exists():
        return
    value = str(path)
    current = os.environ.get(var_name, "")
    if not current:
        os.environ[var_name] = value
        return
    parts = current.split(os.pathsep)
    if value in parts:
        return
    os.environ[var_name] = value + os.pathsep + current

root_dir = Path(__file__).parent / "pytestOut" / "rtl"

# Ensure generated harnesses can find the RTL headers (e.g., <daisy_rtl/daisy_rtl.h>).
# This is required for CI where the include path isn't implicitly configured.
_docc_root = Path(__file__).resolve().parents[1]
_daisy_rtl_include = _docc_root / "docc" / "rtl" / "include"
_prepend_env_path("CPATH", _daisy_rtl_include)
_prepend_env_path("CPLUS_INCLUDE_PATH", _daisy_rtl_include)

def get_output_dir(test_name):
    dir = root_dir / test_name
    Path.mkdir(dir, parents=True, exist_ok=True)
    tmp_dir = Path(mkdtemp(prefix=datetime.now().strftime('%Y-%m-%d_'), dir= dir))
    return tmp_dir

def get_docc_work_dir(test_name):
    dir = get_output_dir(test_name)
    return dir / "DOCC"


def parse_stdout(data, dtype):
    header = data.pop(0)
    assert header == "==BEGIN DUMP_ARRAYS=="

    reference_arrays = {}
    current_array = None
    for line in data:
        if line == "==END   DUMP_ARRAYS==":
            break

        if line.startswith("begin"):
            assert current_array is None
            current_array = line[line.find(":") + 1 :].strip()
            reference_arrays[current_array] = []
        elif line.startswith("end"):
            assert (
                current_array is not None
                and current_array == line[line.find(":") + 1 :].strip()
            )
            reference_arrays[current_array] = np.hstack(reference_arrays[current_array])
            current_array = None
        else:
            assert current_array is not None
            row = np.fromstring(line, dtype=dtype, sep=" ")
            reference_arrays[current_array].append(row)

    return reference_arrays


def parse_arg_captures(capture_path, dtype):
    capture_json = list(capture_path.glob("*.json"))

    results = []

    for i in range(len(capture_json)):
        result = dict()
        data = {}
        with open(capture_json[i], "r") as f:
            data = json.load(f)

        before_args = {}
        after_args = {}
        for capture in data["captures"]:
            array = np.fromfile(capture_path / capture["ext_file"], dtype=dtype)
            argidx = capture["arg_idx"]
            if capture["after"]:
                after_args[argidx] = array
            else:
                before_args[argidx] = array
        result["before"] = before_args
        result["after"] = after_args
        result["data"] = data
        results.append(result)

    return results

def load_sdfg(workdir):
    from docc.sdfg import StructuredSDFG

    indices = list(workdir.rglob("JSON"))
    sdfg = None
    for path in indices:
        with open(path, "r") as f:
            index = json.load(f)
            for entry in index["sdfgs"]:
                if entry["name"] == "main":
                    sdfg = StructuredSDFG.from_file(
                        str((path.parent / entry["file"]).absolute())
                    )
                    break

        if sdfg is not None:
            break

    if sdfg is None:
        raise RuntimeError("Could not find compiled SDFG")

    return sdfg

def test_optional_flag_docc_instrument():
    workdir = Path(__file__).parent / "tests" / "polybench" / "datamining" / "correlation"
    output_dir = get_output_dir("instrument")
    doccWorkDir = output_dir / "DOCC"

    benchmark_path = workdir / "correlation.c"
    output_path = output_dir / "correlation.out"
    cmd = [
        "docc",
        "-g",
        "-O3",
        "-DMEDIUM_DATASET",
        "-DDATA_TYPE_IS_FLOAT",
        "-I" + str(Path(__file__).parent / "tests" / "polybench" / "utilities"),
        str(Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"),
        str(benchmark_path),
        "-o",
        str(output_path),
        "-docc-work-dir=" + str(doccWorkDir),
        "-docc-instrument=ols",
        "-lm",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    trace_file = output_dir / "data_cpu.json"

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(trace_file)
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = "perf::INSTRUCTIONS,perf::CYCLES"

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    with open(trace_file, "r") as f:
        trace = json.load(f)
        events = trace["traceEvents"]
        assert len(events) > 4


        # Check first event as a sample
        event_0 = events[0]
        assert event_0["ph"] == "X"
        assert event_0["cat"] == "aggregated_region,daisy"
        assert event_0["name"] == "kernel_correlation [L110-118]"

        event_0_args = event_0["args"]
        assert event_0_args["module"] == "correlation.c"
        assert event_0_args["function"] == "kernel_correlation"
        assert event_0_args["source_ranges"][0]["from"]["line"] == 110
        assert event_0_args["source_ranges"][0]["to"]["line"] == 118
        assert event_0_args["source_ranges"][0]["from"]["col"] > 0 
        assert event_0_args["source_ranges"][0]["to"]["col"] > 0

        event_0_docc = event_0_args["docc"]
        assert event_0_docc["sdfg_name"] == "main"
        assert Path(event_0_docc["sdfg_file"]).is_file()
        assert event_0_docc["element_id"] > 0
        assert event_0_docc["element_type"] == "map"
        assert event_0_docc["loopnest_index"] == 4

        event_0_loop_info = event_0_docc["loop_info"]
        assert event_0_loop_info["num_loops"] == 3
        assert event_0_loop_info["num_maps"] == 2
        assert event_0_loop_info["num_fors"] == 1
        assert event_0_loop_info["num_whiles"] == 0
        assert event_0_loop_info["max_depth"] == 3
        assert event_0_loop_info["is_perfectly_nested"] == False
        assert event_0_loop_info["is_perfectly_parallel"] == False
        assert event_0_loop_info["is_elementwise"] == False
        assert event_0_loop_info["has_side_effects"] == False

        event_0_metrics = event_0_args["metrics"]
        assert "perf::INSTRUCTIONS" in event_0_metrics
        assert "mean" in event_0_metrics["perf::INSTRUCTIONS"]
        assert "count" in event_0_metrics["perf::INSTRUCTIONS"]
        assert event_0_metrics["perf::INSTRUCTIONS"]["count"] == 1

        assert "perf::CYCLES" in event_0_metrics
        assert "mean" in event_0_metrics["perf::CYCLES"]
        assert "count" in event_0_metrics["perf::CYCLES"]
        assert event_0_metrics["perf::CYCLES"]["count"] == 1

        assert "runtime" in event_0_metrics
        assert "mean" in event_0_metrics["runtime"]
        assert "count" in event_0_metrics["runtime"]
        assert event_0_metrics["runtime"]["count"] == 1

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
    workdir = Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "symm"
    output_dir = get_output_dir("instrument-cuda")
    doccWorkDir = output_dir / "DOCC"

    benchmark_path = workdir / "symm.c"
    output_path = output_dir / "symm.out"
    cmd = [
        "docc",
        "-docc-tune=cuda",
        "-g",
        "-O3",
        "-DMEDIUM_DATASET",
        "-DDATA_TYPE_IS_FLOAT",
        "-I" + str(Path(__file__).parent / "tests" / "polybench" / "utilities"),
        str(Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"),
        str(benchmark_path),
        "-o",
        str(output_path),
        "-docc-work-dir=" + str(doccWorkDir),
        "-docc-instrument=ols",
        "-lm",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    trace_file = output_dir / "data_cuda.json"

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(trace_file)
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = ""
    os.environ["__DAISY_INSTRUMENTATION_EVENTS_CUDA"] = event

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

    trace = json.load(open(trace_file))
    events = trace["traceEvents"]
    assert len(events) > 0
    print(events)

    found_gpu_events = 0
    for i in range(len(events)):
        assert events[i]["cat"] == "region,daisy"
        assert events[i]["ph"] == "X"
        assert events[i]["args"]["module"] == "symm.c"
        assert events[i]["args"]["source_ranges"][0]["from"]["line"] > 0
        assert events[i]["args"]["source_ranges"][0]["to"]["line"] > 0
        assert events[i]["args"]["source_ranges"][0]["from"]["col"] > 0
        assert events[i]["args"]["source_ranges"][0]["to"]["col"] > 0

        includes_events = len(event_names) > 0
        for event_name in event_names:
            includes_events = includes_events and (
                event_name in events[i]["args"]["metrics"]
            )
        if includes_events:
            found_gpu_events += 1

    if event_names:
        assert found_gpu_events > 0


def test_ci_mode():
    """
    This test checks that the CI mode works as expected. It compiles and runs
    the correlation benchmark without and with instrumentation and arg-capturing enabled
    and verifies that the instrumentation data is collected correctly.
    """

    workdir = Path(__file__).parent / "tests" / "polybench" / "datamining" / "correlation"
    output_dir = get_output_dir("ci-mode")
    normal_out = output_dir / "correlation.out"
    instrumented_out = output_dir / "correlation.instrumented.out"
    doccWorkDir = output_dir / "DOCC"

    benchmark_path = workdir / "correlation.c"
    output_path = workdir / "correlation.out"
    cmd = [
        "docc",
        "-g",
        "-O3",
        "-DMEDIUM_DATASET",
        "-DDATA_TYPE_IS_FLOAT",
        "-I" + str(Path(__file__).parent / "tests" / "polybench" / "utilities"),
        str(Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"),
        str(benchmark_path),
        "-o",
        str(normal_out),
        "-docc-work-dir=" + str(doccWorkDir),
        "-docc-tune=none",
        "-lm",
    ]

    os.environ["DOCC_CI"] = "ON"
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0
    os.environ["DOCC_CI"] = ""
    assert instrumented_out.exists()

    trace_file = output_dir / "data_linker_override.json"

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(trace_file)
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = ""
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = "perf::INSTRUCTIONS,perf::CYCLES"

    # Run benchmark
    process = subprocess.Popen(
        [str(instrumented_out)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    # Check arg capture
    # arg_capture_path = Path("/tmp/DOCC").rglob("arg_captures")
    # arg_capture_path = list(arg_capture_path)
    # assert len(arg_capture_path) == 0

    # Check instrumenation
    event_names = ["perf::INSTRUCTIONS", "perf::CYCLES"]
    trace = json.load(open(trace_file))
    events = trace["traceEvents"]
    print(events)
    for i in range(len(events)):
        assert events[i]["cat"] == "region,daisy"
        assert events[i]["ph"] == "X"
        assert events[i]["args"]["module"] == "correlation.c"
        assert events[i]["args"]["source_ranges"][0]["from"]["line"] > 0
        assert events[i]["args"]["source_ranges"][0]["to"]["line"] > 0
        assert events[i]["args"]["source_ranges"][0]["from"]["col"] > 0
        assert events[i]["args"]["source_ranges"][0]["to"]["col"] > 0

        for event_name in event_names:
            assert event_name in events[i]["args"]["metrics"]


@pytest.mark.skip(reason="migrate to docc-auto pending.")
@pytest.mark.parametrize(
    "workdir, benchmark, dtype, reference_to_arg_mapping",
    [
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "datamining" / "correlation",
            "correlation.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"corr": 0}
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "datamining" / "correlation",
            "correlation.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"corr": 0}
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "datamining" / "covariance",
            "covariance.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"cov": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "datamining" / "covariance",
            "covariance.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"cov": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemm",
            "gemm.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"C": 2},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemm",
            "gemm.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"C": 2},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemver",
            "gemver.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemver",
            "gemver.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gesummv",
            "gesummv.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gesummv",
            "gesummv.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "symm",
            "symm.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "symm",
            "symm.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "syr2k",
            "syr2k.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "syr2k",
            "syr2k.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "syrk",
            "syrk.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "syrk",
            "syrk.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "trmm",
            "trmm.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "trmm",
            "trmm.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "2mm",
            "2mm.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "2mm",
            "2mm.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"w": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "3mm",
            "3mm.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "3mm",
            "3mm.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "atax",
            "atax.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "atax",
            "atax.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "bicg",
            "bicg.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "bicg",
            "bicg.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "doitgen",
            "doitgen.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "doitgen",
            "doitgen.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "mvt",
            "mvt.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "mvt",
            "mvt.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "cholesky",
            "cholesky.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "cholesky",
            "cholesky.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "durbin",
            "durbin.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "durbin",
            "durbin.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "gramschmidt",
            "gramschmidt.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "gramschmidt",
            "gramschmidt.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "lu",
            "lu.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "lu",
            "lu.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "ludcmp",
            "ludcmp.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "ludcmp",
            "ludcmp.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "trisolv",
            "trisolv.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "solvers" / "trisolv",
            "trisolv.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "medley" / "deriche",
            "deriche.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "medley" / "deriche",
            "deriche.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"G": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "medley" / "floyd-warshall",
            "floyd-warshall.c",
            "-DDATA_TYPE_IS_INT",
            {"path": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "medley" / "nussinov",
            "nussinov.c",
            "-DDATA_TYPE_IS_INT",
            {"table": 0}
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "adi",
            "adi.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "adi",
            "adi.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "fdtd-2d",
            "fdtd-2d.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "fdtd-2d",
            "fdtd-2d.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "heat-3d",
            "heat-3d.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "heat-3d",
            "heat-3d.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "jacobi-1d",
            "jacobi-1d.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "jacobi-1d",
            "jacobi-1d.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "jacobi-2d",
            "jacobi-2d.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "jacobi-2d",
            "jacobi-2d.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "seidel-2d",
            "seidel-2d.c",
            "-DDATA_TYPE_IS_DOUBLE",
            {"A": 0},
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "seidel-2d",
            "seidel-2d.c",
            "-DDATA_TYPE_IS_FLOAT",
            {"A": 0},
        ),
    ],
)
def test_arg_capture(workdir, benchmark, dtype, reference_to_arg_mapping):
    from docc.sdfg import StructuredSDFG, AnalysisManager

    output_dir = get_output_dir(benchmark+dtype)
    doccWorkDir = output_dir / "DOCC"

    benchmark_path = workdir / benchmark
    output_path = output_dir / benchmark.replace(".c", ".out")
    cmd = [
        "docc",
        "-g",
        "-O3",
        "-DPOLYBENCH_DUMP_ARRAYS",
        "-DMEDIUM_DATASET",
        dtype,
        "-I" + str(Path(__file__).parent / "tests" / "polybench" / "utilities"),
        str(Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"),
        str(benchmark_path),
        "-o",
        str(output_path),
        "-docc-work-dir=" + str(doccWorkDir),
        "-docc-capture-args",
        "-lm",
    ]

    # Compile
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0
    assert Path(output_path).exists()

    # Run
    process = subprocess.Popen(
        [output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    dtype = (
        np.float64
        if "-DDATA_TYPE_IS_DOUBLE" in dtype
        else np.float32 if "-DDATA_TYPE_IS_FLOAT" in dtype else np.int32
    )

    # Parse reference arrays from stderr
    reference_arrays = parse_stdout(stderr.splitlines(), dtype)

    # Load captured args
    arg_capture_path = doccWorkDir.rglob("arg_captures")
    arg_capture_path = list(arg_capture_path)
    print(arg_capture_path, file=sys.stderr)
    assert len(arg_capture_path) == 1
    arg_capture_path = arg_capture_path[0]
    parsed_captures = parse_arg_captures(arg_capture_path, dtype)

    # For every captured invocation index, exercise the cutout-runner pipeline via Python bindings.
    # - element_id is taken from the capture index json
    # - input SDFG is expected in the parent folder as sdfg_0.json
    sdfg_path = arg_capture_path.parent / "sdfg_0.json"
    assert sdfg_path.exists(), f"Expected input SDFG at {sdfg_path}"

    harness_out_dir = output_dir / "cutout_harnesses"
    harness_out_dir.mkdir(parents=True, exist_ok=True)

    index_files = sorted(arg_capture_path.glob("*.index.json"))
    assert index_files, f"No *.index.json files found in {arg_capture_path}"

    for index_file in index_files:
        with open(index_file, "r") as f:
            idx = json.load(f)
        element_id = int(idx["element_id"])
        output_prefix = harness_out_dir / f"{index_file.stem}."
        result = run_cutout_runner(
            str(sdfg_path),
            element_id,
            str(index_file),
            str(output_prefix),
            "sequential",
        )
        assert "main_cpp" in result
        assert "baseline_runtime" in result
        assert Path(result["main_cpp"]).exists(), f"Expected generated harness main at {result['main_cpp']}"
        assert isinstance(result["baseline_runtime"], float)
        assert result["baseline_runtime"] >= 0.0

    sdfg = load_sdfg(doccWorkDir)
    analysis_manager = AnalysisManager(sdfg)
    loop_analysis = analysis_manager.loop_analysis()
    loop_infos = loop_analysis.loop_infos()

    print(loop_infos, file=sys.stderr)

    # Verify output
    for loop_info in loop_infos:
        element_id = loop_info.element_id
        if loop_info.num_maps == 0:
            continue  # We only capture args for maps
        if loop_info.has_side_effects:
            continue  # We do not capture args for maps with side effects
        capture = None
        for parsed_capture in parsed_captures:
            if parsed_capture["data"]["element_id"] == str(element_id):
                capture = parsed_capture
                break
        assert capture is not None, f"No capture found for element ID {element_id}: indvar {loop_info.indvar}"

    # for ref, arg_idx in reference_to_arg_mapping.items():
    #     assert (
    #         arg_idx in after_args
    #     ), f"Argument index {arg_idx} not found in captured args"
    #     assert (
    #         ref in reference_arrays
    #     ), f"Reference array '{ref}' not found in reference arrays"

    #     ref_array = reference_arrays[ref]
    #     arg_array = after_args[arg_idx]

    #     assert (
    #         ref_array.shape == arg_array.shape
    #     ), f"Shape mismatch for '{ref}': reference shape {ref_array.shape}, captured shape {arg_array.shape}"
    #     assert np.array_equal(
    #         reference_arrays[ref], after_args[arg_idx], equal_nan=True
    #     )

@pytest.mark.parametrize(
    "workdir, benchmark, dtype, optlevel, result",
    [
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "datamining" / "covariance",
            "covariance.c",
            "-DDATA_TYPE_IS_DOUBLE",
            "-O3",
            2029315760.0
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "datamining" / "covariance",
            "covariance.c",
            "-DDATA_TYPE_IS_FLOAT",
            "-O3",
            2030995760.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "datamining" / "covariance",
            "covariance.c",
            "-DDATA_TYPE_IS_FLOAT",
            "-O0",
            0.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemm",
            "gemm.c",
            "-DDATA_TYPE_IS_DOUBLE",
            "-O3",
            3968914560.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemm",
            "gemm.c",
            "-DDATA_TYPE_IS_FLOAT",
            "-O3",
            3968914560.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemver",
            "gemver.c",
            "-DDATA_TYPE_IS_DOUBLE",
            "-O3",
            48208560.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "blas" / "gemver",
            "gemver.c",
            "-DDATA_TYPE_IS_FLOAT",
            "-O3",
            48208560.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "3mm",
            "3mm.c",
            "-DDATA_TYPE_IS_DOUBLE",
            "-O3",
            5408294560.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "linear-algebra" / "kernels" / "3mm",
            "3mm.c",
            "-DDATA_TYPE_IS_FLOAT",
            "-O3",
            5408294560.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "medley" / "floyd-warshall",
            "floyd-warshall.c",
            "-DDATA_TYPE_IS_INT",
            "-O3",
            4194560.0,
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "heat-3d",
            "heat-3d.c",
            "-DDATA_TYPE_IS_DOUBLE",
            "-O3",
            24653150560.0
        ),
        pytest.param(
            Path(__file__).parent / "tests" / "polybench" / "stencils" / "heat-3d",
            "heat-3d.c",
            "-DDATA_TYPE_IS_FLOAT",
            "-O3",
            24653150560.0,
        ),
    ],
)
def test_flop_values(workdir, benchmark, dtype, optlevel, result):
    output_dir = get_output_dir(benchmark+dtype)
    doccWorkDir = output_dir / "DOCC"

    benchmark_path = workdir / benchmark
    output_path = output_dir / benchmark.replace(".c", ".out")
    cmd = [
        "docc",
        "-g",
        optlevel,
        "-DLARGE_DATASET",
        "-DPOLYBENCH_TIME",
        dtype,
        "-I" + str(Path(__file__).parent / "tests" / "polybench" / "utilities"),
        str(Path(__file__).parent / "tests" / "polybench" / "utilities" / "polybench.c"),
        str(benchmark_path),
        "-o",
        str(output_path),
        "-lm",
        "-docc-work-dir=" + str(doccWorkDir),
        "-docc-instrument=ols"
    ]

    # Compile
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0
    assert Path(output_path).exists()

    (workdir / "data_cpu.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cpu.json")

    # Run
    process = subprocess.Popen(
        [output_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
    assert process.returncode == 0

    trace = json.load(open(workdir / "data_cpu.json"))
    static_flops = 0.0
    for event in trace["traceEvents"]:
        assert "args" in event
        assert "metrics" in event["args"]
        # Currently with -O0 there are lots of while loops, which do not have a static flop count
        if ("static:::flop" not in event["args"]["metrics"]) and (optlevel == "-O0"):
            continue
        assert "static:::flop" in event["args"]["metrics"]
        assert "mean" in event["args"]["metrics"]["static:::flop"]
        static_flops += float(event["args"]["metrics"]["static:::flop"]["mean"])
    assert static_flops == result
