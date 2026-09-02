# DOCC RTL

The DOCC RTL (`-ldaisy_rtl`) is the runtime library responsible for recording instrumented code regions.


## Measurement API

### Region instrumentation

Regions are registered for measurement by first creating a `__daisy_metadata_t` object and invoking `__daisy_instrumentation_init`.
For example, the following example registers function `foo` for recording with CPU hardware counters.

```
// Basic metadata, can be extended with further region information.
__daisy_metadata_t metadata = {
        .file_name = "test.c",
        .function_name = "foo",
        .line_begin = 1,
        .line_end = 10,
        .target_type = "SEQUENTIAL",
        .region_uuid = "test_foo"
    };
    unsigned long long region_id = __daisy_instrumentation_init(&metadata, __DAISY_EVENT_SET_CPU);
```

For GPU kernels, use `__DAISY_EVENT_SET_CUDA` instead (works for ROCm too). Counters are disabled using `__DAISY_EVENT_SET_NONE`.

Regions are subsequently started using the unique region ID:

```
__daisy_instrumentation_enter(region_id);
// Work happens here...
__daisy_instrumentation_exit(region_id);
```

We use the space between exit and finalize to record "static" counters that the compiler can compute, e.g. bytes transferred or flops.

Call `__daisy_instrumentation_finalize(region_id)` after the final region measurement to finish recording.
After all regions are finished, calling `__daisy_instrumentation_finalize_all` is required to clean up asynchronous measurements and update aggregate metrics.

For the full list of API calls and metadata fields, refer to [rtl/include/daisy_rtl/daisy_rtl.h](include/daisy_rtl/daisy_rtl.h).

### Argument capturing

Besides recording performance metrics, the RTL handles capturing kernel arguments to be used as input for autotuning.
Similar to region instrumentation, capturing requires initialization with `__daisy_capture_init`.
`__daisy_capture_enter` is called when entering a region and `__daisy_capture_end` when exiting.
Arguments are recorded using `__daisy_capture_raw`, `__daisy_capture_1d`, `__daisy_capture_2d` and `__daisy_capture_3d`, depending on the dimensionality of the input data.
For a basic usage example, refer to [rtl/applications/tests/capture_test.c](tests/applications/capture_test.c).


### Python bindings

Python bindings for the instrumentation API, as well as utilities for reading and analyzing traces, are available in [python/docc/benchmarks/rtl.py](../python/docc/benchmarks/rtl.py).


## Recording a trace

After execution, traces are stored using a subset of the [Google trace event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw), annotated with metadata. See the [JSON schema here](schema/daisy_trace.schema.json) for specifics.

Two modes are available:
- full trace (default): records individual region invocations
- aggregate: records summary metrics (invocation count and runtime statistics (min/mean/max))

### Configuration

Tracing can be configured using the following environment variables:
- `DAISY_PAPI_PATH`: Path to the PAPI library. System paths are used if not specified.
- `__DAISY_PAPI_VERSION` (required): PAPI library version (e.g., `0x07020000` for version 7.2.0.0)
- `__DAISY_INSTRUMENTATION_MODE`: Set to `aggregate` to activate aggregate mode (default is full tracing)
- `__DAISY_INSTRUMENTATION_FILE`: Output location of the trace file
- `__DAISY_INSTRUMENTATION_EVENTS`: List of PAPI events to record on the CPU
- `__DAISY_INSTRUMENTATION_EVENTS_CUDA`: List of PAPI events to record on the GPU
- `__DAISY_INSTRUMENTATION_ASYNC_CAPTURE` Enable/disable asynchronous event capturing (on by default for aggregate GPU event recording)

Captured arguments are written to individual files in the directory specified by `__DAISY_CAPTURE_BASE_DIR`.
The default capturing behavior is defined by `__DAISY_CAPTURE_STRATEGY_DEFAULT`:
- `never`: disable capture
- `once`: capture the first invocation
- `always` capture all invocations