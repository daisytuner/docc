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

In aggregate mode, `__DAISY_EVENT_SET_CUDA` regions with `target_type` `CUDA`/`ROCM` are timed with GPU events. Every other region with a GPU `target_type` (e.g. blocking memcpy/free, or all GPU regions when async capture is off) is timed on the host clock, and the device is synchronized right before the start and before the end timestamp. Otherwise a blocking call would absorb the runtime of all previously queued kernels, which their own regions already report.

When `__DAISY_INSTRUMENTATION_EVENTS_CUDA` is set, `__DAISY_EVENT_SET_CUDA` regions with `target_type` `CUDA`/`ROCM` also synchronize the device right before reading their counters at enter and exit. Device counters only reflect completed kernels, so an asynchronous launch would otherwise be read before it ran. GPU-event timing is unaffected: the synchronization precedes the start event and follows the stop event.

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

### Readiness and errors

The first call to any `__daisy_instrumentation_*` function initializes the process-global RTL state. Initialization should never terminate the process, as that can hide various failures. The RTL is ready to measure when PAPI is loaded and initialized and all requested events (`__DAISY_INSTRUMENTATION_EVENTS*`) exist. It is not ready if, for example, `libpapi.so` cannot be loaded, `__DAISY_PAPI_VERSION` is unset or does not match the installed PAPI, or a requested event does not exist.

- `__daisy_instrumentation_is_ready()` reports whether the RTL can measure, and `__daisy_instrumentation_error()` returns the reason if it cannot (`NULL` otherwise). If there is a in-process manager of measurements, it check these before running instrumented code.
- Instrumented code has no way to handle errors: if the RTL is not ready, `__daisy_instrumentation_init` prints `[daisy-rtl] Instrumentation runtime not ready: <reason>` and calls `abort()`. Other errors during measurement, e.g. an event that cannot be added to a counter set, also print a message and abort.
- `__daisy_instrumentation_dump()` returns false without writing a trace when the RTL is not ready.

`__daisy_instrumentation_counter_status(target, force_init, reason)` reports whether hardware counters of a target (`__DAISY_COUNTER_TARGET_CPU`, `_CUDA` or `_ROCM`) can be collected, i.e. whether PAPI's component for it (`perf_event`, `cuda`, `rocm`/`rocp_sdk`) is enabled. PAPI initializes GPU components only when their events are first used, so without `force_init` they are reported as `__DAISY_COUNTER_NOT_INITIALIZED`, as initializing them can be slow or even crash the process (we have seen this with ROCm).
With `force_init` the component is initialized now. If `reason` is not `NULL`, it is set to an explanation unless the counters are available.

DOCC's `-docc-instrument=ols_tuned` mode (or Python `instrumentation_mode="ols_tuned"`) emits one instrumentation span around contiguous optimized sibling loops that share an original-loop provenance ID. The trace identifies the group with `docc.source_loop_id` and lists every member's ID and source range in `docc.member_loops`; scalar element/source/loop fields remain representative metadata from the first member. Non-contiguous or return-containing groups fall back to individual loop regions. Runtime `__DAISY_INSTRUMENTATION_MODE` still independently selects per-invocation or aggregate output.

Each event measures one complete span execution. This mode measures tuned execution only; compare against a separately instrumented baseline to calculate empirical speedup.

### Argument capturing

Besides recording performance metrics, the RTL handles capturing kernel arguments to be used as input for autotuning.
Similar to region instrumentation, capturing requires initialization with `__daisy_capture_init`.
`__daisy_capture_enter` is called when entering a region and `__daisy_capture_end` when exiting.
Arguments are recorded using `__daisy_capture_raw`, `__daisy_capture_1d`, `__daisy_capture_2d` and `__daisy_capture_3d`, depending on the dimensionality of the input data.
For a basic usage example, refer to [rtl/applications/tests/capture_test.c](tests/applications/capture_test.c).


### Python bindings

Python bindings for the instrumentation API, as well as utilities for reading and analyzing traces, are available in [python/docc/benchmarks/rtl.py](../python/docc/benchmarks/rtl.py).

- `instrumentation_is_ready()` and `instrumentation_error()` wrap the readiness queries above.
- `ensure_instrumentation_ready()` initializes the RTL and raises `InstrumentationNotReady` with the reason if it cannot measure.
- `try_ensure_instrumentation_ready()` does the same but warns instead of raising. It runs when `docc.python`, `docc.pytorch` or `docc.torch` is imported, but only if `__DAISY_PAPI_VERSION` is set, because the RTL reads its configuration only once.
- `counter_status(target, force_init=False)` wraps `__daisy_instrumentation_counter_status` for `"cpu"`, `"cuda"` and `"rocm"`.


## Recording a trace

After execution, traces are stored using a subset of the [Google trace event format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview?tab=t.0#heading=h.yr4qxyxotyw), annotated with metadata. See the [JSON schema here](schema/daisy_trace.schema.json) for specifics.

Two modes are available:
- full trace (default): records individual region invocations
- aggregate: records summary metrics (invocation count and runtime statistics (min/mean/max))

The mode is read from the environment once, when the process-global RTL state is first used, and is fixed from then on. Query the active mode with `__daisy_instrumentation_mode()`, which returns `__DAISY_MODE_TRACE` or `__DAISY_MODE_AGGREGATE` (`instrumentation_mode()` in the Python bindings).

### Configuration

Tracing can be configured using the following environment variables:
- `DAISY_PAPI_PATH`: Path to the PAPI library. System paths are used if not specified.
- `__DAISY_PAPI_VERSION` (required): PAPI library version (e.g., `0x07020000` for version 7.2.0.0). Must match the major and minor version of the loaded PAPI library, otherwise the RTL is not ready
- `__DAISY_INSTRUMENTATION_MODE`: Set to `aggregate` to activate aggregate mode (default is full tracing)
- `__DAISY_INSTRUMENTATION_FILE`: Output location of the trace file
- `__DAISY_INSTRUMENTATION_FILE_PID_SUFFIX`: Set to `1` to append `.<pid>` to the trace file path, so that multiple processes sharing the same `__DAISY_INSTRUMENTATION_FILE` (e.g. MPI ranks or forked workers) each write their own trace. This also dumps the trace on `SIGTERM`/`SIGINT` (see below)
- `__DAISY_INSTRUMENTATION_EVENTS`: List of PAPI events to record on the CPU. If an event does not exist, the RTL is not ready
- `__DAISY_INSTRUMENTATION_EVENTS_CUDA`: List of PAPI events to record on the GPU. If an event does not exist, the RTL is not ready
- `__DAISY_INSTRUMENTATION_ASYNC_CAPTURE` Enable/disable asynchronous event capturing (on by default for aggregate GPU event recording)

### Processes ending without a normal exit

The trace is written when the process exits normally (returning from `main` or calling `exit`). It is not written when the process ends otherwise:
- **Termination signals:** with `__DAISY_INSTRUMENTATION_FILE_PID_SUFFIX=1`, the RTL handles `SIGTERM` and `SIGINT` by writing the trace, then restoring the previous disposition and re-raising the signal, so default termination or the application's own handler still apply. Handlers are installed when the RTL state is first used: signals the application ignores at that point stay ignored, and handlers the application installs afterwards replace the dump. The dump is best effort: it is skipped if the signal interrupts the RTL itself or another thread does not leave the RTL within one second. Invocations still in progress and unresolved asynchronous GPU events are not part of the dumped trace.
- **`_exit`, e.g. forked Python `multiprocessing` workers (`os._exit`):** call `__daisy_instrumentation_dump()` (`dump_instrumentation()` in the Python bindings) before exiting to write the trace explicitly. It returns false if the trace could not be written or the RTL is not ready.
- `SIGKILL` and crashes always lose the trace.

Captured arguments are written to individual files in the directory specified by `__DAISY_CAPTURE_BASE_DIR`.
The default capturing behavior is defined by `__DAISY_CAPTURE_STRATEGY_DEFAULT`:
- `never`: disable capture
- `once`: capture the first invocation
- `always` capture all invocations

When RPC tuning metadata is available, `docc.rpc_optimization.expected_performance` contains an open-ended object of backend-predicted metrics (currently including `speedup`). Metric keys and JSON values may evolve independently of the RTL schema. `vector_distance` remains a sibling field in `rpc_optimization`. These values are predictions, not runtime-measured performance.
