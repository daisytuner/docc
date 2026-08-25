#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <list>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#ifdef __APPLE__
#include <pthread.h>
#endif

#include "daisy_rtl/daisy_rtl.h"

// Cross-platform gettid() wrapper
#ifdef __APPLE__
static inline uint64_t gettid() {
    uint64_t tid;
    pthread_threadid_np(NULL, &tid);
    return tid;
}
#elif defined(__linux__)
#ifndef SYS_gettid
#error "SYS_gettid unavailable on this system"
#endif
#if !defined(__GLIBC__) || !__GLIBC_PREREQ(2, 30)
static inline pid_t gettid() { return syscall(SYS_gettid); }
#endif
#else
#error "Unsupported platform for gettid()"
#endif

#define REGION_CACHE_SIZE 10

// -----------------------------------------------------------------------------
//  Dynamic PAPI loading – we avoid a hard dependency on the library at link time
// -----------------------------------------------------------------------------
namespace {
static int (*_PAPI_library_init)(int) = nullptr;
static int (*_PAPI_create_eventset)(int*) = nullptr;
static int (*_PAPI_cleanup_eventset)(int) = nullptr;
static int (*_PAPI_destroy_eventset)(int*) = nullptr;
static int (*_PAPI_query_named_event)(const char*) = nullptr;
static int (*_PAPI_add_named_event)(int, const char*) = nullptr;
static int (*_PAPI_start)(int) = nullptr;
static int (*_PAPI_stop)(int, long long*) = nullptr;
static int (*_PAPI_reset)(int) = nullptr;
static int (*_PAPI_read)(int, long long*) = nullptr;
static const char* (*_PAPI_strerror)(int) = nullptr;
static long long (*_PAPI_get_real_nsec)(void) = nullptr;

static void load_papi_symbols() {
    const char* papi_lib = std::getenv("DAISY_PAPI_PATH");
    if (!papi_lib) papi_lib = "libpapi.so";
    void* handle = dlopen(papi_lib, RTLD_LAZY);
    if (!handle) {
        std::fprintf(stderr, "[daisy-rtl] Failed to load %s: %s\n", papi_lib, dlerror());
        return; // PAPI not available – we will fall back to time based measurement
    }
    _PAPI_library_init = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_library_init"));
    _PAPI_create_eventset = reinterpret_cast<int (*)(int*)>(dlsym(handle, "PAPI_create_eventset"));
    _PAPI_cleanup_eventset = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_cleanup_eventset"));
    _PAPI_destroy_eventset = reinterpret_cast<int (*)(int*)>(dlsym(handle, "PAPI_destroy_eventset"));
    _PAPI_query_named_event = reinterpret_cast<int (*)(const char*)>(dlsym(handle, "PAPI_query_named_event"));
    _PAPI_add_named_event = reinterpret_cast<int (*)(int, const char*)>(dlsym(handle, "PAPI_add_named_event"));
    _PAPI_start = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_start"));
    _PAPI_stop = reinterpret_cast<int (*)(int, long long*)>(dlsym(handle, "PAPI_stop"));
    _PAPI_reset = reinterpret_cast<int (*)(int)>(dlsym(handle, "PAPI_reset"));
    _PAPI_read = reinterpret_cast<int (*)(int, long long*)>(dlsym(handle, "PAPI_read"));
    _PAPI_strerror = reinterpret_cast<const char* (*) (int)>(dlsym(handle, "PAPI_strerror"));
    _PAPI_get_real_nsec = reinterpret_cast<long long (*)(void)>(dlsym(handle, "PAPI_get_real_nsec"));
}
} // namespace

struct DaisyRegion {
    __daisy_metadata metadata;
    enum __daisy_event_set event_set;
    int papi_eventset = -1;

    // Temporary buffers
    std::vector<long long> last_counts;

    // Full results

    // Runtime
    std::vector<long long> starts;
    std::vector<long long> durations;

    // Counters
    std::vector<std::vector<long long>> counts;

    // Aggregated results

    // Runtime
    long long first_start = 0;
    long long last_start;
    long long runtime_n = 0;
    double runtime_mean = 0.0;
    double runtime_variance = 0.0;
    long long runtime_min;
    long long runtime_max;

    // Counters
    std::vector<long long> n;
    std::vector<double> mean;
    std::vector<double> variance;
    std::vector<long long> min;
    std::vector<long long> max;

    // Static counters
    std::unordered_map<std::string, long long> static_counters_n;
    std::unordered_map<std::string, double> static_counters_mean;
    std::unordered_map<std::string, double> static_counters_variance;
    std::unordered_map<std::string, double> static_counters_min;
    std::unordered_map<std::string, double> static_counters_max;

    // CUDA-event timing: kernels are timed with cudaEvents recorded on the stream
    // (no per-launch host sync), so back-to-back launches stay async and the GPU
    // boosts. `last_cuda_start` holds the enter event; each exit pushes a
    // (start, stop) pair, resolved (synchronize + elapsed) lazily at stats time.
    void* last_cuda_start = nullptr;
    std::vector<std::pair<void*, void*>> cuda_pending;
};

struct JsSafeDouble {
    double value;
    explicit JsSafeDouble(double value) : value(value) {}
};

static std::ostream& operator<<(std::ostream& os, JsSafeDouble val) {
    if (std::isfinite(val.value)) {
        os << val.value;
    } else {
        os << "null";
    }
    return os;
}

class DaisyInstrumentationState {
private:
    size_t next_region_id = 1;
    std::unordered_map<size_t, DaisyRegion> regions;
    std::unordered_map<std::string, size_t> region_name_to_id;

    // Keep a small cache of "sleeping" regions, i.e., after finalize we don't destroy the PAPI eventset immediately
    // This avoids re-creating PAPI eventsets for regions that are frequently re-entered
    std::list<std::string> sleeping_regions_by_name;
    std::unordered_map<std::string, size_t> sleeping_regions_id_lookup;

    bool aggregate_events = false;
    bool papi_available = false;
    std::string output_file;

    // Global measurement switch. When false, enter/exit/metric calls are no-ops
    // so no samples accumulate. Atomic so start/stop don't contend on `mutex` and
    // the hot-path check stays cheap. Shared RTL => one instance toggles all callers.
    std::atomic<bool> enabled{true};

    std::vector<std::string> event_names_cpu;
    std::vector<std::string> event_names_cuda;

    std::mutex mutex;

    // CUDA event API loaded lazily via dlopen (no hard cudart link dependency).
    // When available, CUDA regions are timed with events instead of a host clock
    // + per-launch cudaDeviceSynchronize, so launches stay async and clocks boost.
    bool cuda_events_tried_ = false;
    bool cuda_events_available_ = false;
    void* cudart_handle_ = nullptr;
    int (*cuEventCreate_)(void**) = nullptr;
    int (*cuEventRecord_)(void*, void*) = nullptr;
    int (*cuEventSynchronize_)(void*) = nullptr;
    int (*cuEventElapsedTime_)(float*, void*, void*) = nullptr;
    int (*cuEventDestroy_)(void*) = nullptr;
    std::vector<void*> cuda_event_pool_;

    bool ensure_cuda_events() {
        if (cuda_events_tried_) return cuda_events_available_;
        cuda_events_tried_ = true;
        cudart_handle_ = dlopen("libcudart.so", RTLD_NOW | RTLD_GLOBAL);
        if (!cudart_handle_) cudart_handle_ = dlopen("libcudart.so.12", RTLD_NOW | RTLD_GLOBAL);
        if (!cudart_handle_) return false;
        cuEventCreate_ = reinterpret_cast<int (*)(void**)>(dlsym(cudart_handle_, "cudaEventCreate"));
        cuEventRecord_ = reinterpret_cast<int (*)(void*, void*)>(dlsym(cudart_handle_, "cudaEventRecord"));
        cuEventSynchronize_ = reinterpret_cast<int (*)(void*)>(dlsym(cudart_handle_, "cudaEventSynchronize"));
        cuEventElapsedTime_ =
            reinterpret_cast<int (*)(float*, void*, void*)>(dlsym(cudart_handle_, "cudaEventElapsedTime"));
        cuEventDestroy_ = reinterpret_cast<int (*)(void*)>(dlsym(cudart_handle_, "cudaEventDestroy"));
        cuda_events_available_ = cuEventCreate_ && cuEventRecord_ && cuEventSynchronize_ && cuEventElapsedTime_ &&
                                 cuEventDestroy_;
        return cuda_events_available_;
    }

    void* acquire_cuda_event() {
        if (!cuda_event_pool_.empty()) {
            void* e = cuda_event_pool_.back();
            cuda_event_pool_.pop_back();
            return e;
        }
        void* e = nullptr;
        if (cuEventCreate_(&e) != 0) return nullptr;
        return e;
    }

    // Accumulate one duration into a region's aggregate runtime stats (Welford).
    static void accumulate_runtime(DaisyRegion& region, long long duration_ns) {
        if (region.runtime_n == 0) {
            region.runtime_n = 1;
            region.runtime_mean = static_cast<double>(duration_ns);
            region.runtime_variance = 0.0;
            region.runtime_min = duration_ns;
            region.runtime_max = duration_ns;
        } else {
            region.runtime_n += 1;
            double delta1 = duration_ns - region.runtime_mean;
            region.runtime_mean += delta1 / region.runtime_n;
            double delta2 = duration_ns - region.runtime_mean;
            region.runtime_variance += (delta1 * delta2 - region.runtime_variance) / region.runtime_n;
            if (duration_ns < region.runtime_min) region.runtime_min = duration_ns;
            if (duration_ns > region.runtime_max) region.runtime_max = duration_ns;
        }
    }

    // Synchronize + read every pending CUDA-event pair for `region` and fold the
    // elapsed times into its aggregate runtime stats. Events are returned to the
    // pool for reuse. Called from the stats accessors (once per batch).
    void resolve_cuda_pending(DaisyRegion& region) {
        for (auto& [start, stop] : region.cuda_pending) {
            cuEventSynchronize_(stop);
            float ms = 0.0f;
            cuEventElapsedTime_(&ms, start, stop);
            accumulate_runtime(region, static_cast<long long>(static_cast<double>(ms) * 1.0e6));
            cuda_event_pool_.push_back(start);
            cuda_event_pool_.push_back(stop);
        }
        region.cuda_pending.clear();
    }

    double ns_to_us(double ns) { return ns / 1000; }

    void split_string(const char* str, std::vector<std::string>& out) {
        if (!str) return;
        std::string s(str);
        size_t start = 0, end = 0;
        while ((end = s.find(',', start)) != std::string::npos) {
            out.push_back(s.substr(start, end - start));
            start = end + 1;
        }
        if (s.empty()) return;
        out.push_back(s.substr(start));
    }

    void append_event(FILE* f, DaisyRegion& region, size_t index, const std::vector<std::string>& event_names) {
        // Writes one event as a row in the Chrome trace format

        // Target format:
        // "ph": "X",
        // "cat": "region,daisy",
        // "name": "main  [L110–118]",
        // "pid": 9535,
        // "tid": 0,
        // "ts": 11657,
        // "dur": 613913,
        // "args": {
        // "region_id": "__daisy_correlation18122842100848744318_0_0_1396",
        // "function": "main",
        // "loopnest_index": 0,
        // "module": "correlation",
        // "build_id": "",
        // "source_ranges": [
        //     {
        //     "file": "/home/lukas/repos/docc/tests/polybench/datamining/correlation/correlation.c",
        //     "from": { "line": 110, "col": 17 },
        //     "to":   { "line": 118, "col": 24 }
        //     }
        // ],
        // "metrics": { "CYCLES": 2834021434 } }

        const __daisy_metadata& md = region.metadata;
        std::stringstream entry;
        entry << "{";
        entry << "\"ph\":\"X\",";
        entry << "\"cat\":\"region,daisy\",";
        entry << "\"name\":\"" << md.function_name << " [L" << md.line_begin << "-" << md.line_end << "]\",";
        entry << "\"pid\":" << getpid() << ",";
        entry << "\"tid\":" << gettid() << ",";
        if (region.starts.size() > index) {
            entry << "\"ts\":" << (region.starts.at(index) / 1000) << ",";
        }
        if (region.durations.size() > index) {
            entry << "\"dur\":" << ns_to_us(region.durations.at(index)) << ",";
        }
        entry << "\"args\":{";

        // Source location
        entry << "\"function\":\"" << md.function_name << "\",";
        entry << "\"module\":\"" << std::filesystem::path(md.file_name).filename().string() << "\",";
        entry << "\"source_ranges\":[";
        entry << "{\"file\":\"" << md.file_name << "\",";
        entry << "\"from\":{\"line\":" << md.line_begin << ",\"col\":" << md.column_begin << "},";
        entry << "\"to\":{\"line\":" << md.line_end << ",\"col\":" << md.column_end << "}";
        entry << "}],";

        // docc metadata
        entry << "\"docc\":";
        entry << "{";
        entry << "\"transfer_tuning_session_id\":\""
              << (md.transfer_tuning_session_id ? md.transfer_tuning_session_id : "") << "\",";
        if (md.sdfg_name) {
            entry << "\"sdfg_name\":\"" << md.sdfg_name << "\",";
        } else {
            entry << "\"sdfg_name\":\"\",";
        }
        if (md.sdfg_file) {
            entry << "\"sdfg_file\":\"" << md.sdfg_file << "\",";
        } else {
            entry << "\"sdfg_file\":\"\",";
        }
        if (md.arg_capture_path) {
            entry << "\"arg_capture_path\":\"" << md.arg_capture_path << "\",";
        } else {
            entry << "\"arg_capture_path\":\"\",";
        }
        if (md.features_file) {
            entry << "\"features_file\":\"" << md.features_file << "\",";
        } else {
            entry << "\"features_file\":\"\",";
        }
        if (md.opt_report_file) {
            entry << "\"opt_report_file\":\"" << md.opt_report_file << "\"";
        } else {
            entry << "\"opt_report_file\":\"\"";
        }

        if (md.sdfg_name && md.sdfg_file) {
            entry << ",";

            // Element metadata
            entry << "\"element_id\":" << md.element_id << ",";
            entry << "\"element_type\":\"" << md.element_type << "\",";

            // Deprecated, is also inside loop info
            entry << "\"loopnest_index\":" << md.loopnest_index << ",";

            // Loop info
            entry << "\"loop_info\":{";
            entry << "\"loopnest_index\":" << md.loopnest_index << ",";
            entry << "\"num_loops\":" << md.num_loops << ",";
            entry << "\"num_maps\":" << md.num_maps << ",";
            entry << "\"num_fors\":" << md.num_fors << ",";
            entry << "\"num_whiles\":" << md.num_whiles << ",";
            entry << "\"max_depth\":" << md.max_depth << ",";
            entry << "\"is_perfectly_nested\":" << (md.is_perfectly_nested ? "true" : "false") << ",";
            entry << "\"is_perfectly_parallel\":" << (md.is_perfectly_parallel ? "true" : "false") << ",";
            entry << "\"is_elementwise\":" << (md.is_elementwise ? "true" : "false") << ",";
            entry << "\"has_side_effects\":" << (md.has_side_effects ? "true" : "false");
            entry << "}";
        }
        entry << "},";

        if (md.target_type) {
            entry << "\"target_type\":\"" << md.target_type << "\",";
        } else {
            entry << "\"target_type\":\"\",";
        }

        // Metrics
        entry << "\"metrics\":{";
        if (index < region.counts.size()) {
            auto& counts = region.counts.at(index);
            for (size_t i = 0; i < counts.size(); ++i) {
                entry << "\"" << event_names[i] << "\":" << counts[i];
                if (i < counts.size() - 1) {
                    entry << ",";
                }
            }
        }
        entry << "}";

        entry << "}";
        entry << "}";

        std::fprintf(f, "%s", entry.str().c_str());
    }


    void append_event_aggregated(FILE* f, DaisyRegion& region, const std::vector<std::string>& event_names) {
        // Writes one event as a row in the Chrome trace format

        // Target format:
        // "ph": "X",
        // "cat": "region,daisy",
        // "name": "main  [L110–118]",
        // "pid": 9535,
        // "tid": 0,
        // "ts": 11657,
        // "dur": 613913,
        // "args": {
        // "region_id": "__daisy_correlation18122842100848744318_0_0_1396",
        // "function": "main",
        // "loopnest_index": 0,
        // "module": "correlation",
        // "build_id": "",
        // "source_ranges": [
        //     {
        //     "file": "/home/lukas/repos/docc/tests/polybench/datamining/correlation/correlation.c",
        //     "from": { "line": 110, "col": 17 },
        //     "to":   { "line": 118, "col": 24 }
        //     }
        // ],
        // "metrics": { "CYCLES": 2834021434 } }

        const __daisy_metadata& md = region.metadata;
        std::stringstream entry;
        entry << "{";
        entry << "\"ph\":\"X\",";
        entry << "\"cat\":\"aggregated_region,daisy\",";
        entry << "\"name\":\"" << md.function_name << " [L" << md.line_begin << "-" << md.line_end << "]\",";
        entry << "\"pid\":" << getpid() << ",";
        entry << "\"tid\":" << gettid() << ",";
        // First start
        entry << "\"ts\":" << (region.first_start / 1000) << ",";
        // Total duration
        entry << "\"dur\":" << ns_to_us(region.runtime_mean * region.runtime_n) << ",";
        entry << "\"args\":{";

        // Source location
        entry << "\"function\":\"" << md.function_name << "\",";
        entry << "\"module\":\"" << std::filesystem::path(md.file_name).filename().string() << "\",";
        entry << "\"source_ranges\":[";
        entry << "{\"file\":\"" << md.file_name << "\",";
        entry << "\"from\":{\"line\":" << md.line_begin << ",\"col\":" << md.column_begin << "},";
        entry << "\"to\":{\"line\":" << md.line_end << ",\"col\":" << md.column_end << "}";
        entry << "}],";


        // docc metadata
        entry << "\"docc\":";
        entry << "{";
        entry << "\"transfer_tuning_session_id\":\""
              << (md.transfer_tuning_session_id ? md.transfer_tuning_session_id : "") << "\",";
        if (md.sdfg_name) {
            entry << "\"sdfg_name\":\"" << md.sdfg_name << "\",";
        } else {
            entry << "\"sdfg_name\":\"\",";
        }
        if (md.sdfg_file) {
            entry << "\"sdfg_file\":\"" << md.sdfg_file << "\",";
        } else {
            entry << "\"sdfg_file\":\"\",";
        }
        if (md.arg_capture_path) {
            entry << "\"arg_capture_path\":\"" << md.arg_capture_path << "\",";
        } else {
            entry << "\"arg_capture_path\":\"\",";
        }
        if (md.features_file) {
            entry << "\"features_file\":\"" << md.features_file << "\",";
        } else {
            entry << "\"features_file\":\"\",";
        }
        if (md.opt_report_file) {
            entry << "\"opt_report_file\":\"" << md.opt_report_file << "\"";
        } else {
            entry << "\"opt_report_file\":\"\"";
        }

        if (md.sdfg_name && md.sdfg_file) {
            entry << ",";

            // Element metadata
            entry << "\"element_id\":" << md.element_id << ",";
            entry << "\"element_type\":\"" << md.element_type << "\",";

            // Deprecated, is also inside loop info
            entry << "\"loopnest_index\":" << md.loopnest_index << ",";

            // Loop info
            entry << "\"loop_info\":{";
            entry << "\"loopnest_index\":" << md.loopnest_index << ",";
            entry << "\"num_loops\":" << md.num_loops << ",";
            entry << "\"num_maps\":" << md.num_maps << ",";
            entry << "\"num_fors\":" << md.num_fors << ",";
            entry << "\"num_whiles\":" << md.num_whiles << ",";
            entry << "\"max_depth\":" << md.max_depth << ",";
            entry << "\"is_perfectly_nested\":" << (md.is_perfectly_nested ? "true" : "false") << ",";
            entry << "\"is_perfectly_parallel\":" << (md.is_perfectly_parallel ? "true" : "false") << ",";
            entry << "\"is_elementwise\":" << (md.is_elementwise ? "true" : "false") << ",";
            entry << "\"has_side_effects\":" << (md.has_side_effects ? "true" : "false");
            entry << "}";
        }
        entry << "},";

        if (md.target_type) {
            entry << "\"target_type\":\"" << md.target_type << "\",";
        } else {
            entry << "\"target_type\":\"\",";
        }

        entry << "\"metrics\":{";
        // PAPI counters
        for (size_t i = 0; i < event_names.size(); ++i) {
            entry << "\"" << event_names[i] << "\":{";
            entry << "\"mean\":" << region.mean[i] << ",";
            entry << "\"variance\":" << region.variance[i] << ",";
            entry << "\"count\":" << region.n[i] << ",";
            entry << "\"min\":" << region.min[i] << ",";
            entry << "\"max\":" << region.max[i];
            entry << "}";
            entry << ",";
        }
        // Static counters
        for (auto& [name, mean] : region.static_counters_mean) {
            entry << "\"static:::" << name << "\":{";
            entry << "\"mean\":" << JsSafeDouble(mean) << ", ";
            entry << "\"variance\":" << JsSafeDouble(region.static_counters_variance[name]) << ", ";
            entry << "\"count\":" << region.static_counters_n[name] << ", ";
            entry << "\"min\":" << JsSafeDouble(region.static_counters_min[name]) << ", ";
            entry << "\"max\":" << JsSafeDouble(region.static_counters_max[name]);
            entry << "}";
            entry << ",";
        }

        // Runtime stats
        entry << "\"runtime\":{";
        entry << "\"mean\":" << ns_to_us(region.runtime_mean) << ",";
        entry << "\"variance\":" << ns_to_us(region.runtime_variance) << ",";
        entry << "\"count\":" << region.runtime_n << ",";
        entry << "\"min\":" << ns_to_us(region.runtime_min) << ",";
        entry << "\"max\":" << ns_to_us(region.runtime_max);
        entry << "}";

        entry << "}";
        entry << "}";
        entry << "}";

        std::fprintf(f, "%s", entry.str().c_str());
    }

    void verify_events_exist(const std::vector<std::string>& list) {
        for (const auto& event_name : list) {
            int event_code;
            if (_PAPI_query_named_event(event_name.c_str()) != 0) {
                std::fprintf(stderr, "[daisy-rtl] PAPI event '%s' not found.\n", event_name.c_str());
                exit(EXIT_FAILURE);
            }
        }
    }

public:
    DaisyInstrumentationState() {
        load_papi_symbols();

        this->papi_available = _PAPI_library_init && _PAPI_create_eventset && _PAPI_add_named_event && _PAPI_start &&
                               _PAPI_stop && _PAPI_reset && _PAPI_read && _PAPI_get_real_nsec && _PAPI_strerror &&
                               _PAPI_cleanup_eventset && _PAPI_destroy_eventset;
        if (!this->papi_available) {
            std::fprintf(stderr, "[daisy-rtl] PAPI not available.\n");
            exit(EXIT_FAILURE);
        }

        // Initialise PAPI
        const char* ver_str = std::getenv("__DAISY_PAPI_VERSION");
        if (!ver_str) {
            std::fprintf(stderr, "[daisy-rtl] __DAISY_PAPI_VERSION not set.\n");
            exit(EXIT_FAILURE);
        }
        int ver = std::strtol(ver_str, nullptr, 0);
        int retval = _PAPI_library_init(ver);
        if (retval != ver) {
            std::fprintf(
                stderr, "[daisy-rtl] PAPI init failed: %s.\n", _PAPI_strerror ? _PAPI_strerror(retval) : "unknown"
            );
            exit(EXIT_FAILURE);
        }

        // Aggregate events
        const char* aggregate_events_env = std::getenv("__DAISY_INSTRUMENTATION_MODE");
        if (aggregate_events_env && std::strcmp(aggregate_events_env, "aggregate") == 0) {
            this->aggregate_events = true;
        } else {
            this->aggregate_events = false;
        }

        // Output file
        const char* output_file_env = std::getenv("__DAISY_INSTRUMENTATION_FILE");
        if (!output_file_env) {
            this->output_file = "daisy_trace.json";
        } else {
            this->output_file = output_file_env;
        }

        // Events - CPU
        const char* env_events_cpu = std::getenv("__DAISY_INSTRUMENTATION_EVENTS");
        if (env_events_cpu) {
            split_string(env_events_cpu, this->event_names_cpu);
            verify_events_exist(this->event_names_cpu);
        }

        // Events - CUDA
        const char* env_events_cuda = std::getenv("__DAISY_INSTRUMENTATION_EVENTS_CUDA");
        if (env_events_cuda) {
            split_string(env_events_cuda, this->event_names_cuda);
            verify_events_exist(this->event_names_cuda);
        }
    }

    ~DaisyInstrumentationState() {
        // Cleanup cached regions
        for (auto& region_uuid : this->sleeping_regions_by_name) {
            size_t region_id = this->sleeping_regions_id_lookup[region_uuid];
            auto& region = this->regions[region_id];

            if (region.papi_eventset != -1) {
                if (_PAPI_cleanup_eventset(region.papi_eventset) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Failed to clean up PAPI eventset.\n");
                    exit(EXIT_FAILURE);
                }
                if (_PAPI_destroy_eventset(&region.papi_eventset) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Failed to destroy PAPI eventset.\n");
                    exit(EXIT_FAILURE);
                }
                region.papi_eventset = -1;
            }
        }
        this->sleeping_regions_id_lookup.clear();
        this->sleeping_regions_by_name.clear();

        this->region_name_to_id.clear();

        // Write output file
        FILE* f = std::fopen(this->output_file.c_str(), "w");
        if (!f) {
            std::perror("[daisy-rtl] Failed to open output file");
            exit(EXIT_FAILURE);
        }

        // Output file header
        std::fprintf(f, "{\"traceEvents\":[\n");

        // Output all events
        for (auto iter = regions.begin(); iter != regions.end(); ++iter) {
            auto& region = iter->second;

            // Print one aggregated event
            if (this->aggregate_events) {
                std::vector<std::string> event_names = {};
                if (region.event_set == __DAISY_EVENT_SET_CPU) {
                    event_names = this->event_names_cpu;
                } else if (region.event_set == __DAISY_EVENT_SET_CUDA) {
                    event_names = this->event_names_cuda;
                }
                append_event_aggregated(f, region, event_names);

                if (std::next(iter) != regions.end()) {
                    std::fprintf(f, ",\n");
                }
                continue;
            }

            // Print all individual events
            for (size_t event_index = 0; event_index < region.starts.size(); ++event_index) {
                std::vector<std::string> event_names = {};
                if (region.event_set == __DAISY_EVENT_SET_CPU) {
                    event_names = this->event_names_cpu;
                } else if (region.event_set == __DAISY_EVENT_SET_CUDA) {
                    event_names = this->event_names_cuda;
                }
                append_event(f, region, event_index, event_names);
                if (event_index < region.starts.size() - 1) {
                    std::fprintf(f, ",\n");
                }
            }
            if (std::next(iter) != regions.end()) {
                std::fprintf(f, ",\n");
            }
        }

        // Output file footer
        std::fprintf(f, "\n]}\n");

        // Close output file
        std::fclose(f);
    }

    // Whether any PAPI counters are configured for the given event set. When no
    // events are requested (e.g. the __DAISY_INSTRUMENTATION_EVENTS* env vars are
    // unset), we skip PAPI eventset creation/start entirely: starting an empty
    // eventset fails because PAPI cannot determine the component index.
    bool has_events(enum __daisy_event_set event_set) const {
        if (event_set == __DAISY_EVENT_SET_CPU) {
            return !this->event_names_cpu.empty();
        } else if (event_set == __DAISY_EVENT_SET_CUDA) {
            return !this->event_names_cuda.empty();
        }
        return false;
    }

    void start_region_eventset(int eventset) {
        // No eventset was created (no counters configured) - nothing to start.
        if (eventset == -1) {
            return;
        }
        int rc = _PAPI_start(eventset);
        if (rc != 0) {
            std::fprintf(
                stderr,
                "[daisy-rtl] PAPI_start failed (%s). The requested counter set cannot be "
                "programmed together in one pass; split the counters across separate measurement "
                "groups.\n",
                _PAPI_strerror ? _PAPI_strerror(rc) : "unknown error"
            );
            exit(EXIT_FAILURE);
        }
    }

    size_t register_region(const __daisy_metadata* metadata, enum __daisy_event_set event_set) {
        std::lock_guard<std::mutex> lock(mutex);

        // Check cache first
        if (this->sleeping_regions_id_lookup.find(metadata->region_uuid) != this->sleeping_regions_id_lookup.end()) {
            auto region_id = this->sleeping_regions_id_lookup[metadata->region_uuid];

            // Remove from sleeping regions
            this->sleeping_regions_by_name.remove(metadata->region_uuid);
            this->sleeping_regions_id_lookup.erase(metadata->region_uuid);

            // start region
            auto& region = this->regions[region_id];
            this->start_region_eventset(region.papi_eventset);

            return region_id;
        }
        // Check existing region
        if (this->region_name_to_id.find(metadata->region_uuid) != this->region_name_to_id.end()) {
            size_t region_id = this->region_name_to_id[metadata->region_uuid];
            auto& region = this->regions[region_id];

            if (this->has_events(region.event_set)) {
                if (_PAPI_create_eventset(&region.papi_eventset) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Failed to create PAPI eventset.\n");
                    exit(EXIT_FAILURE);
                }
                if (region.event_set == __DAISY_EVENT_SET_CPU) {
                    for (const auto& ev : this->event_names_cpu) {
                        if (_PAPI_add_named_event(region.papi_eventset, ev.c_str()) != 0) {
                            std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                            exit(EXIT_FAILURE);
                        }
                    }
                } else if (region.event_set == __DAISY_EVENT_SET_CUDA) {
                    for (const auto& ev : this->event_names_cuda) {
                        if (_PAPI_add_named_event(region.papi_eventset, ev.c_str()) != 0) {
                            std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                            exit(EXIT_FAILURE);
                        }
                    }
                }
            }

            // start region
            this->start_region_eventset(region.papi_eventset);

            return region_id;
        }

        DaisyRegion region;
        std::memcpy(&region.metadata, metadata, sizeof(__daisy_metadata));
        region.event_set = event_set;

        if (this->has_events(region.event_set)) {
            if (_PAPI_create_eventset(&region.papi_eventset) != 0) {
                std::fprintf(stderr, "[daisy-rtl] Failed to create PAPI eventset.\n");
                exit(EXIT_FAILURE);
            }
            if (region.event_set == __DAISY_EVENT_SET_CPU) {
                for (const auto& ev : this->event_names_cpu) {
                    if (_PAPI_add_named_event(region.papi_eventset, ev.c_str()) != 0) {
                        std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                        exit(EXIT_FAILURE);
                    }
                }
            } else if (region.event_set == __DAISY_EVENT_SET_CUDA) {
                for (const auto& ev : this->event_names_cuda) {
                    if (_PAPI_add_named_event(region.papi_eventset, ev.c_str()) != 0) {
                        std::fprintf(stderr, "[daisy-rtl] Could not add event %s.\n", ev.c_str());
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }

        size_t region_id = next_region_id++;
        regions[region_id] = region;
        region_name_to_id[region.metadata.region_uuid] = region_id;

        this->start_region_eventset(region.papi_eventset);
        return region_id;
    }

    void enter_region(size_t region_id) {
        // No-op while measurement is globally disabled: skip recording a start so
        // the matching (also skipped) exit leaves per-region buffers consistent.
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);

        auto it = regions.find(region_id);
        if (it == regions.end()) {
            std::fprintf(stderr, "[daisy-rtl] Warning: entering unknown region %zu\n", region_id);
            exit(EXIT_FAILURE);
        }

        DaisyRegion& region = it->second;

        // Save start counters (before timing)
        if (region.event_set == __DAISY_EVENT_SET_CPU && this->event_names_cpu.size() > 0) {
            std::vector<long long> counts(this->event_names_cpu.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());
            region.last_counts = counts;
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA && this->event_names_cuda.size() > 0) {
            std::vector<long long> counts(this->event_names_cuda.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());
            region.last_counts = counts;
        }

        // Save start time
        long long start_ns = _PAPI_get_real_nsec();
        if (this->aggregate_events) {
            if (region.first_start == 0) {
                region.first_start = start_ns;
            }
            region.last_start = start_ns;
        } else {
            region.starts.push_back(start_ns);
        }

        // CUDA regions: record a start event on the stream (async, no host sync)
        // so back-to-back launches stay hot; the paired exit records the stop.
        if (region.event_set == __DAISY_EVENT_SET_CUDA && ensure_cuda_events()) {
            region.last_cuda_start = acquire_cuda_event();
            if (region.last_cuda_start) {
                cuEventRecord_(region.last_cuda_start, nullptr);
            }
        }
    }

    void exit_region(size_t region_id) {
        // No-op while disabled: the paired enter was also skipped, so there is no
        // start to close out and no sample to record.
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);

        auto it = regions.find(region_id);
        if (it == regions.end()) {
            std::fprintf(stderr, "[daisy-rtl] Warning: exiting unknown region %zu\n", region_id);
            exit(EXIT_FAILURE);
        }

        DaisyRegion& region = it->second;

        // CUDA-event timing: record the stop event on the stream and defer the
        // elapsed-time read to stats time. The host clock would only see the async
        // launch, so skip it (and the PAPI-counter path, unused for pure timing).
        if (region.event_set == __DAISY_EVENT_SET_CUDA && cuda_events_available_ && region.last_cuda_start) {
            void* stop = acquire_cuda_event();
            if (stop) {
                cuEventRecord_(stop, nullptr);
                region.cuda_pending.emplace_back(region.last_cuda_start, stop);
            } else {
                cuda_event_pool_.push_back(region.last_cuda_start);
            }
            region.last_cuda_start = nullptr;
            return;
        }

        // Save duration (before counters)
        long long end_ns = _PAPI_get_real_nsec();
        if (this->aggregate_events) {
            long long duration_ns = end_ns - region.last_start;

            // Update aggregated runtime stats
            if (region.runtime_n == 0) {
                region.runtime_n = 1;
                region.runtime_mean = static_cast<double>(duration_ns);
                region.runtime_variance = 0.0;
                region.runtime_min = duration_ns;
                region.runtime_max = duration_ns;
            } else {
                region.runtime_n += 1;
                double delta1 = duration_ns - region.runtime_mean;
                region.runtime_mean += delta1 / region.runtime_n;
                double delta2 = duration_ns - region.runtime_mean;
                region.runtime_variance += (delta1 * delta2 - region.runtime_variance) / region.runtime_n;
                if (duration_ns < region.runtime_min) region.runtime_min = duration_ns;
                if (duration_ns > region.runtime_max) region.runtime_max = duration_ns;
            }
        } else {
            long long duration_ns = end_ns - region.starts.back();
            region.durations.push_back(duration_ns);
        }

        // Save end counters
        if (region.event_set == __DAISY_EVENT_SET_CPU && this->event_names_cpu.size() > 0) {
            std::vector<long long> counts(this->event_names_cpu.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());

            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] -= region.last_counts[i];
            }

            if (!this->aggregate_events) {
                region.counts.push_back(counts);
                return;
            }

            // If aggregating, update mean/variance/min/max
            if (region.mean.empty()) {
                region.n.resize(counts.size(), 0);
                region.mean.resize(counts.size(), 0.0);
                region.variance.resize(counts.size(), 0.0);
                region.min.resize(counts.size(), std::numeric_limits<long long>::max());
                region.max.resize(counts.size(), std::numeric_limits<long long>::min());
            }
            for (size_t i = 0; i < counts.size(); ++i) {
                region.n[i] += 1;

                // Mean/variance using Welford's algorithm

                // delta_1 = x_n - mean_{n-1}
                double delta1 = counts[i] - region.mean[i];

                // mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
                region.mean[i] += delta1 / region.n[i];

                // delta_2 = x_n - mean_n
                double delta2 = counts[i] - region.mean[i];

                // variance_n = variance_{n-1} + (delta_1 * delta_2 - variance_{n-1}) / n
                region.variance[i] += (delta1 * delta2 - region.variance[i]) / region.n[i];

                // Min/max
                if (counts[i] < region.min[i]) region.min[i] = counts[i];
                if (counts[i] > region.max[i]) region.max[i] = counts[i];
            }
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA && this->event_names_cuda.size() > 0) {
            std::vector<long long> counts(this->event_names_cuda.size(), 0);
            _PAPI_read(region.papi_eventset, counts.data());

            for (size_t i = 0; i < counts.size(); ++i) {
                counts[i] -= region.last_counts[i];
            }

            if (!this->aggregate_events) {
                region.counts.push_back(counts);
                return;
            }

            // If aggregating, update mean/variance/min/max
            if (region.mean.empty()) {
                region.n.resize(counts.size(), 0);
                region.mean.resize(counts.size(), 0.0);
                region.variance.resize(counts.size(), 0.0);
                region.min.resize(counts.size(), std::numeric_limits<long long>::max());
                region.max.resize(counts.size(), std::numeric_limits<long long>::min());
            }
            for (size_t i = 0; i < counts.size(); ++i) {
                region.n[i] += 1;

                // Mean/variance using Welford's algorithm

                // delta_1 = x_n - mean_{n-1}
                double delta1 = counts[i] - region.mean[i];

                // mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
                region.mean[i] += delta1 / region.n[i];

                // delta_2 = x_n - mean_n
                double delta2 = counts[i] - region.mean[i];

                // variance_n = variance_{n-1} + (delta_1 * delta_2 - variance_{n-1}) / n
                region.variance[i] += (delta1 * delta2 - region.variance[i]) / region.n[i];

                // Min/max
                if (counts[i] < region.min[i]) region.min[i] = counts[i];
                if (counts[i] > region.max[i]) region.max[i] = counts[i];
            }
        }
    }

    void provided_metric(size_t region_id, const char* name, double value) {
        // No-op while disabled so provided metrics stay aligned with the runtime
        // samples, which are also suppressed.
        if (!enabled.load(std::memory_order_relaxed)) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex);

        auto it = regions.find(region_id);
        if (it == regions.end()) {
            std::fprintf(stderr, "[daisy-rtl] Warning: metric for unknown region %zu\n", region_id);
            exit(EXIT_FAILURE);
        }

        DaisyRegion& region = it->second;

        if (region.static_counters_n.find(name) == region.static_counters_n.end()) {
            // Initialize counter
            region.static_counters_n[name] = 1;
            region.static_counters_mean[name] = value;
            region.static_counters_variance[name] = 0.0;
            region.static_counters_min[name] = value;
            region.static_counters_max[name] = value;
            return;
        }

        // Update static counter
        auto& n = region.static_counters_n[name];
        auto& mean = region.static_counters_mean[name];
        auto& variance = region.static_counters_variance[name];
        auto& min = region.static_counters_min[name];
        auto& max = region.static_counters_max[name];

        n += 1;

        // Mean/variance using Welford's algorithm

        // delta_1 = x_n - mean_{n-1}
        double delta1 = value - mean;

        // mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
        mean += delta1 / n;

        // delta_2 = x_n - mean_n
        double delta2 = value - mean;

        // variance_n = variance_{n-1} + (delta_1 * delta_2 - variance_{n-1}) / n
        variance += (delta1 * delta2 - variance) / n;

        // Min/max
        if (value < min) min = value;
        if (value > max) max = value;
    }

    void finalize(size_t region_id) {
        std::lock_guard<std::mutex> lock(mutex);

        auto it = regions.find(region_id);
        if (it == regions.end()) {
            std::fprintf(stderr, "[daisy-rtl] Warning: finalizing unknown region %zu\n", region_id);
            exit(EXIT_FAILURE);
        }

        DaisyRegion& region = it->second;

        // Stop counters
        if (region.event_set == __DAISY_EVENT_SET_CPU && this->event_names_cpu.size() > 0) {
            std::vector<long long> counts(this->event_names_cpu.size(), 0);
            _PAPI_stop(region.papi_eventset, counts.data());
        } else if (region.event_set == __DAISY_EVENT_SET_CUDA && this->event_names_cuda.size() > 0) {
            std::vector<long long> counts(this->event_names_cuda.size(), 0);
            _PAPI_stop(region.papi_eventset, counts.data());
        }

        // Append to cache
        if (this->sleeping_regions_id_lookup.find(region.metadata.region_uuid) ==
            this->sleeping_regions_id_lookup.end()) {
            // Not in cache, add it
            this->sleeping_regions_id_lookup.insert({region.metadata.region_uuid, region_id});
            this->sleeping_regions_by_name.push_back(region.metadata.region_uuid);
        }

        // Check if cache has grown to large
        if (this->sleeping_regions_by_name.size() > REGION_CACHE_SIZE) {
            std::string evict_region_uuid = this->sleeping_regions_by_name.front();
            this->sleeping_regions_by_name.pop_front();

            size_t evict_region_id = this->sleeping_regions_id_lookup[evict_region_uuid];
            this->sleeping_regions_id_lookup.erase(evict_region_uuid);

            auto& evict_region = this->regions[evict_region_id];

            if (evict_region.papi_eventset != -1) {
                if (_PAPI_cleanup_eventset(evict_region.papi_eventset) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] EFailed to clean up PAPI eventset.\n");
                    exit(EXIT_FAILURE);
                }
                if (_PAPI_destroy_eventset(&evict_region.papi_eventset) != 0) {
                    std::fprintf(stderr, "[daisy-rtl] Failed to destroy PAPI eventset.\n");
                    exit(EXIT_FAILURE);
                }
                evict_region.papi_eventset = -1;
            }
        }
    }

public:
    // Read the running aggregate runtime stats for a region (aggregate mode).
    // Returns false if the region is unknown or has no samples yet. Units:
    // mean in microseconds, variance in microseconds^2.
    bool get_runtime_stats(size_t region_id, double* mean_us, double* variance_us2, long long* count) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = regions.find(region_id);
        if (it == regions.end()) {
            return false;
        }
        DaisyRegion& region = it->second;
        resolve_cuda_pending(region);
        if (region.runtime_n <= 0) {
            return false;
        }
        if (mean_us) *mean_us = region.runtime_mean / 1000.0;
        if (variance_us2) *variance_us2 = region.runtime_variance / 1.0e6;
        if (count) *count = region.runtime_n;
        return true;
    }

    // Aggregate over all regions (sum of means/variances, min count). Mirrors
    // parse_region_runtime summing every event's duration.
    bool get_total_stats(double* mean_us, double* variance_us2, long long* count) {
        std::lock_guard<std::mutex> lock(mutex);
        double sum_mean_ns = 0.0;
        double sum_var_ns2 = 0.0;
        long long min_n = 0;
        bool any = false;
        for (auto& kv : regions) {
            DaisyRegion& region = kv.second;
            resolve_cuda_pending(region);
            if (region.runtime_n <= 0) {
                continue;
            }
            sum_mean_ns += region.runtime_mean;
            sum_var_ns2 += region.runtime_variance;
            min_n = any ? std::min(min_n, region.runtime_n) : region.runtime_n;
            any = true;
        }
        if (!any) {
            return false;
        }
        if (mean_us) *mean_us = sum_mean_ns / 1000.0;
        if (variance_us2) *variance_us2 = sum_var_ns2 / 1.0e6;
        if (count) *count = min_n;
        return true;
    }

    // Clear running aggregate runtime stats for all regions (keeps registration).
    void reset_all_stats() {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& kv : regions) {
            DaisyRegion& region = kv.second;
            region.runtime_n = 0;
            region.runtime_mean = 0.0;
            region.runtime_variance = 0.0;
            region.first_start = 0;
            region.starts.clear();
            region.durations.clear();

            // Discard any pending CUDA events (e.g. the warmup launch) so their
            // timing is dropped consistently with the runtime stats above; the
            // events are recycled into the pool.
            for (auto& [start, stop] : region.cuda_pending) {
                cuda_event_pool_.push_back(start);
                cuda_event_pool_.push_back(stop);
            }
            region.cuda_pending.clear();
            region.last_cuda_start = nullptr;

            // Hardware-counter aggregates, cleared so all metrics drop the
            // warmup sample consistently with the runtime stats above.
            region.counts.clear();
            region.n.clear();
            region.mean.clear();
            region.variance.clear();
            region.min.clear();
            region.max.clear();

            // Static counters.
            region.static_counters_n.clear();
            region.static_counters_mean.clear();
            region.static_counters_variance.clear();
            region.static_counters_min.clear();
            region.static_counters_max.clear();
        }
    }

    // Globally enable/disable measurement. Toggling only takes effect at region
    // boundaries (like reset_all_stats), which is where harnesses call it.
    void set_enabled(bool on) { enabled.store(on, std::memory_order_relaxed); }

    bool is_enabled() const { return enabled.load(std::memory_order_relaxed); }
};

static DaisyInstrumentationState& get_daisy_state() {
    static DaisyInstrumentationState instance;
    return instance;
}

#ifdef __cplusplus
extern "C" {
#endif

size_t __daisy_instrumentation_init(__daisy_metadata* metadata, enum __daisy_event_set event_set) {
    if (!metadata) {
        return 0;
    }
    return get_daisy_state().register_region(metadata, event_set);
}

void __daisy_instrumentation_enter(size_t region_id) { get_daisy_state().enter_region(region_id); }

void __daisy_instrumentation_exit(size_t region_id) { get_daisy_state().exit_region(region_id); }

void __daisy_instrumentation_finalize(size_t region_id) { get_daisy_state().finalize(region_id); }

void __daisy_instrumentation_increment(size_t region_id, const char* name, long long value) {
    get_daisy_state().provided_metric(region_id, name, static_cast<double>(value));
}

void __daisy_instrumentation_metric(size_t region_id, const char* name, double value) {
    get_daisy_state().provided_metric(region_id, name, value);
}

bool __daisy_instrumentation_stats(size_t region_id, double* mean_us, double* variance_us2, long long* count) {
    return get_daisy_state().get_runtime_stats(region_id, mean_us, variance_us2, count);
}

bool __daisy_instrumentation_total_stats(double* mean_us, double* variance_us2, long long* count) {
    return get_daisy_state().get_total_stats(mean_us, variance_us2, count);
}

void __daisy_instrumentation_reset_all(void) { get_daisy_state().reset_all_stats(); }

void __daisy_instrumentation_start(void) { get_daisy_state().set_enabled(true); }

void __daisy_instrumentation_stop(void) { get_daisy_state().set_enabled(false); }

bool __daisy_instrumentation_is_enabled(void) { return get_daisy_state().is_enabled(); }

#ifdef __cplusplus
} // extern "C"
#endif
