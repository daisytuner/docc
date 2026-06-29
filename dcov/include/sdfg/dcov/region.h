#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/element.h"

namespace sdfg {
namespace dcov {

/**
 * @brief A statement: a conserved unit of computation (tasklet or
 *        library node).
 *
 * Statements are the invariants of horizontal transformations (loop
 * fission/fusion, scheduling, tiling). A region's identity is derived from the
 * set of statements it contains, never from the loop structure itself.
 */
struct Statement {
    std::string statement_key; ///< Stable fingerprint
    std::string op; ///< Operation
    std::string dtype; ///< Result type
    std::vector<std::string> inputs; ///< Input operands
    std::string output; ///< Output operands
};

/**
 * @brief A single aggregated runtime metric (one PAPI counter, FLOPs, runtime).
 *
 * Mirrors the aggregated trace shape `{mean, variance, count, min, max}`.
 */
struct MetricStat {
    std::string name; ///< Metric name (e.g. "perf::CYCLES", "static:::flop", "runtime")
    double mean = 0.0; ///< Mean across invocations
    double min = 0.0; ///< Minimum observed
    double max = 0.0; ///< Maximum observed
    double variance = 0.0; ///< Variance across invocations
    uint64_t count = 0; ///< Number of invocations aggregated
};

/**
 * @brief Measured runtime profile attached to a region from an instrumentation
 *        trace (daisy_trace.json). Present only when a trace event matched the
 *        region's @ref Region::element_id.
 */
struct RuntimeProfile {
    double runtime_us = 0.0; ///< Mean wall-clock duration (microseconds)
    uint64_t invocations = 0; ///< Number of measured invocations
    std::string target_type; ///< Execution target (e.g. "CPU_PARALLEL")
    std::vector<MetricStat> metrics; ///< Aggregated metrics (PAPI counters, FLOPs, runtime)
};

/**
 * @brief A static, addressable program region.
 *
 * Regions form a containment tree (referenced via @ref parent_key). The
 * @ref region_key is a semantic fingerprint that is stable across schedule and
 * target changes; @ref display_key is the human-readable structural path.
 */
struct Region {
    std::string region_key;
    std::string display_key;
    std::string parent_key; ///< Empty for the function root region
    std::string element_type; ///< "function" | "for" | "map" | "reduce" | "while" | "library"
    std::string op_class; ///< Operation class for library regions, else empty
    std::string schedule_type; ///< Loop schedule (e.g. "SEQUENTIAL", "CPU_PARALLEL"); empty for non-loops
    bool instrumentable = false;
    std::string structural_path;
    DebugInfo debug_info; ///< Source provenance (display/rename detection only)
    std::optional<analysis::LoopInfo> loop_info; ///< Loop-nest metadata for loop regions
    std::vector<Statement> statements;

    size_t element_id = 0; ///< SDFG element id (frozen at py4.norm); links to trace events
    std::optional<RuntimeProfile> profile; ///< Measured runtime, if a trace event matched element_id
    bool has_arg_capture = false; ///< Whether arg-capture files exist for this element_id
    std::vector<std::string> arg_captures; ///< Captured args, e.g. "arg0:in", "arg2:out"
};

/**
 * @brief Top-level container: one compiled module/function and its region tree.
 */
struct Module {
    std::string name;
    std::string source_file;
    std::string module_id; ///< hash(source_file, function): config-independent namespace
    std::vector<std::pair<std::string, std::string>> build_config; ///< (key,value) labels
    std::vector<Region> regions; ///< Pre-order; root region first
};

} // namespace dcov
} // namespace sdfg
