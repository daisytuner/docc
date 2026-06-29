#pragma once

#include <filesystem>
#include <string>

#include "sdfg/dcov/region.h"

namespace sdfg {
namespace dcov {

/**
 * @brief Annotate a region @ref Module with measured runtime and metrics from a
 *        daisy instrumentation trace.
 *
 * The trace (Chrome-trace JSON, `daisy_trace.json`) carries one event per
 * instrumented region, keyed by `args.docc.element_id`. Both per-invocation
 * (`cat: region`) and aggregated (`cat: aggregated_region`) events are accepted;
 * aggregated events additionally populate `args.metrics` with
 * `{mean, variance, count, min, max}` per metric.
 *
 * Events are matched to regions by @ref Region::element_id, which is frozen at
 * the py4.norm stage — so @p module should be built from the same SDFG the
 * trace references (`args.docc.sdfg_file`).
 *
 * @return number of regions that received a profile.
 * @throws std::runtime_error if @p trace_path cannot be read or parsed.
 */
size_t annotate_with_trace(Module& module, const std::filesystem::path& trace_path);

/**
 * @brief Mark regions that have argument captures on disk.
 *
 * Arg-capture files follow `<name>_inv<inv>_arg<idx>_<in|out>_<element_id>.bin`.
 * For each region whose @ref Region::element_id has matching files, sets
 * @ref Region::has_arg_capture and fills @ref Region::arg_captures with entries
 * like "arg0:in".
 *
 * @return number of regions that have at least one capture.
 */
size_t annotate_with_arg_captures(Module& module, const std::filesystem::path& arg_capture_dir);

/**
 * @brief Read the `args.docc.arg_capture_path` recorded in a trace, if any.
 * @return the path string, or empty if absent/unreadable.
 */
std::string arg_capture_path_from_trace(const std::filesystem::path& trace_path);

} // namespace dcov
} // namespace sdfg
