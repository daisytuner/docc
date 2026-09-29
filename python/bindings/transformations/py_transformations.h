#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * @brief Register transformation bindings for Python
 *
 * This function registers Python bindings for all SDFG transformations:
 * - LoopTiling: Tile a loop with a given tile size
 * - LoopInterchange: Interchange two nested loops
 * - LoopDistribute: Distribute a loop into multiple loops
 * - LoopSkewing: Skew a loop by a given factor
 * - TileFusion: Fuse tiled loop nests
 * - MapFusion: Fuse adjacent map operations
 * - Recorder: Record transformation history for replay
 *
 * Each transformation provides:
 * - Constructor with appropriate node references
 * - name() -> transformation name
 * - can_be_applied(builder, analysis_manager) -> bool
 * - apply(builder, analysis_manager)
 * - to_json() -> JSON string representation
 *
 * The Recorder provides:
 * - apply(transformation, builder, analysis_manager, skip_if_not_applicable) -> records and applies
 * - save(path) -> save history to file
 * - history -> JSON string of recorded transformations
 */
void register_transformations(py::module& m);
