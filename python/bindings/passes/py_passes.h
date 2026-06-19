#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

/**
 * @brief Register pass bindings for Python
 *
 * This function registers Python bindings for SDFG passes.
 *
 * Each pass provides:
 * - Default constructor
 * - name() -> pass name
 * - run(builder, analysis_manager) -> bool (True if the SDFG was modified)
 */
void register_passes(py::module& m);
