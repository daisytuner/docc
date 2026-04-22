#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/cutouts/cutouts.h>

#include "analysis/py_analysis.h"
#include "builder/py_structured_sdfg_builder.h"
#include "py_structured_sdfg.h"

namespace py = pybind11;

/**
 * @brief Creates a cutout of a control flow node from an SDFG.
 *
 * A cutout extracts a specific control flow node (e.g., a loop) from an SDFG
 * and creates a new standalone SDFG containing just that node with all
 * necessary data dependencies resolved.
 *
 * @param builder The builder to use for constructing the cutout SDFG
 * @param analysis_manager The analysis manager for the source SDFG
 * @param node The control flow node to extract
 * @return A new PyStructuredSDFG containing the cutout
 */
inline PyStructuredSDFG cutout(
    PyStructuredSDFGBuilder& builder,
    PyAnalysisManager& analysis_manager,
    sdfg::structured_control_flow::ControlFlowNode& node
) {
    auto result = sdfg::util::cutout(builder.builder(), analysis_manager.manager(), node);
    return PyStructuredSDFG::from_sdfg(builder.docc_context(), std::move(result));
}

/**
 * @brief Register the cutout utility function
 */
inline void register_cutout(py::module& m) {
    m.def(
        "cutout",
        &cutout,
        py::arg("builder"),
        py::arg("analysis_manager"),
        py::arg("node"),
        R"doc(
Create a cutout of a control flow node from an SDFG.

A cutout extracts a specific control flow node (e.g., a loop) from an SDFG
and creates a new standalone SDFG containing just that node with all
necessary data dependencies resolved.

Args:
    builder: The StructuredSDFGBuilder to use for constructing the cutout SDFG.
             This should be a fresh builder with only the name set.
    analysis_manager: The AnalysisManager for the source SDFG.
    node: The control flow node to extract (e.g., a For loop, Map, or While).

Returns:
    A new StructuredSDFG containing the cutout.

Example:
    >>> sdfg = StructuredSDFG.from_file("program.sdfg")
    >>> analysis = AnalysisManager(sdfg)
    >>> loop_analysis = analysis.loop_analysis()
    >>> loops = loop_analysis.loops()
    >>> # Create a cutout of the first loop
    >>> builder = StructuredSDFGBuilder("cutout_sdfg")
    >>> cutout_sdfg = cutout(builder, analysis, loops[0])
)doc"
    );
}
