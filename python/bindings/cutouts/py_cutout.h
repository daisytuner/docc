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
    PyStructuredSDFG& sdfg, PyAnalysisManager& analysis_manager, sdfg::structured_control_flow::ControlFlowNode& node
) {
    auto result = sdfg::util::cutout(sdfg.sdfg(), analysis_manager.manager(), node);
    return PyStructuredSDFG::from_sdfg(sdfg.docc_context(), std::move(result));
}

inline void apply_cutout(PyStructuredSDFG& sdfg, size_t element_id, PyStructuredSDFG& cutout_sdfg) {
    sdfg::builder::StructuredSDFGBuilder builder(sdfg.sdfg());
    sdfg::util::apply_cutout(builder, element_id, cutout_sdfg.sdfg());
}

/**
 * @brief Register the cutout utility function
 */
inline void register_cutout(py::module& m) {
    m.def(
        "cutout",
        &cutout,
        py::arg("sdfg"),
        py::arg("analysis_manager"),
        py::arg("node"),
        R"doc(
Create a cutout of a control flow node from an SDFG.

A cutout extracts a specific control flow node (e.g., a loop) from an SDFG
and creates a new standalone SDFG containing just that node with all
necessary data dependencies resolved.

Args:
    sdfg: The StructuredSDFG to use for constructing the cutout SDFG.
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
    >>> cutout_sdfg = cutout(sdfg, analysis, loops[0])
)doc"
    );

    m.def(
        "apply_cutout",
        &apply_cutout,
        py::arg("sdfg"),
        py::arg("element_id"),
        py::arg("cutout_sdfg"),
        R"doc(
Merge an (optimized) cutout back into ``sdfg``, in place.

Replaces the control-flow node with ``element_id`` by the children of
``cutout_sdfg``'s root. Arguments and externals of the cutout keep their names
so they bind to the target's containers; colliding transients are renamed, and
containers missing from the target are added as transients, so the target's
signature is preserved. Merged nodes receive fresh element ids.

Args:
    sdfg: The StructuredSDFG to merge into (mutated in place).
    element_id: Element id of the control-flow node to replace.
    cutout_sdfg: The cutout to merge; its transients may be renamed.

Raises:
    RuntimeError: No node with ``element_id`` exists, or it has no parent scope.
)doc"
    );
}
