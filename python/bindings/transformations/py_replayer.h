#pragma once

#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sdfg/analysis/analysis.h>
#include <sdfg/transformations/replayer.h>

#include "analysis/py_analysis.h"
#include "builder/py_structured_sdfg_builder.h"

namespace py = pybind11;

/**
 * @brief Python wrapper for the Replayer
 *
 * This class allows to replay recorded transformations on an SDFG using a JSON description.
 */
class PyReplayer {
private:
    sdfg::transformations::Replayer replayer_;

public:
    PyReplayer() : replayer_() {
    }

    void apply(PyStructuredSDFGBuilder& builder, PyAnalysisManager& analysis_manager, const std::string& desc) {
        nlohmann::json json_desc = nlohmann::json::parse(desc);
        replayer_.replay(builder.builder(), analysis_manager.manager(), json_desc, false, 0);
    };
};

inline void register_replayer(py::module& m) {
    py::class_<PyReplayer>(m, "Replayer")
        .def(py::init<>(), "Create a Replayer")
        .def(
            "apply",
            &PyReplayer::apply,
            py::arg("builder"),
            py::arg("analysis_manager"),
            py::arg("desc"),
            "Apply a transformation described by the given JSON description to the SDFG using the provided builder and "
            "analysis manager"
        )
        .def("__repr__", [](const PyReplayer&) {
            return "<Replayer>";
        });
}
