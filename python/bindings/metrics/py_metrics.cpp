#include "py_metrics.h"

#include <pybind11/stl.h>

#include <string>

#include "docc_metrics.h"

namespace py = pybind11;

void register_metrics(py::module& m) {
    using docc::metrics::DoccMetrics;

    py::class_<DoccMetrics>(m, "DoccMetrics")
        .def(py::init<>())
        .def("add_target_options", &DoccMetrics::add_target_options, py::arg("target_options"))
        .def(
            "add_frontend_source_info",
            &DoccMetrics::add_frontend_source_info,
            py::arg("frontend"),
            "Add generic info on the source of the SDFG and job, including capturing some env vars"
        )
        .def(
            "capture_env_vars",
            &DoccMetrics::capture_env_vars,
            "Capture some env vars relevant to docc. For example DOCC_CI, DAISY_CI_RUN_NAME, DAISY_CI_STAGE that help "
            "in identifying the outputs"
        )
        .def(
            "add_metric",
            [](DoccMetrics& self, const std::string& key, const py::object& value, const std::string& section) {
                // Accept any Python value (int, float, bool, str, ...) and
                // stringify it so callers do not need to convert manually.
                self.add_metric(key, py::str(value), section);
            },
            py::arg("key"),
            py::arg("value"),
            py::arg("section") = std::string(""),
            "Add a metric. Optionally assign it to a [section]."
        )
        .def(
            "append_to",
            &DoccMetrics::append_to,
            py::arg("output_dir"),
            py::arg("file_name") = std::string("docc_metrics.properties"),
            "Append the collected metrics to a .properties file and return the "
            "written file path."
        );
}
