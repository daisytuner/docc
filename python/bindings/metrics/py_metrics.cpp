#include "py_metrics.h"

#include <pybind11/stl.h>

#include <string>

#include "docc_metrics.h"

namespace py = pybind11;

void register_metrics(py::module& m) {
    using docc::metrics::DoccMetrics;

    py::class_<DoccMetrics>(m, "DoccMetrics")
        .def(py::init<>())
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
