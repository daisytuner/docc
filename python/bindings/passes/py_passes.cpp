#include "py_passes.h"

#include <sstream>

#include <sdfg/passes/offloading/sync_condition_propagation.h>
#include <sdfg/passes/pass.h>
#include <sdfg/passes/symbolic/symbol_promotion.h>
#include <sdfg/passes/symbolic/symbol_propagation.h>

#include "analysis/py_analysis.h"
#include "builder/py_structured_sdfg_builder.h"

using namespace sdfg::passes;

void register_passes(py::module& m) {
    // Base Pass class (abstract)
    py::class_<Pass>(m, "Pass")
        .def_property_readonly("name", &Pass::name, "Get the pass name")
        .def(
            "run",
            [](Pass& self, PyStructuredSDFGBuilder& builder, PyAnalysisManager& analysis_manager) {
                return self.run(builder.builder(), analysis_manager.manager());
            },
            py::arg("builder"),
            py::arg("analysis_manager"),
            "Run the pass, returning True if the SDFG was modified"
        );

    // SymbolPropagation pass
    py::class_<SymbolPropagation, Pass>(m, "SymbolPropagation")
        .def(
            py::init<>(),
            "Create a symbol propagation pass.\n\n"
            "Propagates symbolic assignments through the SDFG, replacing symbol\n"
            "uses with their assigned values where appropriate."
        )
        .def("__repr__", [](SymbolPropagation& self) {
            std::ostringstream oss;
            oss << "<SymbolPropagation name='" << self.name() << "'>";
            return oss.str();
        });

    // SymbolPromotion pass
    py::class_<SymbolPromotion, Pass>(m, "SymbolPromotion")
        .def(
            py::init<>(),
            "Create a symbol promotion pass.\n\n"
            "Promotes symbols from dataflow to symbolic expressions."
        )
        .def("__repr__", [](SymbolPromotion& self) {
            std::ostringstream oss;
            oss << "<SymbolPromotion name='" << self.name() << "'>";
            return oss.str();
        });

    // SyncConditionPropagation pass
    py::class_<SyncConditionPropagation, Pass>(m, "SyncConditionPropagation")
        .def(
            py::init<>(),
            "Create a synchronization condition propagation pass.\n\n"
            "Propagates synchronization conditions into GPU-scheduled maps."
        )
        .def("__repr__", [](SyncConditionPropagation& self) {
            std::ostringstream oss;
            oss << "<SyncConditionPropagation name='" << self.name() << "'>";
            return oss.str();
        });
}
