#include "py_memlet.h"

#include <sstream>

#include <sdfg/data_flow/memlet.h>
#include <sdfg/types/type.h>

using namespace sdfg::data_flow;

void register_memlet(py::module& m) {
    // MemletType enum
    py::enum_<MemletType>(m, "MemletType")
        .value("Computational", MemletType::Computational)
        .value("Reference", MemletType::Reference)
        .value("Dereference_Src", MemletType::Dereference_Src)
        .value("Dereference_Dst", MemletType::Dereference_Dst)
        .export_values();

    // Memlet class
    py::class_<Memlet>(m, "Memlet")
        .def_property_readonly("element_id", &Memlet::element_id, "Get the unique element identifier")
        .def_property_readonly(
            "debug_info", &Memlet::debug_info, py::return_value_policy::reference, "Get the debug information"
        )
        .def_property_readonly("type", &Memlet::type, "Get the memlet type")
        .def_property_readonly(
            "src",
            [](const Memlet& memlet) -> const DataFlowNode& {
                return memlet.src();
            },
            py::return_value_policy::reference,
            "Get the source data flow node"
        )
        .def_property_readonly(
            "dst",
            [](const Memlet& memlet) -> const DataFlowNode& {
                return memlet.dst();
            },
            py::return_value_policy::reference,
            "Get the destination data flow node"
        )
        .def_property_readonly("src_conn", &Memlet::src_conn, "Get the source connector name")
        .def_property_readonly("dst_conn", &Memlet::dst_conn, "Get the destination connector name")
        .def_property_readonly(
            "subset",
            [](const Memlet& memlet) {
                std::vector<std::string> result;
                for (const auto& expr : memlet.subset()) {
                    result.push_back(expr->__str__());
                }
                return result;
            },
            "Get the data access subset as string expressions"
        )
        .def_property_readonly(
            "base_type",
            [](const Memlet& memlet) -> const sdfg::types::IType& {
                return memlet.base_type();
            },
            py::return_value_policy::reference,
            "Get the base type of the data"
        )
        .def("__repr__", [](const Memlet& memlet) {
            std::ostringstream oss;
            oss << "<Memlet src=" << memlet.src().element_id() << " dst=" << memlet.dst().element_id();
            if (!memlet.src_conn().empty()) {
                oss << " src_conn='" << memlet.src_conn() << "'";
            }
            if (!memlet.dst_conn().empty()) {
                oss << " dst_conn='" << memlet.dst_conn() << "'";
            }
            oss << " id=" << memlet.element_id() << ">";
            return oss.str();
        });
}
