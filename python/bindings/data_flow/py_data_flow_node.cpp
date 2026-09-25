#include "py_data_flow_node.h"

#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/data_flow_node.h>
#include <sdfg/types/type.h>

using namespace sdfg::data_flow;

void register_data_flow_node(py::module& m) {
    // DataFlowNode base class (abstract)
    py::class_<DataFlowNode>(m, "DataFlowNode")
        .def_property_readonly("element_id", &DataFlowNode::element_id, "Get the unique element identifier")
        .def_property_readonly(
            "debug_info", &DataFlowNode::debug_info, py::return_value_policy::reference, "Get the debug information"
        )
        .def_property_readonly("side_effect", &DataFlowNode::side_effect, "Check if this node has side effects")
        .def("__repr__", [](const DataFlowNode& node) {
            std::ostringstream oss;
            oss << "<DataFlowNode id=" << node.element_id() << ">";
            return oss.str();
        });

    // AccessNode class
    py::class_<AccessNode, DataFlowNode>(m, "AccessNode")
        .def_property_readonly(
            "data",
            [](const AccessNode& node) -> const std::string& {
                return node.data();
            },
            "Get the name of the data container"
        )
        .def_property_readonly("side_effect", &AccessNode::side_effect, "Check if this node has side effects (writes)")
        .def("__repr__", [](const AccessNode& node) {
            std::ostringstream oss;
            oss << "<AccessNode data='" << node.data() << "' id=" << node.element_id() << ">";
            return oss.str();
        });

    // ConstantNode class
    py::class_<ConstantNode, AccessNode>(m, "ConstantNode")
        .def_property_readonly(
            "type",
            [](const ConstantNode& node) -> const sdfg::types::IType& {
                return node.type();
            },
            py::return_value_policy::reference,
            "Get the type of the constant"
        )
        .def("__repr__", [](const ConstantNode& node) {
            std::ostringstream oss;
            oss << "<ConstantNode value='" << node.data() << "' id=" << node.element_id() << ">";
            return oss.str();
        });
}
