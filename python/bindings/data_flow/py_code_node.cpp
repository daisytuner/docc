#include "py_code_node.h"

#include <sstream>

#include <sdfg/data_flow/code_node.h>
#include <sdfg/data_flow/library_node.h>
#include <sdfg/data_flow/tasklet.h>

#include "py_data_flow_node.h"

using namespace sdfg::data_flow;

void register_code_node(py::module& m) {
    // CodeNode class (inherits from DataFlowNode)
    py::class_<CodeNode, DataFlowNode>(m, "CodeNode")
        .def_property_readonly(
            "inputs",
            [](const CodeNode& node) -> const std::vector<std::string>& {
                return node.inputs();
            },
            "Get the input connector names"
        )
        .def_property_readonly(
            "outputs",
            [](const CodeNode& node) -> const std::vector<std::string>& {
                return node.outputs();
            },
            "Get the output connector names"
        )
        .def(
            "input",
            [](const CodeNode& node, size_t index) -> const std::string& {
                return node.inputs().at(index);
            },
            py::arg("index"),
            "Get input connector name at index"
        )
        .def(
            "output",
            [](const CodeNode& node, size_t index) -> const std::string& {
                return node.outputs().at(index);
            },
            py::arg("index"),
            "Get output connector name at index"
        )
        .def("__repr__", [](const CodeNode& node) {
            std::ostringstream oss;
            oss << "<CodeNode id=" << node.element_id() << " inputs=[";
            for (size_t i = 0; i < node.inputs().size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << "'" << node.inputs()[i] << "'";
            }
            oss << "] outputs=[";
            for (size_t i = 0; i < node.outputs().size(); ++i) {
                if (i > 0) {
                    oss << ", ";
                }
                oss << "'" << node.outputs()[i] << "'";
            }
            oss << "]>";
            return oss.str();
        });

    // Tasklet class (inherits from CodeNode)
    py::class_<Tasklet, CodeNode>(m, "Tasklet")
        .def_property_readonly("code", &Tasklet::code, "Get the tasklet operation code")
        .def("__repr__", [](const Tasklet& node) {
            std::ostringstream oss;
            oss << "<Tasklet code=" << static_cast<int>(node.code()) << " id=" << node.element_id() << ">";
            return oss.str();
        });

    // LibraryNode class (inherits from CodeNode)
    py::class_<LibraryNode, CodeNode>(m, "LibraryNode")
        .def_property_readonly(
            "code",
            [](const LibraryNode& node) -> std::string {
                return node.code().value();
            },
            "Get the library node operation code"
        )
        .def_property_readonly(
            "implementation_type",
            [](const LibraryNode& node) -> std::string {
                return node.implementation_type().value();
            },
            "Get the implementation type"
        )
        .def_property_readonly("side_effect", &LibraryNode::side_effect, "Check if this node has side effects")
        .def("__repr__", [](const LibraryNode& node) {
            std::ostringstream oss;
            oss << "<LibraryNode code='" << node.code().value() << "' id=" << node.element_id() << ">";
            return oss.str();
        });
}
