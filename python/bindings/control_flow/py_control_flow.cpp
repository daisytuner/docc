#include "py_control_flow.h"

#include <sstream>
#include <vector>

#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/for.h>
#include <sdfg/structured_control_flow/if_else.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/return.h>
#include <sdfg/structured_control_flow/sequence.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/structured_control_flow/while.h>

using namespace sdfg::structured_control_flow;

void register_control_flow(py::module& m) {
    // ControlFlowNode base class (abstract)
    py::class_<ControlFlowNode>(m, "ControlFlowNode")
        .def_property_readonly("element_id", &ControlFlowNode::element_id, "Get the unique element identifier")
        .def_property_readonly(
            "debug_info", &ControlFlowNode::debug_info, py::return_value_policy::reference, "Get the debug information"
        )
        .def_property_readonly(
            "parent",
            static_cast<ControlFlowNode* (ControlFlowNode::*) ()>(&ControlFlowNode::get_parent),
            py::return_value_policy::reference,
            "Get the parent node if any"
        )
        .def("__repr__", [](const ControlFlowNode& node) {
            std::ostringstream oss;
            oss << "<ControlFlowNode id=" << node.element_id() << ">";
            return oss.str();
        });

    // Transition class
    py::class_<Transition>(m, "Transition")
        .def_property_readonly("element_id", &Transition::element_id, "Get the unique element identifier")
        .def_property_readonly(
            "assignments",
            [](const Transition& trans) {
                std::unordered_map<std::string, std::string> result;
                for (const auto& [sym, expr] : trans.assignments()) {
                    result[sym->__str__()] = expr->__str__();
                }
                return result;
            },
            "Get the symbol assignments as a dictionary"
        )
        .def_property_readonly("empty", &Transition::empty, "Check if this transition has no assignments")
        .def_property_readonly("size", &Transition::size, "Get the number of assignments")
        .def("__repr__", [](const Transition& trans) {
            std::ostringstream oss;
            oss << "<Transition assignments=" << trans.size() << " id=" << trans.element_id() << ">";
            return oss.str();
        });

    // Sequence class (inherits from ControlFlowNode)
    py::class_<Sequence, ControlFlowNode>(m, "Sequence")
        .def_property_readonly("size", &Sequence::size, "Get the number of children")
        .def(
            "at",
            [](Sequence& seq, size_t i) -> std::pair<ControlFlowNode&, Transition&> { return seq.at(i); },
            py::arg("index"),
            py::return_value_policy::reference,
            "Get child and transition at index"
        )
        .def(
            "child",
            [](Sequence& seq, size_t i) -> ControlFlowNode& { return seq.at(i).first; },
            py::arg("index"),
            py::return_value_policy::reference,
            "Get child node at index"
        )
        .def(
            "transition",
            [](Sequence& seq, size_t i) -> Transition& { return seq.at(i).second; },
            py::arg("index"),
            py::return_value_policy::reference,
            "Get transition at index"
        )
        .def(
            "children",
            [](Sequence& seq) {
                std::vector<ControlFlowNode*> result;
                for (size_t i = 0; i < seq.size(); ++i) {
                    result.push_back(&seq.at(i).first);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all child nodes"
        )
        .def(
            "index",
            [](const Sequence& seq, const ControlFlowNode& child) -> int { return seq.index(child); },
            py::arg("child"),
            "Find the index of a child node"
        )
        .def(
            "__len__", [](const Sequence& seq) { return seq.size(); }, "Get the number of children"
        )
        .def(
            "__getitem__",
            [](Sequence& seq, size_t i) -> ControlFlowNode& {
                if (i >= seq.size()) {
                    throw py::index_error("Sequence index out of range");
                }
                return seq.at(i).first;
            },
            py::arg("index"),
            py::return_value_policy::reference,
            "Get child node at index"
        )
        .def("__repr__", [](const Sequence& seq) {
            std::ostringstream oss;
            oss << "<Sequence children=" << seq.size() << " id=" << seq.element_id() << ">";
            return oss.str();
        });

    // Block class (inherits from ControlFlowNode)
    py::class_<Block, ControlFlowNode>(m, "Block")
        .def_property_readonly(
            "dataflow",
            [](Block& block) -> sdfg::data_flow::DataFlowGraph& { return block.dataflow(); },
            py::return_value_policy::reference,
            "Get the dataflow graph"
        )
        .def("__repr__", [](const Block& block) {
            std::ostringstream oss;
            oss << "<Block id=" << block.element_id() << ">";
            return oss.str();
        });

    // IfElse class (inherits from ControlFlowNode)
    py::class_<IfElse, ControlFlowNode>(m, "IfElse")
        .def_property_readonly("size", &IfElse::size, "Get the number of cases")
        .def(
            "case",
            [](IfElse& ifelse, size_t i) -> Sequence& { return ifelse.at(i).first; },
            py::arg("index"),
            py::return_value_policy::reference,
            "Get case sequence at index"
        )
        .def(
            "condition",
            [](const IfElse& ifelse, size_t i) -> std::string { return ifelse.at(i).second->__str__(); },
            py::arg("index"),
            "Get condition at index as string"
        )
        .def(
            "cases",
            [](IfElse& ifelse) {
                std::vector<Sequence*> result;
                for (size_t i = 0; i < ifelse.size(); ++i) {
                    result.push_back(&ifelse.at(i).first);
                }
                return result;
            },
            py::return_value_policy::reference,
            "Get all case sequences"
        )
        .def(
            "conditions",
            [](const IfElse& ifelse) {
                std::vector<std::string> result;
                for (size_t i = 0; i < ifelse.size(); ++i) {
                    result.push_back(ifelse.at(i).second->__str__());
                }
                return result;
            },
            "Get all conditions as strings"
        )
        .def_property_readonly("is_complete", &IfElse::is_complete, "Check if all cases are covered")
        .def(
            "__len__", [](const IfElse& ifelse) { return ifelse.size(); }, "Get the number of cases"
        )
        .def("__repr__", [](const IfElse& ifelse) {
            std::ostringstream oss;
            oss << "<IfElse cases=" << ifelse.size() << " is_complete=" << (ifelse.is_complete() ? "True" : "False")
                << " id=" << ifelse.element_id() << ">";
            return oss.str();
        });

    // StructuredLoop base class
    py::class_<StructuredLoop, ControlFlowNode>(m, "StructuredLoop")
        .def_property_readonly(
            "indvar",
            [](const StructuredLoop& loop) -> std::string { return loop.indvar()->__str__(); },
            "Get the induction variable"
        )
        .def_property_readonly(
            "init",
            [](const StructuredLoop& loop) -> std::string { return loop.init()->__str__(); },
            "Get the initialization expression"
        )
        .def_property_readonly(
            "update",
            [](const StructuredLoop& loop) -> std::string { return loop.update()->__str__(); },
            "Get the update expression"
        )
        .def_property_readonly(
            "condition",
            [](const StructuredLoop& loop) -> std::string { return loop.condition()->__str__(); },
            "Get the loop condition"
        )
        .def_property_readonly(
            "body",
            [](StructuredLoop& loop) -> Sequence& { return loop.root(); },
            py::return_value_policy::reference,
            "Get the loop body sequence"
        )
        .def("__repr__", [](const StructuredLoop& loop) {
            std::ostringstream oss;
            oss << "<StructuredLoop indvar='" << loop.indvar()->__str__() << "' id=" << loop.element_id() << ">";
            return oss.str();
        });

    // For class (inherits from StructuredLoop)
    py::class_<For, StructuredLoop>(m, "For").def("__repr__", [](const For& loop) {
        std::ostringstream oss;
        oss << "<For indvar='" << loop.indvar()->__str__() << "' init=" << loop.init()->__str__()
            << " condition=" << loop.condition()->__str__() << " id=" << loop.element_id() << ">";
        return oss.str();
    });

    // ScheduleTypeCategory enum
    py::enum_<ScheduleTypeCategory>(m, "ScheduleTypeCategory")
        .value("Offloader", ScheduleTypeCategory::Offloader)
        .value("Parallelizer", ScheduleTypeCategory::Parallelizer)
        .value("Vectorizer", ScheduleTypeCategory::Vectorizer)
        .value("None_", ScheduleTypeCategory::None)
        .export_values();

    // ScheduleType class
    py::class_<ScheduleType>(m, "ScheduleType")
        .def_property_readonly("value", &ScheduleType::value, "Get the schedule type identifier")
        .def_property_readonly("category", &ScheduleType::category, "Get the schedule type category")
        .def_property_readonly(
            "properties", [](const ScheduleType& st) { return st.properties(); }, "Get all schedule properties"
        )
        .def("__repr__", [](const ScheduleType& st) {
            std::ostringstream oss;
            oss << "<ScheduleType value='" << st.value() << "'>";
            return oss.str();
        });

    // Map class (inherits from StructuredLoop)
    py::class_<Map, StructuredLoop>(m, "Map")
        .def_property_readonly(
            "schedule_type", &Map::schedule_type, py::return_value_policy::reference, "Get the scheduling strategy"
        )
        .def("__repr__", [](const Map& loop) {
            std::ostringstream oss;
            oss << "<Map indvar='" << loop.indvar()->__str__() << "' schedule='" << loop.schedule_type().value()
                << "' id=" << loop.element_id() << ">";
            return oss.str();
        });

    // While class (inherits from ControlFlowNode)
    py::class_<While, ControlFlowNode>(m, "While")
        .def_property_readonly(
            "body",
            [](While& w) -> Sequence& { return w.root(); },
            py::return_value_policy::reference,
            "Get the loop body sequence"
        )
        .def("__repr__", [](const While& w) {
            std::ostringstream oss;
            oss << "<While id=" << w.element_id() << ">";
            return oss.str();
        });

    // Break class (inherits from ControlFlowNode)
    py::class_<Break, ControlFlowNode>(m, "Break").def("__repr__", [](const Break& b) {
        std::ostringstream oss;
        oss << "<Break id=" << b.element_id() << ">";
        return oss.str();
    });

    // Continue class (inherits from ControlFlowNode)
    py::class_<Continue, ControlFlowNode>(m, "Continue").def("__repr__", [](const Continue& c) {
        std::ostringstream oss;
        oss << "<Continue id=" << c.element_id() << ">";
        return oss.str();
    });

    // Return class (inherits from ControlFlowNode)
    py::class_<Return, ControlFlowNode>(m, "Return")
        .def_property_readonly("data", &Return::data, "Get the data or constant value being returned")
        .def_property_readonly("type", &Return::type, py::return_value_policy::reference, "Get the return type")
        .def_property_readonly("is_data", &Return::is_data, "Check if returning a data container")
        .def_property_readonly("is_constant", &Return::is_constant, "Check if returning a constant")
        .def("__repr__", [](const Return& r) {
            std::ostringstream oss;
            oss << "<Return data='" << r.data() << "' is_constant=" << (r.is_constant() ? "True" : "False")
                << " id=" << r.element_id() << ">";
            return oss.str();
        });
}
