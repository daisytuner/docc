#include "sdfg/data_flow/code_node.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_graph.h"

namespace sdfg {
namespace data_flow {

CodeNode::CodeNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs
)
    : DataFlowNode(element_id, debug_info, vertex, parent), outputs_(outputs), inputs_(inputs) {};

void CodeNode::validate(const Function& function) const {
    auto& graph = this->get_parent();

    // No two access nodes for same data
    std::unordered_set<std::string> in_conns;
    std::unordered_map<std::string, const AccessNode*> input_names;
    for (auto& iedge : graph.in_edges(*this)) {
        if (std::find(this->inputs().begin(), this->inputs().end(), iedge.dst_conn()) == this->inputs().end()) {
            throw InvalidSDFGException("Input edge " + iedge.dst_conn() + " does not match any input of the code node");
        }
        in_conns.insert(iedge.dst_conn());

        if (dynamic_cast<const ConstantNode*>(&iedge.src()) != nullptr) {
            continue;
        }
        auto& src = static_cast<const AccessNode&>(iedge.src());
        if (input_names.find(src.data()) != input_names.end()) {
            if (input_names.at(src.data()) != &src) {
                throw InvalidSDFGException("Two access nodes with the same data as iedge: " + src.data());
            }
        } else {
            input_names.insert({src.data(), &src});
        }
    }
    for (auto& input : this->inputs()) {
        if (in_conns.find(input) == in_conns.end()) {
            throw InvalidSDFGException("Input " + input + " of code node is not connected to any edge");
        }
    }

    std::unordered_set<std::string> out_conns;
    std::unordered_map<std::string, const AccessNode*> output_names;
    for (auto& oedge : graph.out_edges(*this)) {
        if (std::find(this->outputs().begin(), this->outputs().end(), oedge.src_conn()) == this->outputs().end()) {
            throw InvalidSDFGException("Output edge " + oedge.src_conn() + " does not match any output of the code node");
        }
        out_conns.insert(oedge.src_conn());

        if (dynamic_cast<const ConstantNode*>(&oedge.dst()) != nullptr) {
            continue;
        }
        auto& dst = static_cast<const AccessNode&>(oedge.dst());
        if (output_names.find(dst.data()) != output_names.end()) {
            if (output_names.at(dst.data()) != &dst) {
                throw InvalidSDFGException("Two access nodes with the same data as oedge: " + dst.data());
            }
        } else {
            output_names.insert({dst.data(), &dst});
        }
    }
    for (auto& output : this->outputs()) {
        if (out_conns.find(output) == out_conns.end()) {
            throw InvalidSDFGException("Output " + output + " of code node is not connected to any edge");
        }
    }
}

const std::vector<std::string>& CodeNode::outputs() const { return this->outputs_; };

const std::vector<std::string>& CodeNode::inputs() const { return this->inputs_; };

std::vector<std::string>& CodeNode::outputs() { return this->outputs_; };

std::vector<std::string>& CodeNode::inputs() { return this->inputs_; };

const std::string& CodeNode::output(size_t index) const { return this->outputs_[index]; };

const std::string& CodeNode::input(size_t index) const { return this->inputs_[index]; };

bool CodeNode::has_constant_input(size_t index) const {
    for (auto& iedge : this->get_parent().in_edges(*this)) {
        if (iedge.dst_conn() == this->inputs_[index]) {
            if (dynamic_cast<const ConstantNode*>(&iedge.src())) {
                return true;
            }
        }
    }

    return false;
}

} // namespace data_flow
} // namespace sdfg
