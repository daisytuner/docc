#include "sdfg/structured_control_flow/control_flow_node.h"

namespace sdfg {
namespace structured_control_flow {

ControlFlowNode::ControlFlowNode(size_t element_id, const DebugInfo& debug_info, ControlFlowNode* parent)
    : Element(element_id, debug_info), parent_(parent) {}

ControlFlowNode* ControlFlowNode::get_parent() { return parent_; };

const ControlFlowNode* ControlFlowNode::get_parent() const { return parent_; }

std::vector<ControlFlowNode*> ControlFlowNode::parent_chain(ControlFlowNode& child) {
    std::vector<ControlFlowNode*> result;
    auto parent = child.get_parent();

    while (parent) {
        result.push_back(parent);
        parent = parent->get_parent();
    }

    return result;
}

} // namespace structured_control_flow
} // namespace sdfg
