#include "sdfg/data_flow/library_node.h"

#include <string>
#include "sdfg/data_flow/data_flow_graph.h"
#include "sdfg/symbolic/symbolic.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace data_flow {

LibraryNode::LibraryNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    DataFlowGraph& parent,
    const LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const bool side_effect,
    const ImplementationType& implementation_type
)
    : CodeNode(element_id, debug_info, vertex, parent, outputs, inputs), code_(code), side_effect_(side_effect),
      implementation_type_(implementation_type) {}

const LibraryNodeCode& LibraryNode::code() const { return this->code_; };

const ImplementationType& LibraryNode::implementation_type() const { return this->implementation_type_; };

ImplementationType& LibraryNode::implementation_type() { return this->implementation_type_; };

bool LibraryNode::side_effect() const { return this->side_effect_; };

std::string LibraryNode::toStr() const { return std::string(this->code_.value()); }

symbolic::Expression LibraryNode::flop() const { return SymEngine::null; }

PointerAccessType LibraryNode::pointer_access_type(const Memlet& edge) const {
    auto& conn = edge.dst_conn();
    auto idx = std::find(inputs_.begin(), inputs_.end(), conn) - inputs_.begin();
    return pointer_access_type(idx);
}

EdgeRemoveOption LibraryNode::can_remove_out_edge(const data_flow::DataFlowGraph& graph, const Memlet* memlet) const {
    if (graph.out_edges_for_connector(*this, memlet->src_conn()).size() > 1) {
        return EdgeRemoveOption::Trivially;
    } else if (!side_effect_ && outputs_.size() == 1) {
        return EdgeRemoveOption::RemoveNodeAfter;
    } else {
        // cannot remove the last edge per connector in general
        return EdgeRemoveOption::NotRemovable;
    }
}

EdgeRemoveOption LibraryNode::can_remove_in_edge(const data_flow::DataFlowGraph& graph, const Memlet* memlet) const {
    return EdgeRemoveOption::NotRemovable;
}

bool LibraryNode::pointer_use_creates_side_effects(const DataFlowGraph& dataflow, const Function& func) {
    for (int i = 0; i < inputs_.size(); ++i) {
        auto& conn = inputs_.at(i);
        auto* edge = dataflow.in_edge_for_connector(*this, conn);
        if (edge && edge->result_type(func)->type_id() == types::TypeID::Pointer) {
            auto access = pointer_access_type(i);
            if (!access || !access->no_capture() || (access->may_contain_reads() && !access->access_read_pattern()) ||
                (access->may_contain_writes() && !access->access_write_pattern())) {
                return true;
            }
        }
    }
    return false;
}

} // namespace data_flow
} // namespace sdfg
