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

} // namespace data_flow
} // namespace sdfg
