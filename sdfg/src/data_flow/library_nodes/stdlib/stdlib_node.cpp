#include "sdfg/data_flow/library_nodes/stdlib/stdlib_node.h"

namespace sdfg::stdlib {

StdlibNode::StdlibNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const bool side_effect,
    const data_flow::ImplementationType& implementation_type
)
    : LibraryNode(element_id, debug_info, vertex, parent, code, outputs, inputs, side_effect, implementation_type) {}

symbolic::Expression StdlibNode::flop() const { return symbolic::zero(); }

} // namespace sdfg::stdlib
