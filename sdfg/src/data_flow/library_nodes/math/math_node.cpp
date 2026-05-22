#include "sdfg/data_flow/library_nodes/math/math_node.h"

namespace sdfg {
namespace math {

MathNode::MathNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::LibraryNodeCode& code,
    const std::vector<std::string>& outputs,
    const std::vector<std::string>& inputs,
    const data_flow::ImplementationType& implementation_type,
    bool side_effects
)
    : LibraryNode(element_id, debug_info, vertex, parent, code, outputs, inputs, side_effects, implementation_type) {}

} // namespace math
} // namespace sdfg
