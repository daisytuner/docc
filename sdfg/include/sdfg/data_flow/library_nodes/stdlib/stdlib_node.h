#pragma once
#include "sdfg/data_flow/library_node.h"

namespace sdfg::stdlib {

class StdlibNode : public data_flow::LibraryNode {
protected:
    /**
     * @brief Protected constructor for library nodes
     * @param element_id Unique element identifier
     * @param debug_info Debug information for this node
     * @param vertex Graph vertex for this node
     * @param parent Parent dataflow graph
     * @param code Operation code/identifier
     * @param outputs Output connector names
     * @param inputs Input connector names
     * @param side_effect Whether this operation has side effects
     * @param implementation_type Implementation strategy
     */
    StdlibNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const data_flow::LibraryNodeCode& code,
        const std::vector<std::string>& outputs,
        const std::vector<std::string>& inputs,
        const bool side_effect,
        const data_flow::ImplementationType& implementation_type
    );

public:
    [[nodiscard]] symbolic::Expression flop() const override;
};

} // namespace sdfg::stdlib
