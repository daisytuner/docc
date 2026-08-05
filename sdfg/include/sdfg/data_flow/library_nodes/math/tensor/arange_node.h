#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Arange("ml::Arange");

class ArangeNode : public TensorNode {
private:
    std::vector<symbolic::Expression> shape_;

public:
    ArangeNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    // In-edge indices: _out is the result buffer pointer (written to); _start, _end, _step are read-only scalar inputs.
    // All four are modelled as in-edges following the ElementWiseDataflowTensorNode convention.
    static auto constexpr RESULT_PTR_IDX = 0; // connector: _out
    static auto constexpr START_IDX = 1; // connector: _start
    static auto constexpr END_IDX = 2; // connector: _end
    static auto constexpr STEP_IDX = 3; // connector: _step

    const std::vector<symbolic::Expression>& shape() const;

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;

    bool supports_integer_types() const override;

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class ArangeNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
