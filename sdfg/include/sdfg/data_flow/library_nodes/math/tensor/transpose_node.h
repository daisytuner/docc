#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"

#include "sdfg/serializer/json_serializer.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Transpose("ml::Transpose");

class TransposeNode : public TensorNode {
public:
    TransposeNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        const std::vector<int64_t>& perm
    );

    void validate(const Function& function) const override;

    bool expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    const std::vector<symbolic::Expression>& shape() const { return shape_; }

    const std::vector<int64_t>& perm() const { return perm_; }

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_ex, const symbolic::Expression new_ex) override;

    std::unique_ptr<DataFlowNode> clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent)
        const override {
        return std::unique_ptr<DataFlowNode>(new TransposeNode(element_id, debug_info(), vertex, parent, shape_, perm_)
        );
    }

    bool supports_integer_types() const override { return true; }

private:
    std::vector<symbolic::Expression> shape_;
    std::vector<int64_t> perm_;
};

/**
 * @class TransposeNodeSerializer
 * @brief Serializer for TransposeNode
 */
class TransposeNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
