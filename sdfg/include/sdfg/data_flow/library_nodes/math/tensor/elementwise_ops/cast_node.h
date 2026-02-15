#pragma once

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"

#include "sdfg/codegen/dispatchers/block_dispatcher.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Cast("ml::Cast");

class CastNode : public ElementWiseUnaryNode {
private:
    types::PrimitiveType target_type_;

public:
    CastNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        types::PrimitiveType target_type
    );

    bool expand_operation(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Sequence& body,
        const std::string& input_name,
        const std::string& output_name,
        const types::Tensor& input_type,
        const types::Tensor& output_type,
        const data_flow::Subset& subset
    ) override;

    bool supports_integer_types() const override { return true; }

    types::PrimitiveType target_type() const { return target_type_; }

    void validate(const Function& function) const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;
};

class CastNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
