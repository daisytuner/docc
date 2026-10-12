/**
 * @file clamp_node.h
 * @brief Clamp tensor node with PyTorch semantics
 */
#pragma once

#include <memory>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_node.h"
#include "sdfg/function.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace math {
namespace tensor {

inline data_flow::LibraryNodeCode LibraryNodeType_Clamp("ml::Clamp");

/** @brief Computes Y = min(max(X, min), max) elementwise, where either bound may be omitted.
 *
 * Inputs are the written tensor Y, the tensor X and the scalar bounds "min" and/or "max". X and Y must match the node
 * shape. The node is expanded into a map nest with per-element branches, so a NaN in X or in a bound propagates to
 * the result like in PyTorch.
 */
class ClampNode : public TensorNode {
private:
    std::vector<symbolic::Expression> shape_;
    bool has_min_;
    bool has_max_;

public:
    ClampNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const std::vector<symbolic::Expression>& shape,
        bool has_min,
        bool has_max,
        const data_flow::ImplementationType& impl_type = data_flow::ImplementationType_NONE
    );

    static constexpr int Y_INPUT_IDX = 0;
    static constexpr int X_INPUT_IDX = 1;

    const std::vector<symbolic::Expression>& shape() const;

    bool has_min() const;

    bool has_max() const;

    int min_input_idx() const;

    int max_input_idx() const;

    void validate(const Function& function) const override;

    bool supports_integer_types() const override;

    passes::LibNodeExpander::ExpandOutcome
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    std::string toStr() const override;

    symbolic::SymbolSet symbols() const override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;

    void replace(const symbolic::ExpressionMapping& replacements) override;
};

class ClampNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;

    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace tensor
} // namespace math
} // namespace sdfg
