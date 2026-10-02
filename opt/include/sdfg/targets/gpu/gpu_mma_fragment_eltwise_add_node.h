#pragma once

#include "gpu_mma_fragment.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fragment.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

inline data_flow::LibraryNodeCode LibraryNodeType_GpuMmaFragmentEltwiseAdd("gpu::MmaFragmentEltwiseAdd");

/**
 * @brief Handles the addition of two MMA fragments and stores the result in a third fragment.
 *
 * MMA Fragments typically initialize with 0.
 * After the entire matrix has been processed, the original C matrix needs to be added onto that to match the math of a
 * full Matmul. This is a cooperative loop that will do fragD = fragAcc + fragC for each element in the fragment. If we
 * ever need to consider alpha or beta it would also go here. Here fragAcc may have one type (accumulator type) and
 * fragC and fragD can have the type with witch the data is stored in memory.
 */
class GpuMmaFragmentEltwiseAddNode : public data_flow::LibraryNode {
    MmaBlockSize block_size_;
    types::PrimitiveType output_type_;
    types::PrimitiveType acc_type_;

public:
    static constexpr auto FRAG_D_INPUT_IDX = 0;
    static constexpr auto FRAG_ACC_INPUT_IDX = 1;
    static constexpr auto FRAG_C_INPUT_IDX = 2;

    GpuMmaFragmentEltwiseAddNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const MmaBlockSize& block_size,
        types::PrimitiveType output_type,
        types::PrimitiveType acc_type,
        const data_flow::ImplementationType& impl_type
    );

    types::PrimitiveType output_type() const {
        return output_type_;
    }
    types::PrimitiveType acc_type() const {
        return acc_type_;
    }
    const MmaBlockSize& block_size() const {
        return block_size_;
    }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;
};

class GpuMmaFragmentEltwiseAddNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace sdfg::gpu
