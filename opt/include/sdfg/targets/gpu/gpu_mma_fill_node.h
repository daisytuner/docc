#pragma once

#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

inline data_flow::LibraryNodeCode LibraryNodeType_GpuMmaFill("gpu::GpuMmaFill");

/**
 * @brief Prepares (declares and zero-initializes) a GPU MMA accumulator fragment.
 *
 * The accumulator lives behind the node's single Y input, which is treated like a pointer to the
 * accumulator storage. After this node runs, the fragment is ready to be accumulated into by one
 * or more GpuMmaNode steps.
 */
class GpuMmaFillNode : public data_flow::LibraryNode {
    MmaBlockSize mma_block_size_;
    types::PrimitiveType fill_type_;

public:
    static constexpr auto Y_INPUT_IDX = 0;

    GpuMmaFillNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const MmaBlockSize& mma_block_size,
        types::PrimitiveType fill_type,
        const data_flow::ImplementationType& impl_type
    );

    const MmaBlockSize& mma_block_size() const {
        return mma_block_size_;
    }
    types::PrimitiveType fill_type() const {
        return fill_type_;
    }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class GpuMmaFillNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace sdfg::gpu
