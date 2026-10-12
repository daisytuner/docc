#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_mma_fragment.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

inline data_flow::LibraryNodeCode LibraryNodeType_GpuMmaFragmentStore("gpu::MmaFragmentStore");

/**
 * @brief Stores a single MMA fragment to memory.
 *
 * Writes the fragment behind its "frag" input to the base pointer behind its "ptr" input (following
 * the given MmaFromMemoryLayout). Both inputs are treated like pointers.
 */
class GpuMmaFragmentStoreNode : public data_flow::LibraryNode {
    MmaBlockSize block_size_;
    MmaFragmentType fragment_type_;
    GpuMmaFromMemoryLayout layout_;
    types::PrimitiveType element_type_;

public:
    static constexpr auto FRAG_INPUT_IDX = 0;
    static constexpr auto PTR_INPUT_IDX = 1;

    GpuMmaFragmentStoreNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const MmaBlockSize& block_size,
        MmaFragmentType fragment_type,
        const GpuMmaFromMemoryLayout& layout,
        types::PrimitiveType element_type,
        const data_flow::ImplementationType& impl_type
    );

    const GpuMmaFromMemoryLayout& layout() const {
        return layout_;
    }
    const MmaBlockSize& block_size() const {
        return block_size_;
    }
    MmaFragmentType fragment_type() const {
        return fragment_type_;
    }
    types::PrimitiveType element_type() const {
        return element_type_;
    }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;

    bool can_relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) const override;
    bool relocalize_operand(int input_idx, const math::tensor::TensorLayout& packed) override;
};

class GpuMmaFragmentStoreNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace sdfg::gpu
