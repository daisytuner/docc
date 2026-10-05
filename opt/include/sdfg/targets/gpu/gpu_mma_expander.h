#pragma once

#include <sdfg/passes/expansion/lib_node_expander.h>

#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"

namespace sdfg::gpu {

class GpuMmaExpander : public passes::CodeLibNodeExpander<math::tensor::MatMulNode> {
protected:
    const GpuArch* arch_;

public:
    GpuMmaExpander(const GpuArch* arch) : arch_(arch), CodeLibNodeExpander(math::tensor::LibraryNodeType_MatMul) {
    }
    virtual ~GpuMmaExpander() = default;
    const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const override;

    static void create_fragment_load(
        passes::LibNodeExpander::AccessNodeExpand& standalone,
        types::PrimitiveType input_type,
        const data_flow::ImplementationType& impl_type,
        const DebugInfo& org_debug_info,
        builder::StructuredSDFGBuilder& builder,
        std::string frag_name,
        const MmaBlockSize& mma_block_size,
        MmaFragmentType frag_type,
        const GpuMmaFromMemoryLayout& from_memory,
        unsigned input_index,
        Block& block
    );

    static void create_fragment_store(
        passes::LibNodeExpander::AccessNodeExpand& standalone,
        types::PrimitiveType output_type,
        const data_flow::ImplementationType& impl_type,
        const DebugInfo& org_debug_info,
        builder::StructuredSDFGBuilder& builder,
        std::string frag_name,
        const MmaBlockSize& mma_block_size,
        MmaFragmentType frag_type,
        const GpuMmaFromMemoryLayout& to_memory,
        unsigned input_index,
        Block& block
    );

    LibNodeExpander::ExpandOutcome handle_expand(
        LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::MatMulNode& node
    ) const override;

    static void create_eltwise_add_block(
        AccessNodeExpand& standalone,
        const MmaBlockSize& mma_block_size,
        types::PrimitiveType acc_type,
        types::PrimitiveType output_type,
        const data_flow::ImplementationType& impl_type,
        const DebugInfo& debug_info,
        builder::StructuredSDFGBuilder& builder,
        const std::string& c_frag_name,
        const std::string& acc_frag_name,
        const std::string& store_frag_name,
        Block& block
    );

    static passes::LibNodeExpander::ExpandOutcome expand_mma_standalone(
        LibNodeExpander::AccessNodeExpand& standalone,
        const GpuArch& arch,
        GpuMmaTiling& mma_tiling,
        const math::tensor::TensorLayout& layout_a,
        const math::tensor::TensorLayout& layout_b,
        const math::tensor::TensorLayout& layout_y,
        types::PrimitiveType input_type,
        types::PrimitiveType acc_type,
        types::PrimitiveType output_type,
        const data_flow::ImplementationType& impl_type,
        bool include_c_add,
        const DebugInfo& org_debug_info,
        const std::array<int, 3>& args_order // indices of inputs in order of {y, a, b}
    );

    static void create_fragment_mma(
        LibNodeExpander::AccessNodeExpand& standalone,
        const GpuArch& arch,
        GpuMmaTiling& mma_tiling,
        types::PrimitiveType input_type,
        const std::string& frag_a,
        const std::string& frag_b,
        types::PrimitiveType acc_type,
        const std::string& frag_acc,
        const data_flow::ImplementationType& impl_type,
        const DebugInfo& org_debug_info,
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block
    );

protected:
    virtual bool matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const;
    virtual GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const;
};

} // namespace sdfg::gpu
