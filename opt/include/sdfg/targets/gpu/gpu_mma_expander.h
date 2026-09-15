#pragma once

#include <sdfg/passes/expansion/lib_node_expander.h>

#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"

namespace sdfg::gpu {

struct GpuMmaTiling {
    int mma_block_m = 0;
    int mma_block_n = 0;
    int mma_block_k = 0;
    int wave_tile_blocks_m = 0;
    int wave_tile_blocks_n = 0;
    int threads_per_mma_block_m = 0;
    int macro_blocks_m = 0;
    int macro_blocks_n = 0;
};

class GpuMmaExpander : public passes::CodeLibNodeExpander<math::tensor::MatMulNode> {
protected:

public:
    GpuMmaExpander() : CodeLibNodeExpander(math::tensor::LibraryNodeType_MatMul) {}
    virtual ~GpuMmaExpander() = default;
    const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const override;

    virtual void set_implementation_type_mma(math::tensor::MatMulNode& mat_mul_node, const GpuMmaTiling& mma_tiling)
        const = 0;

    LibNodeExpander::ExpandOutcome handle_expand(
        LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::MatMulNode& node
    ) const override;

    math::tensor::TensorLayout
    add_to_offset(const math::tensor::TensorLayout& layout, const symbolic::Expression& offset_add) const;

protected:
    virtual bool matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const = 0;
    virtual GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const = 0;
    virtual ScheduleType get_schedule_type(gpu::TargetLevel dim, const symbolic::Integer& size) const = 0;
};

} // namespace sdfg::gpu
