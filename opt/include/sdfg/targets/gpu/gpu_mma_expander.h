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
    GpuMmaExpander(const GpuArch* arch) : arch_(arch), CodeLibNodeExpander(math::tensor::LibraryNodeType_MatMul) {}
    virtual ~GpuMmaExpander() = default;
    const LibNodeExpander* for_lib_node(const data_flow::LibraryNode& node) const override;

    LibNodeExpander::ExpandOutcome handle_expand(
        LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::MatMulNode& node
    ) const override;

    math::tensor::TensorLayout
    add_to_offset(const math::tensor::TensorLayout& layout, const symbolic::Expression& offset_add) const;

protected:
    virtual bool matches_possible_mma_pattern(const math::tensor::MatMulNode& node) const;
    virtual GpuMmaTiling get_mma_tiling(const symbolic::MultiExpression& res_shape) const;
    virtual ScheduleType get_schedule_type(gpu::TargetLevel dim, const symbolic::Integer& size) const;
};

} // namespace sdfg::gpu
