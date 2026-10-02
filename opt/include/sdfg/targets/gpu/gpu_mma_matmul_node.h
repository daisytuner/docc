#pragma once

#include "sdfg/data_flow/library_node.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_arch.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu {

inline data_flow::LibraryNodeCode LibraryNodeType_GpuMmaMatmul("gpu::GpuMmaMatmul");

/**
 * @brief A single GPU tensor-core matrix-multiply-accumulate (MMA) step.
 *
 * Unlike math::tensor::MatMulNode, this is a code-generation leaf node dedicated to MMA usage.
 * It performs one `Y += A * B` MMA on mma fragments A and B into the accumulator behind its Y input, which is expected
 * to be a preinitialized accumulator fragment (see GpuMmaFillNode).
 *
 * It has no K loop (iterating K is done externally), does not add C, and does not store the result
 * back. Y is an input treated like a pointer to the accumulator storage.
 * So
 * ```
 *   Decl(fragAcc)
 *   GpuMmaFillNode(fragAcc, 0.0, acc_type)
 *   for (int k = 0; k < K; k += mma_k_size) {
 *     Decl(fragA, fragB)
 *     GpuMmaLoad(fragB, A+offset, ...layout)
 *     GpuMmaLoad(fragA, B+offset, ...layout)
 *     GpuMmaNode(fragA, fragB, fragAcc){mma_block_dims, input_type, acc_type}
 *   }
 *   // Load tile of C, add onto it, store fragAcc to ptr & write back to C (can load C into another frag add as frags
 * or write acc to ptr and add manually there)
 * ```
 * would achieve the same as an entire MMA Matmul before
 *
 * Currently reuses TensorLayouts for convenience, but is very strict. They need to have immediate shape, shape must be
 * divisible by mma_block sizes and strides must follow gemm-semantics (there can be one stride that is customizable,
 * the other must be 1 to represent either col-major or row-major with leading dimension). Types must be supported by
 * the underlying hardware
 */
class GpuMmaMatmulNode : public data_flow::LibraryNode {
    MmaBlockSize mma_block_size_;

    types::PrimitiveType input_type_;
    types::PrimitiveType acc_type_;

public:
    static constexpr auto Y_INPUT_IDX = 0;
    static constexpr auto A_INPUT_IDX = 1;
    static constexpr auto B_INPUT_IDX = 2;


    GpuMmaMatmulNode(
        size_t element_id,
        const DebugInfo& debug_info,
        const graph::Vertex vertex,
        data_flow::DataFlowGraph& parent,
        const MmaBlockSize& mma_block_size,
        types::PrimitiveType input_type,
        types::PrimitiveType acc_type,
        const data_flow::ImplementationType& impl_type
    );

    const MmaBlockSize& mma_block_size() const {
        return mma_block_size_;
    }

    types::PrimitiveType input_type() const {
        return input_type_;
    }
    types::PrimitiveType acc_type() const {
        return acc_type_;
    }

    /// M dimension (rows of A / rows of the accumulator).
    symbolic::Expression m() const {
        return symbolic::integer(mma_block_size_.m);
    }
    /// N dimension (columns of B / columns of the accumulator).
    symbolic::Expression n() const {
        return symbolic::integer(mma_block_size_.n);
    }
    /// K dimension (contraction dimension of A and B for a single MMA step).
    symbolic::Expression k() const {
        return symbolic::integer(mma_block_size_.k);
    }

    void validate(const Function& function) const override;

    symbolic::SymbolSet symbols() const override;

    void replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) override;
    void replace(const symbolic::ExpressionMapping& replacements) override;

    std::unique_ptr<data_flow::DataFlowNode>
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const override;

    std::string toStr() const override;

    symbolic::Expression flop() const override;

    data_flow::PointerAccessType pointer_access_type(int input_idx) const override;
};

class GpuMmaMatmulNodeSerializer : public serializer::LibraryNodeSerializer {
public:
    nlohmann::json serialize(const data_flow::LibraryNode& library_node) override;
    data_flow::LibraryNode& deserialize(
        const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
    ) override;
};

} // namespace sdfg::gpu
