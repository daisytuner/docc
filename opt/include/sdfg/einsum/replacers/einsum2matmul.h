#pragma once

#include <optional>
#include <string>
#include <vector>

#include <sdfg/passes/expansion/lib_node_expander.h>

#include "sdfg/einsum/einsum_detection.h"
#include "sdfg/einsum/einsum_replacer.h"
#include "sdfg/types/type.h"

namespace sdfg::einsum {

/**
 * @brief Proof-of-concept replacer that turns a matmul-shaped EinsumCluster into a MatMulNode.
 *
 * Mirrors the applicability check of the Einsum2Gemm transformation, but operates on the
 * non-destructive EinsumCluster and materializes a math::tensor::MatMulNode in a fresh
 * sequence + block after the original computation.
 *
 * POC limitations: only the canonical, non-transposed form C[i,j] += A[i,k] * B[k,j] with
 * exactly two multiplicand inputs and zero initialization is supported.
 */
class Einsum2MatMul : public EinsumReplacer {
public:
    std::string name() const override { return "Einsum2MatMul"; }

    math::tensor::TensorLayout build_tensor_layout(
        const EinsumIndexing& indexing,
        bool swapped,
        const symbolic::Expression& outer_dim,
        const symbolic::Expression& inner_dim,
        const symbolic::Expression& linearized_offset
    ) const;

    ReplaceOutcome replace(EinsumReplacementContext& context, const EinsumCluster& cluster) const override;

    /**
     * @brief Checks whether this replacer could be applied to the given cluster.
     *
     * Runs the same applicability analysis as replace() but discards the result.
     */
    virtual bool can_be_applied(const EinsumCluster& cluster) const;

    /// Results of the applicability analysis that are reused by the replacement step.
    struct MatMulAnalysis {
        symbolic::Symbol indvar_outer_1 = SymEngine::null;
        symbolic::Symbol indvar_outer_2 = SymEngine::null;
        symbolic::Symbol indvar_inner = SymEngine::null;
        symbolic::Expression m = SymEngine::null;
        symbolic::Expression n = SymEngine::null;
        symbolic::Expression k = SymEngine::null;
        int a_idx = -1;
        int b_idx = -1;
        std::optional<math::tensor::TensorLayout> layout_a;
        std::optional<math::tensor::TensorLayout> layout_b;
        std::optional<math::tensor::TensorLayout> layout_y;
        /// whether the indices in our contributors array are swapped to how we canonically name them
        /// this is not "transposed" as that will simply depend on the strides associated with the respective indices.
        /// This is just so we use the stride matching the indvars
        types::PrimitiveType input_type = types::PrimitiveType::Void;
        types::PrimitiveType output_type = types::PrimitiveType::Void;
    };

    /**
     * @brief Runs the applicability analysis for the canonical MatMul form.
     */
    bool analyze(const EinsumCluster& cluster, MatMulAnalysis& result) const;
};

} // namespace sdfg::einsum
