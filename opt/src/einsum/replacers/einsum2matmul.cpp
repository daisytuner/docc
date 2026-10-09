#include "sdfg/einsum/replacers/einsum2matmul.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <vector>

#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace sdfg::einsum {

namespace {

int uses_indices(const EinsumIndexing& indices, const symbolic::Symbol& first, const symbolic::Symbol& second) {
    if (indices.contributions.size() != 2) {
        return 0;
    }
    if (symbolic::eq(indices.contributions.at(0).indvar, first)) {
        if (symbolic::eq(indices.contributions.at(1).indvar, second)) {
            return 1;
        }
    } else if (symbolic::eq(indices.contributions.at(0).indvar, second)) {
        if (symbolic::eq(indices.contributions.at(1).indvar, first)) {
            return 2;
        }
    }

    return 0;
}

} // namespace

math::tensor::TensorLayout Einsum2MatMul::build_tensor_layout(
    const EinsumIndexing& indexing,
    bool swapped,
    const symbolic::Expression& outer_dim,
    const symbolic::Expression& inner_dim,
    const symbolic::Expression& outer_offset,
    const symbolic::Expression& inner_offset,
    const symbolic::Expression& external_offset
) const {
    std::vector<symbolic::Expression> shape = {outer_dim, inner_dim};
    symbolic::Expression offset = external_offset;

    // Reorder the contributions so that ordered[d] corresponds to shape[d] (outer, inner).
    std::vector<EinsumIndexContribution> ordered(indexing.contributions.begin(), indexing.contributions.end());
    if (swapped) {
        std::reverse(ordered.begin(), ordered.end());
    }
    auto strides = indexing.get_strides(ordered, shape);

    offset = SymEngine::add({offset, symbolic::mul(outer_offset, strides[0]), symbolic::mul(inner_offset, strides[1])});
    return std::move(math::tensor::TensorLayout(shape, strides, offset));
}

bool Einsum2MatMul::analyze(const EinsumCluster& cluster, MatMulAnalysis& analysis) const {
    // Three loop dimensions (M, N, K), all zero-initialized reductions.
    if (cluster.dims.size() != 3) {
        return false;
    }
    for (int d = 0; d < 3; ++d) {
        auto& dim = cluster.dims.at(d);
        // Prevent triangular access: initial values must not depend on the outer loopvars (technically, should be
        // constant inside the entire loop stack of EinsumCluster)
        if (!symbolic::eq(dim.init, symbolic::zero())) {
            for (int e = 0; e < d; ++e) {
                const auto& other_dim = cluster.dims.at(e);
                if (symbolic::uses(dim.init, other_dim.indvar)) {
                    return false;
                }
            }
        }
    }

    auto& out_indices = cluster.get_output_indexing();

    // Output is a 2D matrix C[outer1, outer2].
    if (out_indices.contributions.size() != 2) {
        return false;
    }
    std::array<size_t, 3> permutation = {0, 1, 2};
    do {
        if (symbolic::eq(out_indices.contributions.at(0).indvar, cluster.dims.at(permutation[0]).indvar) &&
            symbolic::eq(out_indices.contributions.at(1).indvar, cluster.dims.at(permutation[1]).indvar)) {
            analysis.indvar_outer_1 = cluster.dims.at(permutation[0]).indvar;
            analysis.indvar_outer_2 = cluster.dims.at(permutation[1]).indvar;
            analysis.indvar_inner = cluster.dims.at(permutation[2]).indvar;
            analysis.m = symbolic::sub(cluster.dims.at(permutation[0]).bound, cluster.dims.at(permutation[0]).init);
            analysis.n = symbolic::sub(cluster.dims.at(permutation[1]).bound, cluster.dims.at(permutation[1]).init);
            analysis.k = symbolic::sub(cluster.dims.at(permutation[2]).bound, cluster.dims.at(permutation[2]).init);
            analysis.m_init = cluster.dims.at(permutation[0]).init;
            analysis.n_init = cluster.dims.at(permutation[1]).init;
            analysis.k_init = cluster.dims.at(permutation[2]).init;

            break;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    if (analysis.indvar_outer_1.is_null() || analysis.indvar_outer_2.is_null() || analysis.indvar_inner.is_null()) {
        return false;
    }

    // Prevent triangular access: bounds must not depend on the loop indvars.
    for (int d = 0; d < 3; ++d) {
        const auto& dim = cluster.dims.at(d);

        if (symbolic::uses(dim.bound, analysis.indvar_outer_1) || symbolic::uses(dim.bound, analysis.indvar_outer_2) ||
            symbolic::uses(dim.bound, analysis.indvar_inner)) {
            return false;
        }
    }

    // Exactly two multiplicand inputs A and B (no alpha scaling, no subtraction).
    if (cluster.inputs.size() != 2 || cluster.subtraction) {
        return false;
    }

    // Identify A (uses outer1) and B (uses outer2)
    auto& input_indices = cluster.get_input_indexings();
    bool a_swapped = false, b_swapped = false;
    for (size_t i = 0; i < input_indices.size(); ++i) {
        const auto& indices = input_indices.at(i);
        if (indices.contributions.size() != 2) {
            return false;
        }
        if (auto a_use = uses_indices(indices, analysis.indvar_outer_1, analysis.indvar_inner)) {
            analysis.a_idx = static_cast<int>(i);
            a_swapped = a_use == 2;
        } else if (auto b_use = uses_indices(indices, analysis.indvar_inner, analysis.indvar_outer_2)) {
            analysis.b_idx = static_cast<int>(i);
            b_swapped = b_use == 2;
        }
    }
    if (analysis.a_idx == -1 || analysis.b_idx == -1 || analysis.a_idx == analysis.b_idx) {
        return false;
    }

    if (!cluster.output_node) {
        return false;
    }

    // Determine and check the element type from the reduction core.
    for (auto& iedge : cluster.in_edges) {
        analysis.input_type = iedge->base_type().primitive_type();
        break;
    }
    if (analysis.input_type == types::PrimitiveType::Void) {
        return false;
    }
    analysis.output_type = cluster.output_edge->base_type().primitive_type();
    if (analysis.output_type == types::PrimitiveType::Void) {
        return false;
    }


    analysis.layout_a = build_tensor_layout(
        input_indices.at(analysis.a_idx),
        a_swapped,
        analysis.m,
        analysis.k,
        analysis.m_init,
        analysis.k_init,
        cluster.get_linearized_outer_offset(analysis.a_idx)
    );
    analysis.layout_b = build_tensor_layout(
        input_indices.at(analysis.b_idx),
        b_swapped,
        analysis.k,
        analysis.n,
        analysis.k_init,
        analysis.n_init,
        cluster.get_linearized_outer_offset(analysis.b_idx)
    );
    analysis.layout_y = build_tensor_layout(
        out_indices,
        false,
        analysis.m,
        analysis.n,
        analysis.m_init,
        analysis.n_init,
        cluster.get_linearized_outer_offset(-1)
    );

    return true;
}

bool Einsum2MatMul::can_be_applied(const EinsumCluster& cluster) const {
    MatMulAnalysis analysis;
    return this->analyze(cluster, analysis);
}

ReplaceOutcome Einsum2MatMul::replace(EinsumReplacementContext& context, const EinsumCluster& cluster) const {
    using Dir = passes::LibNodeExpander::InputUse;

    // --- Applicability (mirrors Einsum2Gemm, restricted to the canonical MatMul form) ---

    MatMulAnalysis analysis;
    if (!this->analyze(cluster, analysis)) {
        return context.unapplicable();
    }

    // --- Replacement ---

    std::vector<Dir> access_dirs(cluster.inputs.size(), Dir::IndirectRead);
    access_dirs.at(access_dirs.size() - 1) = Dir::IndirectReadWrite;
    auto standalone = context.replacement_requires_access_nodes(access_dirs, true);
    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();
    auto& sequence = standalone->replace_with_sequence();
    auto& block = builder.add_block(sequence, cluster.block->debug_info());

    auto& a_access = standalone->add_indirect_read_access(block, static_cast<size_t>(analysis.a_idx));
    auto& b_access = standalone->add_indirect_read_access(block, static_cast<size_t>(analysis.b_idx));
    auto& y_access = standalone->add_output_access(block, 0);

    auto& matmul_node = builder.add_library_node<math::tensor::MatMulNode>(
        block,
        cluster.block->debug_info(),
        analysis.layout_a.value(),
        analysis.layout_b.value(),
        analysis.input_type,
        &analysis.layout_y.value()
    );

    types::Scalar scalar_type = types::Scalar(analysis.input_type);
    types::Pointer tensor_a(scalar_type);
    types::Pointer tensor_b(scalar_type);
    types::Pointer tensor_y(scalar_type);
    builder.add_computational_memlet(block, y_access, matmul_node, "Y", {}, tensor_y, cluster.block->debug_info());
    builder.add_computational_memlet(block, a_access, matmul_node, "A", {}, tensor_a, cluster.block->debug_info());
    builder.add_computational_memlet(block, b_access, matmul_node, "B", {}, tensor_b, cluster.block->debug_info());

    return standalone->successfully_expanded();
}

} // namespace sdfg::einsum
