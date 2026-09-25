#include <ranges>

#include "sdfg/targets/gpu/gpu_mma_einsum_transform.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/einsum/einsum_detection.h"
#include "sdfg/einsum/einsum_replacer.h"
#include "sdfg/einsum/passes/einsum_passes.h"
#include "sdfg/einsum/replacers/einsum2matmul.h"
#include "sdfg/einsum/transformations/einsum_lift.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/targets/gpu/gpu_mma_expander.h"
#include "sdfg/visitor/structured_sdfg_walker.h"

namespace sdfg::gpu {

GpuMmaEinsumReplacer::GpuMmaEinsumReplacer(const GpuArch* arch) : arch_(arch) {}

bool GpuMmaEinsumReplacer::matches_possible_mma_pattern(const MatMulAnalysis& analysis) const {
    auto* mma_arch = arch_->mma_support();
    if (!mma_arch) {
        return false;
    }

    // basic sanity checks
    auto& dims_a = analysis.layout_a.value();
    auto& dims_b = analysis.layout_b.value();
    if (dims_a.dims() != 2 || dims_b.dims() != 2) {
        return false;
    }

    if (!symbolic::eq(dims_a.get_dim(1), dims_b.get_dim(0))) {
        // K dimension must match
        return false;
    }

    auto layout_a = dims_a.is_2d_col_or_row_major();
    auto layout_b = dims_b.is_2d_col_or_row_major();
    if (layout_a == math::tensor::TensorLayout::LAYOUT_OTHER || layout_b == math::tensor::TensorLayout::LAYOUT_OTHER ||
        analysis.layout_y.value().is_2d_col_or_row_major() == math::tensor::TensorLayout::LAYOUT_OTHER) {
        // only support row-major or col-major layout
        return false;
    }

    auto& m = dims_a.get_dim(0);
    auto& n = dims_b.get_dim(1);
    auto& k = dims_a.get_dim(1);

    auto m_blocks = GpuMmaSupport::get_integer_block_count(m, mma_arch->mma_block_m);
    auto n_blocks = GpuMmaSupport::get_integer_block_count(n, mma_arch->mma_block_n);
    auto k_blocks = GpuMmaSupport::get_integer_block_count(k, mma_arch->mma_block_k);

    if (!m_blocks || !n_blocks || !k_blocks) {
        return false;
    }

    if (!mma_arch->valid_block_counts(mma_arch->mma_block_m, m_blocks, n_blocks, k_blocks)) {
        return false;
    }

    auto input_type = analysis.input_type;
    auto output_type = analysis.output_type;

    return mma_arch->supported_types(input_type, output_type);
}

bool GpuMmaEinsumReplacer::analyze(const einsum::EinsumCluster& cluster, EinsumMmaAnalysis& result) const {
    if (Einsum2MatMul::analyze(cluster, result)) {
        return matches_possible_mma_pattern(result);
    } else {
        return false;
    }
}

einsum::ReplaceOutcome GpuMmaEinsumReplacer::
    replace(einsum::EinsumReplacementContext& context, const einsum::EinsumCluster& cluster) const {
    using Dir = passes::LibNodeExpander::InputUse;

    auto* mma_arch = arch_->mma_support();

    // --- Applicability (mirrors Einsum2Gemm, restricted to the canonical MatMul form) ---

    EinsumMmaAnalysis analysis;
    if (!this->analyze(cluster, analysis)) {
        return context.unapplicable();
    }

    auto mma_tiling = mma_arch->get_mma_tiling({analysis.m, analysis.n, analysis.k});
    auto impl_type = mma_arch->get_matmul_impl_type(*arch_, mma_tiling);
    if (!impl_type) {
        return context.unable();
    }

    // --- Replacement ---

    std::vector<Dir> access_dirs(cluster.inputs.size() + 1, Dir::Scalar);
    auto standalone = context.replacement_requires_access_nodes(access_dirs, true);
    if (!standalone) {
        return context.unable();
    }

    return GpuMmaExpander::expand_mma(
        *standalone,
        *arch_,
        mma_tiling,
        analysis.layout_a.value(),
        analysis.layout_b.value(),
        analysis.layout_y.value(),
        analysis.input_type,
        analysis.output_type,
        impl_type.value()
    );
}

bool GpuMmaEinsumReplacer::can_be_applied(const einsum::EinsumCluster& cluster) const {
    EinsumMmaAnalysis analysis;
    return this->analyze(cluster, analysis);
}

bool GpuMmaEinsumTransform::run_internal(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool verify_only
) {
    auto& sdfg = builder.subject();

    if (!arch_) {
        auto loop_stack = ControlFlowNode::parent_chain(outermost_mma_loop_);
        for (auto entry : std::views::reverse(loop_stack)) {
            if (auto* map = dyn_cast<StructuredLoop*>(entry)) {
                if (map->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                    arch_ = GpuArch::get_from_schedule_type(map->schedule_type());
                    break;
                }
            }
        }
    }

    einsum::EinsumDetection detector;
    detector.run(outermost_mma_loop_);

    std::vector<einsum::EinsumNode*> mapped_einsums;
    for (auto& einsum_node : detector.einsums()) {
        GpuMmaEinsumReplacer repl(arch_);
        // einsum::Einsum2MatMul repl;
        if (verify_only) {
            matched_ |= repl.can_be_applied(*einsum_node);
        } else {
            auto outcome = einsum::replace_einsum_cluster(builder, *einsum_node, repl);
            if (outcome.applied) {
                matched_ = true;
            }
        }
    }

    return matched_;
}

GpuMmaEinsumTransform::GpuMmaEinsumTransform(StructuredLoop& outermoost_mma_loop, const gpu::GpuArch* arch)
    : outermost_mma_loop_(outermoost_mma_loop), arch_(arch) {}

bool GpuMmaEinsumTransform::try_apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return run_internal(builder, analysis_manager, false);
}

void GpuMmaEinsumTransform::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto result = run_internal(builder, analysis_manager, false);
    if (!result) {
        throw std::runtime_error("GpuMmaEinsumTransform failed to apply");
    }
}

bool GpuMmaEinsumTransform::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return run_internal(builder, analysis_manager, true);
}

void GpuMmaEinsumTransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    if (arch_) {
        j["parameters"]["arch"] = arch_->unique_id();
    }
    j["subgraph"] = nlohmann::json::object();
}

} // namespace sdfg::gpu
