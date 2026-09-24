#include <ranges>

#include "sdfg/targets/gpu/gpu_mma_einsum_transform.h"

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/einsum/passes/einsum_passes.h"
#include "sdfg/einsum/transformations/einsum_lift.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/targets/gpu/gpu_mma_expander.h"
#include "sdfg/visitor/structured_sdfg_walker.h"

namespace sdfg::gpu {

bool GpuMmaEinsumTransform::run_internal(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, bool revert_changes
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

    einsum::EinsumTracker state;

    bool made_changes = false;

    auto it = visitor::StructuredSDFGWalker::from_node(outermost_mma_loop_.root());
    auto end = visitor::StructuredSDFGWalker::sequence_exit(outermost_mma_loop_.root());
    for (; it != end; ++it) {
        auto [node, scope] = *it;
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&node)) {
            made_changes |= einsum::EinsumDetectionPass::find_einsum_core_ops(state, builder, analysis_manager, *block);

            std::list<einsum::EinsumNode*> einsum_queue(state.einsum_nodes.begin(), state.einsum_nodes.end());
            made_changes |=
                einsum::EinsumDetectionPass::consume_input_operations(state, builder, analysis_manager, einsum_queue);

            made_changes |=
                einsum::EinsumDetectionPass::consume_surrounding_loops(state, builder, analysis_manager, einsum_queue);
        }
    }

    std::vector<einsum::EinsumNode*> mapped_einsums;
    for (auto& einsum_node : state.einsum_nodes) {
        gpu::GpuMmaExpander expander(arch_);
        if (revert_changes) {
            auto handleable = expander.for_lib_node(*einsum_node);
            if (handleable) {
                this->matched_ = true;
            }
        } else {
            auto outcome = passes::expansion::expand_single_node(
                builder, *static_cast<Block*>(einsum_node->get_parent().get_parent()), *einsum_node, expander
            );
            if (outcome.expanded) {
                mapped_einsums.push_back(einsum_node);
            }
        }
    }

    for (auto ein : mapped_einsums) {
        state.einsum_nodes.erase(ein);
    }

    // revert any other changes
    for (auto* ein : state.einsum_nodes) {
        ein->expand(builder, analysis_manager);
    }

    if (!mapped_einsums.empty()) {
        this->matched_ = true;
    }

    return made_changes;
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

void GpuMmaEinsumTransform::to_json(nlohmann::json& j) const {}

} // namespace sdfg::gpu
