#include "sdfg/passes/scheduler/rocm_scheduler.h"

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/collapse_to_depth.h"
#include "sdfg/transformations/offloading/gpu_loop_reordering.h"
#include "sdfg/transformations/offloading/gpu_tiling.h"
#include "sdfg/transformations/offloading/rocm_parallelize_nested_map.h"
#include "sdfg/transformations/offloading/rocm_transform.h"

namespace sdfg {
namespace passes {
namespace scheduler {

SchedulerAction ROCMScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& loop,
    bool offload_unknown_sizes
) {
    if (auto map_node = dynamic_cast<structured_control_flow::Map*>(&loop)) {
        bool applied_collapse = false;
        // transformations::CollapseToDepth collapse_to_depth(*map_node, 2);
        // if (collapse_to_depth.can_be_applied(builder, analysis_manager)) {
        //     collapse_to_depth.apply(builder, analysis_manager);
        //     analysis_manager.invalidate_all();
        //     applied_collapse = true;
        // }
        // auto collapsed_map = collapse_to_depth.outer_loop();
        // Apply ROCM parallelization to the loop
        rocm::ROCMTransform rocm_transform(*map_node, 64, offload_unknown_sizes); // 64 is ROCM default wavefront
                                                                                  // size
        if (rocm_transform.can_be_applied(builder, analysis_manager)) {
            rocm_transform.apply(builder, analysis_manager);


            transformations::GPULoopReordering gpu_loop_reordering_pass(*map_node);
            if (gpu_loop_reordering_pass.can_be_applied(builder, analysis_manager)) {
                gpu_loop_reordering_pass.apply(builder, analysis_manager);
            }

            auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
            auto descendants = loop_analysis.descendants(map_node);
            for (auto& descendant : descendants) {
                if (auto nested_map = dynamic_cast<structured_control_flow::Map*>(descendant)) {
                    transformations::ROCMParallelizeNestedMap nested_rocm_transform(*nested_map, 8);
                    if (nested_rocm_transform.can_be_applied(builder, analysis_manager)) {
                        nested_rocm_transform.apply(builder, analysis_manager);
                    }
                }
            }


            analysis_manager.invalidate_all();
            auto& loop_analysis2 = analysis_manager.get<analysis::LoopAnalysis>();
            for (auto& descendant : loop_analysis2.descendants(map_node)) {
                if (auto target_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(descendant)) {
                    transformations::GPUTiling gpu_tiling_transform(*target_loop, 8);
                    if (gpu_tiling_transform.can_be_applied(builder, analysis_manager)) {
                        gpu_tiling_transform.apply(builder, analysis_manager);
                    }
                }
            }

            analysis_manager.invalidate_all();
            return NEXT;
        }
        if (applied_collapse) {
            return NEXT;
        }
    }

    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);

    // Check if in not outermost loop
    if (loop_info.loopnest_index == -1 || loop_info.num_maps <= 1 || loop_info.is_perfectly_nested ||
        loop_info.has_side_effects) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

SchedulerAction ROCMScheduler::schedule(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::While& loop,
    bool offload_unknown_sizes
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loop_info = loop_analysis.loop_info(&loop);
    // Check if in not outermost loop
    if (loop_info.loopnest_index == -1 || loop_info.has_side_effects) {
        return NEXT;
    } else {
        // Visit 1st-level children
        return CHILDREN;
    }
}

std::unordered_set<ScheduleTypeCategory> ROCMScheduler::compatible_types() { return {ScheduleTypeCategory::None}; }

} // namespace scheduler
} // namespace passes
} // namespace sdfg
