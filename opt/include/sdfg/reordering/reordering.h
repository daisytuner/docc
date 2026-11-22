#pragma once

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/passes/dataflow/dead_data_elimination.h>
#include <sdfg/passes/dataflow/tasklet_fusion.h>
#include <sdfg/passes/debug_dump_pass.h>
#include <sdfg/passes/pipeline.h>
#include <sdfg/passes/redundant_load_elimination_pass.h>
#include <sdfg/passes/structured_control_flow/block_fusion.h>
#include <sdfg/passes/structured_control_flow/dead_cfg_elimination.h>
#include <sdfg/structured_sdfg.h>

#include "sdfg/reordering/passes/loop_fusion/loop_fusion_pass.h"
#include "sdfg/reordering/passes/map_fusion.h"
#include "sdfg/reordering/passes/perfect_loop_distribution.h"
#include "sdfg/reordering/passes/stride_minimization.h"

namespace sdfg {
namespace reordering {

inline passes::Pipeline loop_normalization() {
    passes::Pipeline pipeline("Loop Normalization");

    // Register passes for loop normalization
    pipeline.register_pass<PerfectLoopDistributionPass>();
    pipeline.register_pass<StrideMinimization>();

    return pipeline;
}

inline passes::Pipeline stride_minimization() {
    passes::Pipeline pipeline("Stride Minimization");

    pipeline.register_pass<StrideMinimization>();

    return pipeline;
}

inline passes::Pipeline map_fusion(bool allow_init_hoist = true, bool allow_prod_into_cons = true) {
    passes::Pipeline p("MapFusion");

    p.register_pass<MapFusionPass>(allow_init_hoist, allow_prod_into_cons);
    p.register_pass<passes::BlockFusionPass>();
    p.register_pass<passes::DeadDataElimination>(true);
    p.register_pass<passes::DeadCFGElimination>(true);

    return p;
}

/**
 * Normalize a StructuredSDFG in place.
 *
 * Runs the loop-normalization pipeline. When `enable_fusion` is set, also performs
 * map fusion before and after loop normalization.
 *
 * This function performs:
 * 1. (Optional) Initial map fusion without init-into-reduction hoisting
 * 2. Loop distribution and stride minimization
 * 3. (Optional) Final map fusion with init-into-reduction hoisting
 */
inline void normalize(sdfg::StructuredSDFG& sdfg, bool enable_fusion = true) {
    ::sdfg::passes::CompileStatistics::enter_stage_if_enabled("normalize");
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);

    if (!enable_fusion) {
        // Minimize strides and distribute in the sdfg to get a normal form
        auto pipeline = loop_normalization();
        pipeline.run(builder, analysis_manager);
    } else {
        // Minimize strides in the sdfg and fuse to get a normal form
        auto pipeline = stride_minimization();
        pipeline.run(builder, analysis_manager);

        passes::Pipeline dce = passes::Pipeline::dead_code_elimination();
        passes::DeadDataElimination dde;

        // New Map Fusion, simpler than previous, but what it can do should be cheaper to do
        loop_fusion::LoopFusionPass loop_fusion_pass({.allow_init_hoist = true});
        loop_fusion_pass.run(builder, analysis_manager);

        // Cleanup of artifacts of MapFusion
        dde.run(builder, analysis_manager);
        dce.run(builder, analysis_manager);
        passes::Pipeline block_fusion("BlockFusion");
        block_fusion.register_pass<passes::BlockFusionPass>();
        block_fusion.run(builder, analysis_manager);

        passes::RedundantLoadEliminationPass rle;
        rle.run(builder, analysis_manager);
        dde.run(builder, analysis_manager);
        passes::TaskletFusionPass task_fuse_pass;
        task_fuse_pass.run(builder, analysis_manager);

        // Fuse maps (final run: allow init-into-reduction hoisting now that distribution is done)
        auto map_fusion_hoist = map_fusion(true, false);
        map_fusion_hoist.run(builder, analysis_manager);
    }
    ::sdfg::passes::CompileStatistics::exit_stage_if_enabled();
}

} // namespace reordering
} // namespace sdfg
