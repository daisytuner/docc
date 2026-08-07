#include "sdfg/passes/normalization/normalize.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/tasklet_fusion.h"
#include "sdfg/passes/debug_dump_pass.h"
#include "sdfg/passes/loop_fusion/loop_fusion_pass.h"
#include "sdfg/passes/normalization/normalization.h"
#include "sdfg/passes/redundant_load_elimination_pass.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace passes {
namespace normalization {

void normalize(sdfg::StructuredSDFG& sdfg, bool enable_fusion) {
    CompileStatistics::enter_stage_if_enabled("normalize");
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

        Pipeline dce = Pipeline::dead_code_elimination();
        DeadDataElimination dde;

        // New Map Fusion, simpler than previous, but what it can do should be cheaper to do
        loop_fusion::LoopFusionPass loop_fusion_pass({.allow_init_hoist = true});
        loop_fusion_pass.run(builder, analysis_manager);

        // Cleanup of artifacts of MapFusion
        dde.run(builder, analysis_manager);
        dce.run(builder, analysis_manager);
        Pipeline block_fusion("BlockFusion");
        block_fusion.register_pass<BlockFusionPass>();
        block_fusion.run(builder, analysis_manager);

        RedundantLoadEliminationPass rle;
        rle.run(builder, analysis_manager);
        dde.run(builder, analysis_manager);
        TaskletFusionPass task_fuse_pass;
        task_fuse_pass.run(builder, analysis_manager);

        // Fuse maps (final run: allow init-into-reduction hoisting now that distribution is done)
        auto map_fusion_hoist = map_fusion(true, false);
        map_fusion_hoist.run(builder, analysis_manager);
    }
    CompileStatistics::exit_stage_if_enabled();
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
