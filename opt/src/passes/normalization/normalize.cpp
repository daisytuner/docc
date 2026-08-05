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
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);

    // Distribute and permute
    auto pipeline = loop_normalization();
    pipeline.run(builder, analysis_manager);

    if (enable_fusion) {
        sdfg::passes::Pipeline dce = sdfg::passes::Pipeline::dead_code_elimination();
        sdfg::passes::DeadDataElimination dde;
        // DebugDumpPass::dump(sdfg, "py4.1.pre-fusion");

        // New Map Fusion, simpler than previous, but what it can do should be cheaper to do
        sdfg::passes::loop_fusion::LoopFusionPass loop_fusion_pass({.allow_init_hoist = true});
        loop_fusion_pass.run(builder, analysis_manager);

        // DebugDumpPass::dump(sdfg, "py4.2.post-fusion");

        // Cleanup of artifacts of MapFusion
        dde.run(builder, analysis_manager);
        dce.run(builder, analysis_manager);
        sdfg::passes::Pipeline block_fusion("BlockFusion");
        block_fusion.register_pass<sdfg::passes::BlockFusionPass>();
        block_fusion.run(builder, analysis_manager);

        sdfg::passes::RedundantLoadEliminationPass rle;
        rle.run(builder, analysis_manager);
        dde.run(builder, analysis_manager);
        sdfg::passes::TaskletFusionPass task_fuse_pass;
        task_fuse_pass.run(builder, analysis_manager);

        // Fuse maps (final run: allow init-into-reduction hoisting now that distribution is done)
        auto map_fusion_hoist = map_fusion(true, false);
        map_fusion_hoist.run(builder, analysis_manager);
    }
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
