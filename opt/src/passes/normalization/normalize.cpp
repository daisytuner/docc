#include "sdfg/passes/normalization/normalize.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/tasklet_fusion.h"
#include "sdfg/passes/map_fusion_by_domain_pass.h"
#include "sdfg/passes/normalization/normalization.h"
#include "sdfg/passes/redundant_load_elimination_pass.h"
#include "sdfg/structured_sdfg.h"

namespace sdfg {
namespace passes {
namespace normalization {

void normalize(sdfg::StructuredSDFG& sdfg, bool enable_fusion) {
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);

    if (enable_fusion) {
        // Fuse maps (no init-into-reduction hoisting yet; this run precedes loop distribution)
        auto map_fusion_no_hoist = map_fusion(false);
        map_fusion_no_hoist.run(builder, analysis_manager);
    }

    // Distribute and permute
    auto pipeline = loop_normalization();
    pipeline.run(builder, analysis_manager);

    if (enable_fusion) {
        // Fuse maps (final run: allow init-into-reduction hoisting now that distribution is done)
        auto map_fusion_hoist = map_fusion(true);
        map_fusion_hoist.run(builder, analysis_manager);
    }
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
