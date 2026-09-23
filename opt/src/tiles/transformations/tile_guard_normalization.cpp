#include "sdfg/tiles/transformations/tile_guard_normalization.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace tiles {

bool TileGuardNormalization::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Branch-condition-aware assumptions: loop peeling hoists the full-tile nest
    // under an `if (tile in bounds)`, so the discharge must see that guard. The
    // manager's cached instance omits branch conditions (the cheap path), so build
    // a dedicated one.
    analysis::AssumptionsAnalysis aa(builder.subject(), /*with_branch_conditions=*/true);
    aa.run(analysis_manager);
    auto params = aa.parameters();
    bool modified = false;
    visitor::for_each_block(builder.subject().root(), [&](structured_control_flow::Block& block) {
        for (auto* lib_node : block.dataflow().library_nodes()) {
            auto* node = dynamic_cast<TileCopyNode*>(lib_node);
            if (node == nullptr || node->guard().trivial()) {
                continue;
            }
            auto assums = aa.get(block, /*include_trivial_bounds=*/true);
            if (node->normalize_guard(params, assums)) {
                modified = true;
            }
        }
    });
    if (modified) {
        analysis_manager.invalidate_all();
    }
    return modified;
}

} // namespace tiles
} // namespace sdfg
