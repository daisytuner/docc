#include "sdfg/passes/rpc/rpc_scheduling_pass.h"
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/transformations/rpc_node_transform.h"


namespace sdfg {
namespace passes {
namespace scheduler {

bool RpcOptimizationPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& root = builder.subject().root();

    // Normalize only if it hasn't been done yet
    bool normalize = !options_.already_normalized;

    transformations::RPCNodeTransform
        rpc_transform(root, options_.target, options_.category, *rpc_context_, enable_fusion_, normalize);

    rpc_transform.set_report(report_);
    if (rpc_transform.can_be_applied(builder, analysis_manager)) {
        rpc_transform.apply(builder, analysis_manager);
        return true;
    }
    return false;
}

} // namespace scheduler
} // namespace passes
} // namespace sdfg
