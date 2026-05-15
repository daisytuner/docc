#include "target_mapping.h"

#include "sdfg/passes/targets/target_mapping_pass.h"
#include "sdfg/plugins/target_mapping.h"


namespace docc::plugins {

void apply_lib_node_target_mapping(
    sdfg::plugins::Context& docc_context,
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    const target::TargetOptions& options
) {
    if (auto* target_handler = docc_context.get_target_handler(options.target)) {
        if (auto target_sched_time_mapping = target_handler->safe_apply_sched_time_mapping_fn_get()) {
            if (auto success = target_sched_time_mapping(builder, analysis_manager, options)) {
                // for now, targets are exclusive
                return;
            }
        }
    } else { // not the best place to do this here, should be more general, but we only arrive here, when its not "none"
        std::cerr << "[WARNING] Target '" << options.target << "' is unknown, ignoring!" << std::endl;
    }

    // Generic code. Find a way to declare TargetMappers with each plugin and then discover those from target and use
    // the generic pass std::vector<std::shared_ptr<sdfg::plugins::TargetMapper>> mappers{};
    // sdfg::passes::TargetMappingPass mappingPass(mappers);
    // mappingPass.run_pass(builder, analysis_manager);
}

} // namespace docc::plugins
