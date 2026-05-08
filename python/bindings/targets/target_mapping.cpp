#include "target_mapping.h"

#include "sdfg/passes/targets/target_mapping_pass.h"
#include "sdfg/plugins/target_mapping.h"


namespace docc::plugins {

void apply_lib_node_target_mapping(
    sdfg::plugins::Context& docc_context,
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    TargetOptions& options
) {
    auto* target_handler = docc_context.get_target_handler(options.target);
    if (target_handler) {
        auto target_sched_time_mapping = target_handler->apply_sched_time_mapping;
        if (target_sched_time_mapping) {
            auto success = target_sched_time_mapping(builder, analysis_manager, options);
            if (success) {
                // for now, targets are exclusive
                return;
            }
        }
    }

    // Generic code. Find a way to declare TargetMappers with each plugin and then discover those from target and use
    // the generic pass std::vector<std::shared_ptr<sdfg::plugins::TargetMapper>> mappers{};
    // sdfg::passes::TargetMappingPass mappingPass(mappers);
    // mappingPass.run_pass(builder, analysis_manager);
}

} // namespace docc::plugins
