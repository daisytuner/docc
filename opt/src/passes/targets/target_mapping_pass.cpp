#include <sdfg/passes/targets/target_mapping_pass.h>

namespace sdfg::passes {

TargetMappingVisitor::TargetMappingVisitor(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    const std::vector<std::shared_ptr<plugins::TargetMapper>>& target_mappers
)
    : NonStoppingStructuredSDFGVisitor(builder, analysis_manager), target_mappers_(target_mappers) {}

bool TargetMappingVisitor::accept(structured_control_flow::Block& node) {
    bool applied = false;

    for (auto* lib_node : node.dataflow().library_nodes()) {
        for (const auto& target_mapper : target_mappers_) {
            if (target_mapper->try_map(builder_, analysis_manager_, *lib_node)) {
                applied = true;
                break; // Stop after the first successful mapping
            }
        }
    }

    return applied;
}

TargetMappingPass::TargetMappingPass(std::vector<std::shared_ptr<plugins::TargetMapper>> target_mappers)
    : target_mappers_(target_mappers) {}

std::string TargetMappingPass::name() { return "TargetMapper"; }

bool TargetMappingPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    TargetMappingVisitor visitor(builder, analysis_manager, target_mappers_);
    auto applied = visitor.visit();
    return applied;
}

} // namespace sdfg::passes
