#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/plugins/target_mapping.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg::passes {

/**
 * Generified version to map LibraryNodes for now.
 * * We start out just mapping the implementation type. But it should be ready to also do memory-offloading expansion at
 * the same time
 * * Goal is to extend the TargetMapper interface with more functions to also map other elements that are commonly
 * mapped.
 *
 * Take the boilerplate out of mapping common elements of library nodes and prepare for the list of mappers to become
 * dynamic and plugin-based. For now its used statically
 */
class TargetMappingVisitor : public visitor::NonStoppingStructuredSDFGVisitor {
protected:
    const std::vector<std::shared_ptr<plugins::TargetMapper>>& target_mappers_;

public:
    TargetMappingVisitor(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        const std::vector<std::shared_ptr<plugins::TargetMapper>>& target_mappers
    );

    bool accept(structured_control_flow::Block& node) override;
};

class TargetMappingPass : public Pass {
protected:
    std::vector<std::shared_ptr<plugins::TargetMapper>> target_mappers_;

public:
    TargetMappingPass(std::vector<std::shared_ptr<plugins::TargetMapper>> target_mappers);

    std::string name() override;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace sdfg::passes
