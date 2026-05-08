#pragma once

#include <sdfg/passes/pass.h>
#include <sdfg/plugins/target_mapping.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>

namespace sdfg {
namespace tenstorrent {

class TTLibNodeMapper : public plugins::TargetMapper {
public:
    bool try_map(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        data_flow::LibraryNode& node
    ) const override;
};

/**
 * Name is wrong, as it is no longer MathNode specific.
 * @deprecated Use TTLibNodeMapper
 */
class MathNodeImplementationOverride : public visitor::StructuredSDFGVisitor {
public:
    MathNodeImplementationOverride(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "MathNodeImplementationOverride"; };

    bool accept(structured_control_flow::Block& node) override;
};

typedef passes::VisitorPass<MathNodeImplementationOverride> MathNodeImplementationOverridePass;

} // namespace tenstorrent
} // namespace sdfg
