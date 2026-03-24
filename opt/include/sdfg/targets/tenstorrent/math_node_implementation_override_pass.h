#pragma once

#include "sdfg/passes/pass.h"
#include "sdfg/targets/tenstorrent/library_node_mapping.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace tenstorrent {


/**
 * Name is wrong, as it is no longer MathNode specific.
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
