#pragma once

#include <sdfg/passes/pass.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>

namespace sdfg {
namespace passes {
namespace normalization {

class PerfectLoopDistribution : public visitor::StructuredSDFGVisitor {
private:
    bool can_be_applied(structured_control_flow::StructuredLoop& loop);

    void apply(structured_control_flow::StructuredLoop& loop);

public:
    PerfectLoopDistribution(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "PerfectLoopDistribution"; };

    bool accept(structured_control_flow::For& node) override;

    bool accept(structured_control_flow::Map& node) override;
};

typedef passes::VisitorPass<PerfectLoopDistribution> PerfectLoopDistributionPass;

} // namespace normalization
} // namespace passes
} // namespace sdfg
