#pragma once

#include <vector>
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {

/**
 * @brief Phased collapse pass: collects applicable collapses, then applies them.
 *
 * Given a set of maps, performs CollapseToDepth on each in two phases:
 * 1. can_be_applied phase: collects all maps where collapse is applicable
 * 2. apply phase: applies the collapse to all collected maps
 *
 * After application, the input vector is updated in-place with the collapsed outer loops.
 */
class CollapsePass : public Pass {
private:
    std::vector<structured_control_flow::Map*>& maps_;
    size_t target_depth_;

public:
    CollapsePass(std::vector<structured_control_flow::Map*>& maps, size_t target_depth);
    ~CollapsePass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "CollapsePass"; }
};

} // namespace passes
} // namespace sdfg
