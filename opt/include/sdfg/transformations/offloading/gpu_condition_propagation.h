#pragma once

#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace transformations {

class GPUConditionPropagation : public Transformation {
private:
    structured_control_flow::Map& map_;

public:
    GPUConditionPropagation(structured_control_flow::Map& map_);

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual std::string name() const override;

    virtual void to_json(nlohmann::json& j) const override;

    static GPUConditionPropagation from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

/**
 * @brief Root-anchored sweep of GPUConditionPropagation.
 *
 * Applies GPUConditionPropagation to every GPU-scheduled Map in the subtree rooted at `root`
 */
class GPUConditionPropagationScope : public Transformation {
private:
    structured_control_flow::StructuredLoop& root_;

public:
    GPUConditionPropagationScope(structured_control_flow::StructuredLoop& root);

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual std::string name() const override;

    virtual void to_json(nlohmann::json& j) const override;

    static GPUConditionPropagationScope from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

class BarrierFinder : public visitor::StructuredSDFGVisitor {
public:
    BarrierFinder(builder::StructuredSDFGBuilder& builder, sdfg::analysis::AnalysisManager& analysis_manager);

    using visitor::StructuredSDFGVisitor::accept;

    bool visit(structured_control_flow::ControlFlowNode* node);

    bool accept(structured_control_flow::Block& node) override;
};

} // namespace transformations
} // namespace sdfg
