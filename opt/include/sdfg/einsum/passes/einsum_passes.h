#pragma once

#include <string>
#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/einsum/einsum_node.h"
#include "sdfg/einsum/einsum_state.h"
#include "sdfg/optimization_report/pass_report_consumer.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace einsum {


class EinsumDetectionPass : public passes::Pass {
public:
    virtual std::string name() override { return "EinsumDetectionPass"; }

    static bool find_einsum_core_ops(
        EinsumTracker& state,
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::Block& block
    );

    static bool consume_input_operations(
        EinsumTracker& state,
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::list<einsum::EinsumNode*>& queue
    );

    static bool consume_surrounding_loops(
        EinsumTracker& state,
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        std::list<einsum::EinsumNode*> einsum_queue
    );

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

class EinsumConversion : public visitor::NonStoppingStructuredSDFGVisitor {
private:
    PassReportConsumer* report_;

public:
    EinsumConversion(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        PassReportConsumer* report = nullptr
    );

    static std::string name() { return "EinsumConversion"; }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef passes::VisitorPass<EinsumConversion> EinsumConversionPass;

class EinsumLower : public visitor::StructuredSDFGVisitor {
public:
    EinsumLower(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() { return "EinsumLower"; }

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef passes::VisitorPass<EinsumLower> EinsumLowerPass;

/**
 * @class EinsumExpansion
 * @brief Visitor that expands the EinsumNode with the legacy expand() method
 *
 * The Expansion visitor traverses the SDFG and expands library nodes that
 * have ImplementationType_NONE. After the first expand, it cuts its traversal,
 * requiring another run to get to further relevant nodes in the same block.
 *
 * This is a workaround for Einsum. The other MathNode.expand() impls have been migrated to a new pass with an API that
 * would allow splitting Blocks up as part of replacement, but in a generic way. Once the logic of EinsumNode to do this
 * has been integrated into that generic pass, EinsumNode can also be migrated and all expansions will have access
 * to the same logic.
 */
class EinsumExpansion : public visitor::StructuredSDFGVisitor {
public:
    /**
     * @brief Construct the expansion visitor
     * @param builder SDFG builder for creating new nodes
     * @param analysis_manager Analysis manager for querying properties
     */
    EinsumExpansion(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /**
     * @brief Get the pass name
     * @return Name of the pass
     */
    static std::string name() { return "EinsumExpansion"; };

    /**
     * @brief Visit a block and attempt to expand its library nodes
     * @param node Block to visit
     * @return True if any expansion occurred
     */
    bool accept(structured_control_flow::Block& node) override;
};

/**
 * @typedef EinsumExpansionPass
 * @brief Pass wrapper for the EinsumExpansion visitor
 */
typedef passes::VisitorPass<EinsumExpansion> EinsumExpansionPass;

} // namespace einsum
} // namespace sdfg
