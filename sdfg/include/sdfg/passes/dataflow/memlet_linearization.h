/**
 * @file memlet_linearization.h
 * @brief Pass for linearizing memlet base types
 *
 * This pass converts pointers to nested array types into flat pointers with
 * the element type of the innermost array. The subset is flattened into a
 * linearized access using the num_elements property of the arrays.
 */

#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

/**
 * @class MemletLinearization
 * @brief Linearizes memlet base types by flattening nested arrays
 *
 * This pass transforms memlets that have pointer base types pointing to nested
 * arrays. It converts such pointers into flat pointers to the innermost element
 * type, and adjusts the subset accordingly by linearizing multi-dimensional
 * array accesses into a single linear index.
 *
 * For example, a pointer to int[3][4] with subset [i, j] becomes a pointer to
 * int with subset [i * 4 + j].
 *
 * This linearization simplifies subsequent passes and code generation by
 * eliminating nested array types in pointer pointees.
 */
class MemletLinearization : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    /**
     * @brief Constructs a new MemletLinearization pass
     * @param builder The structured SDFG builder
     * @param analysis_manager The analysis manager
     */
    MemletLinearization(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /**
     * @brief Returns the name of the pass
     * @return The string "MemletLinearization"
     */
    static std::string name() {
        return "MemletLinearization";
    }

    /**
     * @brief Accepts a block and linearizes memlets in its dataflow graph
     * @param block The block to process
     * @return true if any memlets were linearized
     */
    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef VisitorPass<MemletLinearization> MemletLinearizationPass;

} // namespace passes
} // namespace sdfg
