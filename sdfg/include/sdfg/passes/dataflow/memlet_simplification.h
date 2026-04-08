/**
 * @file map_subset_simplification.h
 * @brief Pass for simplifying mixed-radix index expressions in memlet subsets
 *
 * This pass simplifies index expressions that arise from MapCollapse.
 * When a nested loop is collapsed into a single loop with index `i`,
 * array accesses become expressions like:
 *
 *   s0*(i/s0) + s1*((i/s1)%d1) + s2*((i/s2)%d2) + (i%d3)
 *
 * For C-contiguous arrays where s[k] = product of dimensions after k, this
 * expression equals `i` (mixed-radix number theorem). This pass recognizes
 * and simplifies such patterns.
 *
 * The pass only applies to Maps in loop normal form (init=0, stride=1) where
 * the index is proven non-negative.
 */

#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

/**
 * @class MemletSimplification
 * @brief Simplifies mixed-radix index expressions in memlet subsets
 *
 * This pass targets memlets with single-element subsets (flat pointer accesses)
 * within Maps that are in loop normal form. It recognizes patterns of the form:
 *
 *   sum of stride[k] * ((idx / stride[k]) % dim[k])
 *
 * where the strides form a C-contiguous layout, and simplifies them to just `idx`.
 *
 * Prerequisites:
 * - MapCollapse has been applied (creates div/mod decomposition)
 * - Maps are in loop normal form (zero init, unit stride)
 */
class MemletSimplification : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    /**
     * @brief Constructs a new MemletSimplification pass
     * @param builder The structured SDFG builder
     * @param analysis_manager The analysis manager
     */
    MemletSimplification(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    /**
     * @brief Returns the name of the pass
     * @return The string "MemletSimplification"
     */
    static std::string name() { return "MemletSimplification"; }

    /**
     * @brief Accepts a Map and simplifies memlet subsets in immediate blocks
     * @param map The map to process
     * @return true if any memlet subsets were simplified
     */
    virtual bool accept(structured_control_flow::Map& map) override;

private:
    /**
     * @brief Attempts to simplify a mixed-radix index expression
     * @param expr The expression to simplify
     * @param expected_base The expected base index variable
     * @return The simplified expression if pattern matches, nullopt otherwise
     */
    std::optional<symbolic::Expression>
    try_simplify_mixed_radix(const symbolic::Expression& expr, const symbolic::Symbol& expected_base);
};

typedef VisitorPass<MemletSimplification> MemletSimplificationPass;

} // namespace passes
} // namespace sdfg
