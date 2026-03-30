/**
 * @file memlet_delinearization_analysis.h
 * @brief Analysis for delinearizing memlet subsets
 *
 * This analysis attempts to delinearize memlet subsets by recovering
 * multi-dimensional structure from linearized expressions using the
 * symbolic delinearize function with block-level assumptions.
 *
 * The delinearization technique is based on the algorithm described in:
 * https://dl.acm.org/doi/10.1145/2751205.2751248
 */

#pragma once

#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"

namespace sdfg {
namespace analysis {

/**
 * @class MemletDelinearizationAnalysis
 * @brief Analysis that delinearizes memlet subsets
 *
 * This analysis traverses all blocks in the SDFG and attempts to delinearize
 * the subset of each memlet. For memlets where delinearization is successful,
 * it stores the delinearized subset using sparse storage (only successful
 * delinearizations are stored in the map).
 *
 * The analysis uses the symbolic::delinearize function with assumptions from
 * the AssumptionsAnalysis for each block.
 *
 * The delinearization technique is based on the algorithm described in:
 * "Polyhedral AST generation is more than scanning polyhedra"
 * by Grosser et al., ACM TACO 2015.
 * https://dl.acm.org/doi/10.1145/2751205.2751248
 */
class MemletDelinearizationAnalysis : public Analysis {
private:
    std::unordered_map<const data_flow::Memlet*, std::unique_ptr<data_flow::Subset>> delinearized_subsets_;

    void traverse(structured_control_flow::ControlFlowNode& node, analysis::AnalysisManager& analysis_manager);

    void process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager);

protected:
    void run(analysis::AnalysisManager& analysis_manager) override;

public:
    MemletDelinearizationAnalysis(StructuredSDFG& sdfg);

    std::string name() const override { return "MemletDelinearizationAnalysis"; }

    /**
     * @brief Get the delinearized subset for a memlet
     * @param memlet The memlet to query
     * @return Pointer to the delinearized subset if delinearization was successful, nullptr otherwise
     */
    const data_flow::Subset* get(const data_flow::Memlet& memlet) const;
};

} // namespace analysis
} // namespace sdfg
