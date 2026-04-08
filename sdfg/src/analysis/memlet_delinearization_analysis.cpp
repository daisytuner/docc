#include "sdfg/analysis/memlet_delinearization_analysis.h"

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/delinearization.h"

namespace sdfg {
namespace analysis {

MemletDelinearizationAnalysis::MemletDelinearizationAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void MemletDelinearizationAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    delinearized_subsets_.clear();
    traverse(sdfg_.root(), analysis_manager);
}

void MemletDelinearizationAnalysis::
    traverse(structured_control_flow::ControlFlowNode& node, analysis::AnalysisManager& analysis_manager) {
    if (auto block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        process_block(*block, analysis_manager);
    } else if (auto sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); i++) {
            traverse(sequence->at(i).first, analysis_manager);
        }
    } else if (auto if_else = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < if_else->size(); i++) {
            traverse(if_else->at(i).first, analysis_manager);
        }
    } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(&node)) {
        traverse(while_stmt->root(), analysis_manager);
    } else if (auto loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        traverse(loop->root(), analysis_manager);
    }
    // Break, Continue, Return nodes don't contain blocks
}

void MemletDelinearizationAnalysis::
    process_block(structured_control_flow::Block& block, analysis::AnalysisManager& analysis_manager) {
    auto& assumptions_analysis = analysis_manager.get<AssumptionsAnalysis>();
    auto& assumptions = assumptions_analysis.get(block);

    auto& dfg = block.dataflow();
    for (auto& memlet : dfg.edges()) {
        const auto& subset = memlet.subset();

        // Skip empty subsets
        if (subset.size() != 1) {
            continue;
        }

        // Attempt to delinearize the subset
        auto result = symbolic::delinearize(subset.at(0), assumptions);
        if (!result.success) {
            continue;
        }
        const auto& delinearized = result.indices;

        // Check if delinearization changed the subset
        bool changed = (delinearized.size() != subset.size());
        if (!changed) {
            for (size_t i = 0; i < subset.size(); i++) {
                if (!symbolic::eq(subset[i], delinearized[i])) {
                    changed = true;
                    break;
                }
            }
        }

        // Store result only if delinearization was successful (sparse storage)
        if (changed) {
            delinearized_subsets_[&memlet] = std::make_unique<data_flow::Subset>(std::move(delinearized));
        }
    }
}

const data_flow::Subset* MemletDelinearizationAnalysis::get(const data_flow::Memlet& memlet) const {
    auto it = delinearized_subsets_.find(&memlet);
    if (it == delinearized_subsets_.end()) {
        return nullptr;
    }
    return it->second.get();
}

} // namespace analysis
} // namespace sdfg
