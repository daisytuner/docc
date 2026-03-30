#pragma once

#include <vector>
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/map.h"

namespace sdfg {
namespace passes {

enum class GPUTarget {
    CUDA,
    ROCM,
};

/**
 * @brief Phased GPU nested map parallelization pass.
 *
 * Given a set of outer maps, finds all nested maps within them and applies
 * nested parallelization (CUDA or ROCM) in two phases:
 * 1. can_be_applied phase: collects all nested maps where parallelization is applicable
 * 2. apply phase: applies parallelization to all collected maps
 */
class GPUNestedParallelizationPass : public Pass {
private:
    const std::vector<structured_control_flow::Map*>& maps_;
    GPUTarget target_;
    size_t block_size_;

public:
    GPUNestedParallelizationPass(
        const std::vector<structured_control_flow::Map*>& maps, GPUTarget target, size_t block_size
    );
    ~GPUNestedParallelizationPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override { return "GPUNestedParallelizationPass"; }
};

} // namespace passes
} // namespace sdfg
