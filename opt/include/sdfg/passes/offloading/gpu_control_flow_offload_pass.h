#pragma once

#include <string>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/passes/pass.h"
#include "sdfg/transformations/offloading/cuda_offload_transform.h"
#include "sdfg/transformations/offloading/rocm_offload_transform.h"

namespace sdfg {
namespace passes {

struct CUDAControlFlowOffloadTraits {
    using OffloadTransform = cuda::CUDAOffloadTransform;

    static std::string name() {
        return "CUDAControlFlowOffloadPass";
    }

    static bool is_gpu_implementation(const data_flow::ImplementationType& implementation_type);
};

struct ROCMControlFlowOffloadTraits {
    using OffloadTransform = rocm::ROCMOffloadTransform;

    static std::string name() {
        return "ROCMControlFlowOffloadPass";
    }

    static bool is_gpu_implementation(const data_flow::ImplementationType& implementation_type);
};

/**
 * @brief Offloads non-loop control flow directly nested in the SDFG root.
 *
 * Maximal runs of consecutive eligible root children (Block, IfElse, While, Sequence) are
 * wrapped into a single-iteration Map and offloaded via Traits::OffloadTransform. A child is
 * eligible if its subtree contains no Return, no scheduled loop, no data offloading node and
 * no GPU-implemented library node. If a run cannot be offloaded, it is restored and each of
 * its children is retried on its own.
 */
template<typename Traits>
class GPUControlFlowOffloadPass : public Pass {
private:
    bool allow_dynamic_sizes_;

public:
    explicit GPUControlFlowOffloadPass(bool allow_dynamic_sizes = false);
    ~GPUControlFlowOffloadPass() override = default;

    bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    std::string name() override {
        return Traits::name();
    }
};

using CUDAControlFlowOffloadPass = GPUControlFlowOffloadPass<CUDAControlFlowOffloadTraits>;
using ROCMControlFlowOffloadPass = GPUControlFlowOffloadPass<ROCMControlFlowOffloadTraits>;

} // namespace passes
} // namespace sdfg
