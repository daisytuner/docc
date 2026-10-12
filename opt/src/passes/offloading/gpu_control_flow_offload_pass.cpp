#include "sdfg/passes/offloading/gpu_control_flow_offload_pass.h"

#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace passes {

bool CUDAControlFlowOffloadTraits::is_gpu_implementation(const data_flow::ImplementationType& implementation_type) {
    return implementation_type == cuda::ImplementationType_CUDA ||
           implementation_type == cuda::ImplementationType_CUDAWithTransfers ||
           implementation_type == cuda::ImplementationType_CUDAWithoutTransfers;
}

bool ROCMControlFlowOffloadTraits::is_gpu_implementation(const data_flow::ImplementationType& implementation_type) {
    return implementation_type == rocm::ImplementationType_ROCM ||
           implementation_type == rocm::ImplementationType_ROCMWithTransfers ||
           implementation_type == rocm::ImplementationType_ROCMWithoutTransfers;
}

template<typename Traits>
GPUControlFlowOffloadPass<Traits>::GPUControlFlowOffloadPass(bool allow_dynamic_sizes)
    : allow_dynamic_sizes_(allow_dynamic_sizes) {
}

template<typename Traits>
bool GPUControlFlowOffloadPass<
    Traits>::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // TODO: implement wrapping and offloading of root control flow.
    return false;
}

template class GPUControlFlowOffloadPass<CUDAControlFlowOffloadTraits>;
template class GPUControlFlowOffloadPass<ROCMControlFlowOffloadTraits>;

} // namespace passes
} // namespace sdfg
