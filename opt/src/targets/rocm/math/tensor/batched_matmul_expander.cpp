#include "sdfg/targets/rocm/math/tensor/batched_matmul_expander.h"

#include "sdfg/expanders/batched_matmul_expander.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace offloading {

bool RocmBatchedMatMulExpander::
    expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Decision only: supply the ROCm (rocBLAS) GEMM implementation type. The shared
    // expanders layer performs the actual batched-matmul -> BatchedGEMM lowering.
    return expanders::BatchedMatMulExpander::
        expand_batched_matmul(builder, analysis_manager, node_, rocm::ImplementationType_ROCMWithTransfers);
}

} // namespace offloading
} // namespace sdfg
