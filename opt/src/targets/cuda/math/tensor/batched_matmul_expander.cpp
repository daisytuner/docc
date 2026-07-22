#include "sdfg/targets/cuda/math/tensor/batched_matmul_expander.h"

#include "sdfg/expanders/batched_matmul_expander.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg {
namespace offloading {

bool CudaBatchedMatMulExpander::
    expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Decision only: supply the CUDA (cuBLAS) GEMM implementation type. The shared
    // expanders layer performs the actual batched-matmul -> BatchedGEMM lowering.
    return expanders::BatchedMatMulExpander::
        expand_batched_matmul(builder, analysis_manager, node_, cuda::ImplementationType_CUDAWithTransfers);
}

} // namespace offloading
} // namespace sdfg
