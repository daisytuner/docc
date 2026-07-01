#include "sdfg/targets/cuda/math/tensor/conv_expander.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/rocm/math/tensor/conv_expander.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace offloading {

bool RocmConvExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // for now we reuse the cuda impl until we find sth. where they need to diverge
    return CudaConvExpander::expand_conv(builder, analysis_manager, node_);
}
} // namespace offloading
} // namespace sdfg
