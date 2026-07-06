#include "sdfg/targets/rocm/math/tensor/batchnorm_expander.h"

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_expansion_utils.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/cuda/math/tensor/batchnorm_expander.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace offloading {

bool RocmBatchNormExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return CudaBatchNormExpander::expand_batch_norm(builder, analysis_manager, node_);
}

} // namespace offloading
} // namespace sdfg
