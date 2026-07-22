#include "sdfg/targets/rocm/math/tensor/batchnorm_expander.h"

#include "sdfg/expanders/batchnorm_expander.h"

namespace sdfg {
namespace offloading {

bool RocmBatchNormExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Decision only; the lowering logic lives in the target-neutral `expanders` layer.
    return expanders::BatchNormExpander::expand_batch_norm(builder, analysis_manager, node_);
}

} // namespace offloading
} // namespace sdfg
