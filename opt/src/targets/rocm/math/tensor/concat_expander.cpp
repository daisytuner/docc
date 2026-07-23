#include "sdfg/targets/rocm/math/tensor/concat_expander.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/targets/cuda/math/tensor/concat_expander.h"

namespace sdfg {
namespace offloading {

bool RocmConcatExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return CudaConcatExpander::expand_concat_separately(builder, analysis_manager, this->node_);
}

} // namespace offloading
} // namespace sdfg
