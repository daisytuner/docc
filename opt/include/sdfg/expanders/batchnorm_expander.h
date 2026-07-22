#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/batchnorm_node.h"

namespace sdfg {
namespace expanders {

/**
 * @class BatchNormExpander
 * @brief Target-neutral batch-normalization expansion logic.
 *
 * Lowers a `BatchNormNode` into primitive SDFG constructs (the GPU-friendly form
 * that moves sqrt/division into the innermost loop for parallelism). The per-target
 * expanders (CUDA / ROCm) own only the decision to invoke this.
 */
class BatchNormExpander {
public:
    static bool expand_batch_norm(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        math::tensor::BatchNormNode& node
    );
};

} // namespace expanders
} // namespace sdfg
