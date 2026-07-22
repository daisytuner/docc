#pragma once

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"

namespace sdfg {
namespace expanders {

/**
 * @class BatchedMatMulExpander
 * @brief Target-neutral batched-matmul expansion logic.
 *
 * Lowers a batched `MatMulNode` into a strided `BatchedGEMMNode`. The only
 * target-specific input is the GEMM implementation type, which the per-target
 * expanders (CUDA / ROCm) supply as their *decision*; all the geometry/layout
 * analysis lives here once.
 */
class BatchedMatMulExpander {
public:
    static bool expand_batched_matmul(
        builder::StructuredSDFGBuilder& builder,
        analysis::AnalysisManager& analysis_manager,
        math::tensor::MatMulNode& node,
        const data_flow::ImplementationType& impl_type
    );
};

} // namespace expanders
} // namespace sdfg
