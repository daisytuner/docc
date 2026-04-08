#include "sdfg/passes/offloading/rocblas_offloading_expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/transformations/offloading/rocblas_offloading_expansion.h"

namespace sdfg {
namespace rocm {

RocblasBLASOffloadingExpansionVisitor::RocblasBLASOffloadingExpansionVisitor(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool RocblasBLASOffloadingExpansionVisitor::visit() { return visitor::NonStoppingStructuredSDFGVisitor::visit(); }

bool RocblasBLASOffloadingExpansionVisitor::accept(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    for (auto lib_node : dataflow.library_nodes()) {
        if (auto* blas_node = dynamic_cast<math::blas::BLASNode*>(lib_node)) {
            ROCBLASOffloadingExpansion expansion(*blas_node);
            if (expansion.can_be_applied(builder_, analysis_manager_)) {
                expansion.apply(builder_, analysis_manager_);
                return true;
            }
        }
    }
    return false;
}

} // namespace rocm
} // namespace sdfg
