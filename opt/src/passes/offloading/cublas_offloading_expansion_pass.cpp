#include "sdfg/passes/offloading/cublas_offloading_expansion_pass.h"

#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/transformations/offloading/cublas_offloading_expansion.h"

namespace sdfg {
namespace cuda {

CublasBLASOffloadingExpansionVisitor::CublasBLASOffloadingExpansionVisitor(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool CublasBLASOffloadingExpansionVisitor::visit() { return visitor::NonStoppingStructuredSDFGVisitor::visit(); }

bool CublasBLASOffloadingExpansionVisitor::accept(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    for (auto lib_node : dataflow.library_nodes()) {
        if (auto* blas_node = dynamic_cast<math::blas::BLASNode*>(lib_node)) {
            CUBLASOffloadingExpansion expansion(*blas_node);
            if (expansion.can_be_applied(builder_, analysis_manager_)) {
                expansion.apply(builder_, analysis_manager_);
                return true;
            }
        }
    }
    return false;
}

} // namespace cuda
} // namespace sdfg
