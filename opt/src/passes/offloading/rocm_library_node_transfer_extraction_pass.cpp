#include "sdfg/passes/offloading/rocm_library_node_transfer_extraction_pass.h"

#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/transformations/offloading/rocblas_data_transfer_extraction.h"
#include "sdfg/transformations/offloading/rocm_stdlib_data_transfer_extraction.h"

namespace sdfg {
namespace rocm {

RocmLibraryNodeTransferExtractionVisitor::RocmLibraryNodeTransferExtractionVisitor(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool RocmLibraryNodeTransferExtractionVisitor::visit() { return visitor::NonStoppingStructuredSDFGVisitor::visit(); }

bool RocmLibraryNodeTransferExtractionVisitor::accept(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    for (auto lib_node : dataflow.library_nodes()) {
        if (auto* blas_node = dynamic_cast<math::blas::BLASNode*>(lib_node)) {
            ROCBLASDataTransferExtraction expansion(*blas_node);
            if (expansion.can_be_applied(builder_, analysis_manager_)) {
                expansion.apply(builder_, analysis_manager_);
                return true;
            }
        }
        if (auto* memset_node = dynamic_cast<stdlib::MemsetNode*>(lib_node)) {
            ROCMStdlibDataTransferExtraction expansion(*memset_node);
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
