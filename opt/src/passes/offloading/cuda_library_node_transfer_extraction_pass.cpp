#include "sdfg/passes/offloading/cuda_library_node_transfer_extraction_pass.h"

#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/transformations/offloading/cublas_data_transfer_extraction.h"
#include "sdfg/transformations/offloading/cuda_stdlib_data_transfer_extraction.h"

namespace sdfg {
namespace cuda {

CudaLibraryNodeTransferExtractionVisitor::CudaLibraryNodeTransferExtractionVisitor(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager
)
    : visitor::NonStoppingStructuredSDFGVisitor(builder, analysis_manager) {}

bool CudaLibraryNodeTransferExtractionVisitor::visit() { return visitor::NonStoppingStructuredSDFGVisitor::visit(); }

bool CudaLibraryNodeTransferExtractionVisitor::accept(structured_control_flow::Block& block) {
    auto& dataflow = block.dataflow();
    for (auto lib_node : dataflow.library_nodes()) {
        if (auto* blas_node = dynamic_cast<math::blas::BLASNode*>(lib_node)) {
            CUBLASDataTransferExtraction expansion(*blas_node);
            if (expansion.can_be_applied(builder_, analysis_manager_)) {
                expansion.apply(builder_, analysis_manager_);
                return true;
            }
        }
        if (auto* memset_node = dynamic_cast<stdlib::MemsetNode*>(lib_node)) {
            CUDAStdlibDataTransferExtraction expansion(*memset_node);
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
