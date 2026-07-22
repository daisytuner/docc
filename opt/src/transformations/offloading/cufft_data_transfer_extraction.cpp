#include "sdfg/transformations/offloading/cufft_data_transfer_extraction.h"

#include <cassert>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace cuda {

CUFFTDataTransferExtraction::CUFFTDataTransferExtraction(math::tensor::FFTNodeBase& fft_node) : fft_node_(fft_node) {}

std::string CUFFTDataTransferExtraction::name() const { return "CUFFTDataTransferExtraction"; }

bool CUFFTDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->fft_node_.implementation_type().value() != cuda::ImplementationType_CUDAWithTransfers.value()) {
        return false;
    }

    // Restrict to FFT nodes in their own block
    auto& dfg = this->fft_node_.get_parent();
    if (dfg.nodes().size() != dfg.in_degree(this->fft_node_) + dfg.out_degree(this->fft_node_) + 1) {
        return false;
    }

    return true;
}

void CUFFTDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dfg = this->fft_node_.get_parent();
    auto* block = dyn_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    auto* sequence = dyn_cast<structured_control_flow::Sequence*>(block->get_parent());
    assert(sequence);

    const bool forward = this->fft_node_.direction() == math::tensor::FFTDirection::Forward;

    // Element counts on the input (__X) and output (__Y) connectors (Hermitian layout).
    auto x_count = forward ? this->fft_node_.real_extent() : this->fft_node_.complex_extent();
    auto y_count = forward ? this->fft_node_.complex_extent() : this->fft_node_.real_extent();

    // Input __X: read-only -> H2D copy with allocation, then free.
    auto x_edge = dfg.in_edge_for_connector(this->fft_node_, "__X");
    auto& x_access = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(x_edge->src()));
    auto& x_container = x_access.data();
    auto& x_type = static_cast<const types::Pointer&>(builder.subject().type(x_container));
    auto x_size = symbolic::mul(x_count, types::get_contiguous_element_size(x_type, true));

    auto dX = create_device_container(builder, x_type, x_size);
    create_copy_to_device_with_allocation(
        builder, *sequence, *block, x_container, dX, x_size, x_type, this->fft_node_.debug_info()
    );
    create_deallocate(builder, *sequence, *block, dX, x_type, this->fft_node_.debug_info());

    // Output __Y: write-only -> device alloc, then D2H copy with deallocation.
    auto y_edge = dfg.in_edge_for_connector(this->fft_node_, "__Y");
    auto& y_access = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(y_edge->src()));
    auto& y_container = y_access.data();
    auto& y_type = static_cast<const types::Pointer&>(builder.subject().type(y_container));
    auto y_size = symbolic::mul(y_count, types::get_contiguous_element_size(y_type, true));

    auto dY = create_device_container(builder, y_type, y_size);
    create_allocate(builder, *sequence, *block, dY, y_size, y_type, this->fft_node_.debug_info());
    create_copy_from_device_with_deallocation(
        builder, *sequence, *block, y_container, dY, y_size, y_type, this->fft_node_.debug_info()
    );

    // Redirect the node's operands to the device containers.
    x_access.data(dX);
    y_access.data(dY);

    this->fft_node_.implementation_type() = cuda::ImplementationType_CUDAWithoutTransfers;
}

void CUFFTDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["subgraph"] = {{"0", {{"element_id", this->fft_node_.element_id()}, {"type", "unknown"}}}};
}

} // namespace cuda
} // namespace sdfg
