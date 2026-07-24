#include "sdfg/transformations/offloading/cuda_fft2d_data_transfer_extraction.h"

#include <cassert>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
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

std::string CUDAFFT2DDataTransferExtraction::create_device_container(
    builder::StructuredSDFGBuilder& builder, const types::Pointer& type, const symbolic::Expression& size
) {
    auto new_type = type.clone();
    new_type->storage_type(types::StorageType(
        "NV_Generic", size, types::StorageType::AllocationType::Unmanaged, types::StorageType::AllocationType::Unmanaged
    ));
    auto device_container = builder.find_new_name(CUDA_DEVICE_PREFIX);
    builder.add_container(device_container, *new_type);
    return device_container;
}

void CUDAFFT2DDataTransferExtraction::create_allocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& alloc_block = builder.add_block_before(sequence, block, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        alloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUDAFFT2DDataTransferExtraction::create_deallocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const types::Pointer& type
) {
    auto& dealloc_block = builder.add_block_after(sequence, block, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        dealloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->node_.debug_info(),
        SymEngine::null,
        symbolic::zero()
    );
}

void CUDAFFT2DDataTransferExtraction::create_copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_before(sequence, block, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUDAFFT2DDataTransferExtraction::create_copy_from_device_with_deallocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_after(sequence, block, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->node_.debug_info(),
        size,
        symbolic::zero()
    );
}

CUDAFFT2DDataTransferExtraction::CUDAFFT2DDataTransferExtraction(
    data_flow::LibraryNode& node, const std::vector<symbolic::Expression>& real_shape, bool forward
)
    : node_(node), real_shape_(real_shape), forward_(forward) {}

std::string CUDAFFT2DDataTransferExtraction::name() const { return "CUDAFFT2DDataTransferExtraction"; }

bool CUDAFFT2DDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->node_.implementation_type().value() != cuda::ImplementationType_CUDAWithTransfers.value()) {
        return false;
    }

    // Restrict to nodes that are the sole compute node in their block.
    auto& dfg = this->node_.get_parent();
    if (dfg.nodes().size() != dfg.in_degree(this->node_) + dfg.out_degree(this->node_) + 1) {
        return false;
    }

    return true;
}

void CUDAFFT2DDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dfg = this->node_.get_parent();
    auto* block = dyn_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);
    auto* sequence = dyn_cast<structured_control_flow::Sequence*>(block->get_parent());
    assert(sequence);

    // Element counts. real = matrices*fftH*fftW ; spec = matrices*fftH*halfW (halfW = fftW/2+1).
    const auto& M = this->real_shape_[0];
    const auto& fftH = this->real_shape_[1];
    const auto& fftW = this->real_shape_[2];
    auto halfW = symbolic::add(symbolic::div(fftW, symbolic::integer(2)), symbolic::integer(1));
    auto real_count = symbolic::mul(symbolic::mul(M, fftH), fftW);
    auto spec_count = symbolic::mul(symbolic::mul(M, fftH), halfW);

    // X is read-only, Y is write-only. Which one is complex depends on the direction.
    const auto& x_count = this->forward_ ? real_count : spec_count;
    const auto& y_count = this->forward_ ? spec_count : real_count;

    // Move the read-only input X onto the device: alloc + H2D before, free after.
    {
        auto edge = dfg.in_edge_for_connector(this->node_, "X");
        auto& access = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(edge->src()));
        auto& container = access.data();
        auto& type = static_cast<const types::Pointer&>(builder.subject().type(container));
        auto size = symbolic::mul(x_count, types::get_contiguous_element_size(type, true));

        auto device = create_device_container(builder, type, size);
        create_copy_to_device_with_allocation(builder, *sequence, *block, container, device, size, type);
        create_deallocate(builder, *sequence, *block, device, type);
        access.data(device);
    }

    // Output Y: write-only -> device alloc before, D2H copy + dealloc after.
    {
        auto y_edge = dfg.in_edge_for_connector(this->node_, "Y");
        auto& y_access = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(y_edge->src()));
        auto& y_container = y_access.data();
        auto& y_type = static_cast<const types::Pointer&>(builder.subject().type(y_container));
        auto y_size = symbolic::mul(y_count, types::get_contiguous_element_size(y_type, true));

        auto dY = create_device_container(builder, y_type, y_size);
        create_allocate(builder, *sequence, *block, dY, y_size, y_type);
        create_copy_from_device_with_deallocation(builder, *sequence, *block, y_container, dY, y_size, y_type);
        y_access.data(dY);
    }

    this->node_.implementation_type() = cuda::ImplementationType_CUDAWithoutTransfers;
}

void CUDAFFT2DDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["subgraph"] = {{"0", {{"element_id", this->node_.element_id()}, {"type", "unknown"}}}};
}

} // namespace cuda
} // namespace sdfg
