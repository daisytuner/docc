#include "sdfg/transformations/offloading/cuda_softmax_data_transfer_extraction.h"

#include <cassert>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace cuda {

std::string CUDASoftmaxDataTransferExtraction::create_device_container(
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

void CUDASoftmaxDataTransferExtraction::create_allocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& alloc_block = builder.add_block_before(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        alloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->softmax_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUDASoftmaxDataTransferExtraction::create_deallocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const types::Pointer& type
) {
    auto& dealloc_block = builder.add_block_after(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        dealloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->softmax_node_.debug_info(),
        SymEngine::null,
        symbolic::zero()
    );
}

void CUDASoftmaxDataTransferExtraction::create_copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_before(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->softmax_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUDASoftmaxDataTransferExtraction::create_copy_from_device_with_deallocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_after(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<CUDADataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->softmax_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

CUDASoftmaxDataTransferExtraction::CUDASoftmaxDataTransferExtraction(math::tensor::SoftmaxNode& softmax_node)
    : softmax_node_(softmax_node) {}

std::string CUDASoftmaxDataTransferExtraction::name() const { return "CUDASoftmaxDataTransferExtraction"; }

bool CUDASoftmaxDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->softmax_node_.implementation_type().value() != cuda::ImplementationType_CUDAWithTransfers.value()) {
        return false;
    }

    auto& dfg = this->softmax_node_.get_parent();
    if (dfg.nodes().size() != dfg.in_degree(this->softmax_node_) + dfg.out_degree(this->softmax_node_) + 1) {
        return false;
    }

    return true;
}

void CUDASoftmaxDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dfg = this->softmax_node_.get_parent();
    auto* block = dyn_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    auto* sequence = dyn_cast<structured_control_flow::Sequence*>(block->get_parent());
    assert(sequence);

    auto* x_edge = dfg.in_edge_for_connector(this->softmax_node_, "X");
    auto* y_edge = dfg.in_edge_for_connector(this->softmax_node_, "Y");
    if (!x_edge || !y_edge) {
        throw InvalidSDFGException("CUDASoftmaxDataTransferExtraction: Softmax node is missing X or Y connector");
    }

    auto& x_access = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(x_edge->src()));
    auto& y_access = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(y_edge->src()));

    auto prim_type = this->softmax_node_.primitive_type(dfg);
    types::Scalar base_type(prim_type);
    types::Pointer ptr_type(base_type);

    symbolic::Expression total_elems = symbolic::one();
    for (const auto& dim : this->softmax_node_.shape()) {
        total_elems = symbolic::mul(total_elems, dim);
    }
    auto total_size = symbolic::mul(total_elems, types::get_contiguous_element_size(ptr_type, true));

    auto dX = this->create_device_container(builder, ptr_type, total_size);
    auto dY = this->create_device_container(builder, ptr_type, total_size);

    this->create_copy_to_device_with_allocation(builder, *sequence, *block, x_access.data(), dX, total_size, ptr_type);
    this->create_allocate(builder, *sequence, *block, dY, total_size, ptr_type);
    this->create_copy_from_device_with_deallocation(builder, *sequence, *block, y_access.data(), dY, total_size, ptr_type);
    this->create_deallocate(builder, *sequence, *block, dX, ptr_type);

    x_access.data(dX);
    y_access.data(dY);

    this->softmax_node_.implementation_type() = cuda::ImplementationType_CUDAWithoutTransfers;
}

void CUDASoftmaxDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->softmax_node_.element_id()}, {"type", "unknown"}}}};
    j["softmax_node_element_id"] = this->softmax_node_.element_id();
}

} // namespace cuda
} // namespace sdfg
