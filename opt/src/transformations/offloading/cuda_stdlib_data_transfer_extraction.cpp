#include "sdfg/transformations/offloading/cuda_stdlib_data_transfer_extraction.h"

#include <cassert>
#include <string>
#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/exceptions.h"
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

std::string CUDAStdlibDataTransferExtraction::create_device_container(
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

void CUDAStdlibDataTransferExtraction::create_allocate(
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
        this->memset_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void CUDAStdlibDataTransferExtraction::create_deallocate(
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
        this->memset_node_.debug_info(),
        SymEngine::null,
        symbolic::zero()
    );
}

void CUDAStdlibDataTransferExtraction::create_copy_from_device_with_deallocation(
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
        this->memset_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

CUDAStdlibDataTransferExtraction::CUDAStdlibDataTransferExtraction(::sdfg::stdlib::MemsetNode& memset_node)
    : memset_node_(memset_node) {}

std::string CUDAStdlibDataTransferExtraction::name() const { return "CUDAStdlibDataTransferExtraction"; }

bool CUDAStdlibDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->memset_node_.implementation_type().value() != cuda::ImplementationType_CUDAWithTransfers.value()) {
        return false;
    }

    // Restrict to memset nodes in their own block
    auto& dfg = this->memset_node_.get_parent();
    if (dfg.nodes().size() != dfg.in_degree(this->memset_node_) + dfg.out_degree(this->memset_node_) + 1) {
        return false;
    }

    return true;
}

void CUDAStdlibDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get data flow graph and block
    auto& dfg = this->memset_node_.get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Get sequence
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(block));
    assert(sequence);

    // Capture output accesses
    auto ptr_edge = dfg.in_edge_for_connector(this->memset_node_, "_ptr");
    auto& host_access_node =
        const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(ptr_edge->src()));
    auto& host_container_name = host_access_node.data();

    // Use the host container's actual type to avoid type mismatches
    auto& host_type = builder.subject().type(host_container_name);
    auto& type = static_cast<const types::Pointer&>(host_type);

    auto ptr_size = this->memset_node_.num();
    auto dPtr = this->create_device_container(builder, type, ptr_size);

    // Allocate device buffer
    this->create_allocate(builder, *sequence, *block, dPtr, ptr_size, type);

    // Copy from device to host and deallocate
    this->create_copy_from_device_with_deallocation(builder, *sequence, *block, host_container_name, dPtr, ptr_size, type);

    // Redirect output to device container
    host_access_node.data(dPtr);

    // Change the implementation type to without transfers
    this->memset_node_.implementation_type() = cuda::ImplementationType_CUDAWithoutTransfers;
}

void CUDAStdlibDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->memset_node_.element_id()}, {"type", "unknown"}}}};
    j["memset_node_element_id"] = this->memset_node_.element_id();
}

} // namespace cuda
} // namespace sdfg
