#include "sdfg/transformations/offloading/rocm_stdlib_data_transfer_extraction.h"

#include <cassert>
#include <string>
#include <unordered_map>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace rocm {

std::string ROCMStdlibDataTransferExtraction::create_device_container(
    builder::StructuredSDFGBuilder& builder, const types::Pointer& type, const symbolic::Expression& size
) {
    auto new_type = type.clone();
    new_type->storage_type(types::StorageType(
        "AMD_Generic", size, types::StorageType::AllocationType::Unmanaged, types::StorageType::AllocationType::Unmanaged
    ));
    auto device_container = builder.find_new_name(ROCM_DEVICE_PREFIX);
    builder.add_container(device_container, *new_type);
    return device_container;
}

void ROCMStdlibDataTransferExtraction::create_allocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& alloc_block = builder.add_block_before(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        alloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->lib_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void ROCMStdlibDataTransferExtraction::create_deallocate(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& device_container,
    const types::Pointer& type
) {
    auto& dealloc_block = builder.add_block_after(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        dealloc_block,
        device_container,
        device_container,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->lib_node_.debug_info(),
        SymEngine::null,
        symbolic::zero()
    );
}

void ROCMStdlibDataTransferExtraction::create_copy_from_device_with_deallocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_after(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        type,
        type,
        this->lib_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

void ROCMStdlibDataTransferExtraction::create_copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    const symbolic::Expression& size,
    const types::Pointer& type
) {
    auto& copy_block = builder.add_block_before(sequence, block, {}, block.debug_info());
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        copy_block,
        host_container,
        device_container,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        type,
        type,
        this->lib_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

ROCMStdlibDataTransferExtraction::ROCMStdlibDataTransferExtraction(data_flow::LibraryNode& lib_node)
    : lib_node_(lib_node) {}

std::string ROCMStdlibDataTransferExtraction::name() const { return "ROCMStdlibDataTransferExtraction"; }

bool ROCMStdlibDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (this->lib_node_.implementation_type().value() != rocm::ImplementationType_ROCMWithTransfers.value()) {
        return false;
    }

    // Restrict to nodes in their own block
    auto& dfg = this->lib_node_.get_parent();
    if (dfg.nodes().size() != dfg.in_degree(this->lib_node_) + dfg.out_degree(this->lib_node_) + 1) {
        return false;
    }

    // Supported stdlib nodes
    if (dynamic_cast<stdlib::MemsetNode*>(&this->lib_node_)) {
        return true;
    } else if (dynamic_cast<stdlib::MemcpyNode*>(&this->lib_node_)) {
        return true;
    } else {
        return false;
    }
}

void ROCMStdlibDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Get data flow graph and block
    auto& dfg = this->lib_node_.get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    assert(block);

    // Get sequence
    auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(block->get_parent());
    assert(sequence);

    if (dynamic_cast<stdlib::MemsetNode*>(&this->lib_node_)) {
        this->apply_memset(builder, analysis_manager, dfg, *sequence, *block);
    } else if (dynamic_cast<stdlib::MemcpyNode*>(&this->lib_node_)) {
        this->apply_memcpy(builder, analysis_manager, dfg, *sequence, *block);
    }

    // Change the implementation type to without transfers
    this->lib_node_.implementation_type() = rocm::ImplementationType_ROCMWithoutTransfers;
}

void ROCMStdlibDataTransferExtraction::apply_memset(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    data_flow::DataFlowGraph& dfg,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block
) {
    auto& memset_node = static_cast<stdlib::MemsetNode&>(this->lib_node_);

    // Capture output accesses
    auto ptr_edge = dfg.in_edge_for_connector(memset_node, "_ptr");
    auto& host_access_node =
        const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(ptr_edge->src()));
    auto& host_container_name = host_access_node.data();

    // Use the host container's actual type to avoid type mismatches
    auto& host_type = builder.subject().type(host_container_name);
    auto& type = static_cast<const types::Pointer&>(host_type);

    auto ptr_size = memset_node.num();
    auto dPtr = this->create_device_container(builder, type, ptr_size);

    // Allocate device buffer
    this->create_allocate(builder, sequence, block, dPtr, ptr_size, type);

    // Copy from device to host and deallocate
    this->create_copy_from_device_with_deallocation(builder, sequence, block, host_container_name, dPtr, ptr_size, type);

    // Redirect output to device container
    host_access_node.data(dPtr);
}

void ROCMStdlibDataTransferExtraction::apply_memcpy(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    data_flow::DataFlowGraph& dfg,
    structured_control_flow::Sequence& sequence,
    structured_control_flow::Block& block
) {
    auto& memcpy_node = static_cast<stdlib::MemcpyNode&>(this->lib_node_);
    auto ptr_size = memcpy_node.count();

    // Handle _src (read) - need H2D transfer
    auto src_edge = dfg.in_edge_for_connector(memcpy_node, "_src");
    auto& src_access_node = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(src_edge->src()
    ));
    auto& src_container_name = src_access_node.data();
    auto& src_type = static_cast<const types::Pointer&>(builder.subject().type(src_container_name));

    auto dSrc = this->create_device_container(builder, src_type, ptr_size);
    this->create_copy_to_device_with_allocation(builder, sequence, block, src_container_name, dSrc, ptr_size, src_type);
    this->create_deallocate(builder, sequence, block, dSrc, src_type);
    src_access_node.data(dSrc);

    // Handle _dst (write) - need D2H transfer
    auto dst_edge = dfg.in_edge_for_connector(memcpy_node, "_dst");
    auto& dst_access_node = const_cast<data_flow::AccessNode&>(static_cast<const data_flow::AccessNode&>(dst_edge->src()
    ));
    auto& dst_container_name = dst_access_node.data();
    auto& dst_type = static_cast<const types::Pointer&>(builder.subject().type(dst_container_name));

    auto dDst = this->create_device_container(builder, dst_type, ptr_size);
    this->create_allocate(builder, sequence, block, dDst, ptr_size, dst_type);
    this->create_copy_from_device_with_deallocation(builder, sequence, block, dst_container_name, dDst, ptr_size, dst_type);
    dst_access_node.data(dDst);
}

void ROCMStdlibDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["subgraph"] = {{"0", {{"element_id", this->lib_node_.element_id()}, {"type", "unknown"}}}};
}

} // namespace rocm
} // namespace sdfg
