#include "sdfg/transformations/offloading/gpu_tensor_data_transfer_extraction.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/gpu/math/tensor/tensor_operands.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/types/pointer.h"
#include "symengine/symengine_rcp.h"

namespace sdfg::gpu::tensor {

static data_flow::AccessNode* connected_access(data_flow::LibraryNode& node, const std::string& connector) {
    auto* edge = node.get_parent().in_edge_for_connector(node, connector);
    if (edge == nullptr) {
        return nullptr;
    }
    return dynamic_cast<data_flow::AccessNode*>(const_cast<data_flow::DataFlowNode*>(&edge->src()));
}

GPUTensorDataTransferExtraction::GPUTensorDataTransferExtraction(data_flow::LibraryNode& lib_node)
    : lib_node_(lib_node) {
}

bool GPUTensorDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (lib_node_.implementation_type() != with_transfers()) {
        return false;
    }

    auto& dfg = lib_node_.get_parent();
    auto operands = tensor_operands(lib_node_, dfg);
    if (!operands) {
        return false;
    }

    // Restrict to nodes in their own block; aliased operands may share one access node.
    std::unordered_set<const data_flow::DataFlowNode*> neighbors;
    for (auto& edge : dfg.in_edges(lib_node_)) {
        neighbors.insert(&edge.src());
    }
    for (auto& edge : dfg.out_edges(lib_node_)) {
        neighbors.insert(&edge.dst());
    }
    if (dfg.nodes().size() != neighbors.size() + 1) {
        return false;
    }
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (block == nullptr || dynamic_cast<structured_control_flow::Sequence*>(block->get_parent()) == nullptr) {
        return false;
    }

    auto& sdfg = builder.subject();
    for (auto& operand : *operands) {
        auto* access = connected_access(lib_node_, operand.connector);
        if (access == nullptr || dynamic_cast<const types::Pointer*>(&sdfg.type(access->data())) == nullptr) {
            return false;
        }
    }
    return true;
}

void GPUTensorDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dfg = lib_node_.get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dfg.get_parent());
    auto& sequence = static_cast<structured_control_flow::Sequence&>(*block.get_parent());
    auto operands = *tensor_operands(lib_node_, dfg);

    std::unordered_map<std::string, std::string> host_of_connector;
    for (auto& operand : operands) {
        host_of_connector.emplace(operand.connector, connected_access(lib_node_, operand.connector)->data());
    }

    for (auto& buffer : tensor_buffers(operands, host_of_connector)) {
        // Keep the host container's own pointer type so the offloading memlets match it.
        auto type = builder.subject().type(buffer.host).clone();
        auto device_type = type->clone();
        device_type->storage_type(
            types::StorageType(
                storage(),
                buffer.num_bytes,
                types::StorageType::AllocationType::Unmanaged,
                types::StorageType::AllocationType::Unmanaged
            )
        );
        auto device_container = builder.find_new_name(device_prefix());
        builder.add_container(device_container, *device_type);

        auto& before = builder.add_block_before(sequence, block, block.debug_info());
        add_transfer(
            builder,
            before,
            buffer.host,
            device_container,
            buffer.copy_to_device ? offloading::DataTransferDirection::H2D : offloading::DataTransferDirection::NONE,
            offloading::BufferLifecycle::ALLOC,
            *type,
            buffer.num_bytes
        );

        auto& after = builder.add_block_after(sequence, block, block.debug_info());
        if (buffer.copy_to_host) {
            add_transfer(
                builder,
                after,
                buffer.host,
                device_container,
                offloading::DataTransferDirection::D2H,
                offloading::BufferLifecycle::FREE,
                *type,
                buffer.num_bytes
            );
        } else {
            add_transfer(
                builder,
                after,
                device_container,
                device_container,
                offloading::DataTransferDirection::NONE,
                offloading::BufferLifecycle::FREE,
                *type,
                SymEngine::null
            );
        }

        for (auto& connector : buffer.connectors) {
            connected_access(lib_node_, connector)->data(device_container);
        }
    }

    lib_node_.set_implementation_type(without_transfers());
}

void GPUTensorDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["subgraph"] = {{"0", {{"element_id", lib_node_.element_id()}, {"type", "unknown"}}}};
}

// CUDA

std::string CUDATensorDataTransferExtraction::name() const {
    return "CUDATensorDataTransferExtraction";
}

const data_flow::ImplementationType& CUDATensorDataTransferExtraction::with_transfers() const {
    return cuda::ImplementationType_CUDAWithTransfers;
}

const data_flow::ImplementationType& CUDATensorDataTransferExtraction::without_transfers() const {
    return cuda::ImplementationType_CUDAWithoutTransfers;
}

std::string CUDATensorDataTransferExtraction::storage() const {
    return "NV_Generic";
}

const std::string& CUDATensorDataTransferExtraction::device_prefix() const {
    return cuda::CUDA_DEVICE_PREFIX;
}

void CUDATensorDataTransferExtraction::add_transfer(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    offloading::DataTransferDirection direction,
    offloading::BufferLifecycle lifecycle,
    const types::IType& type,
    const symbolic::Expression& size
) {
    offloading::add_offloading_node<cuda::CUDADataOffloadingNode>(
        builder,
        block,
        host_container,
        device_container,
        direction,
        lifecycle,
        type,
        type,
        lib_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

// ROCm

std::string ROCMTensorDataTransferExtraction::name() const {
    return "ROCMTensorDataTransferExtraction";
}

const data_flow::ImplementationType& ROCMTensorDataTransferExtraction::with_transfers() const {
    return rocm::ImplementationType_ROCMWithTransfers;
}

const data_flow::ImplementationType& ROCMTensorDataTransferExtraction::without_transfers() const {
    return rocm::ImplementationType_ROCMWithoutTransfers;
}

std::string ROCMTensorDataTransferExtraction::storage() const {
    return "AMD_Generic";
}

const std::string& ROCMTensorDataTransferExtraction::device_prefix() const {
    return rocm::ROCM_DEVICE_PREFIX;
}

void ROCMTensorDataTransferExtraction::add_transfer(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& host_container,
    const std::string& device_container,
    offloading::DataTransferDirection direction,
    offloading::BufferLifecycle lifecycle,
    const types::IType& type,
    const symbolic::Expression& size
) {
    offloading::add_offloading_node<rocm::ROCMDataOffloadingNode>(
        builder,
        block,
        host_container,
        device_container,
        direction,
        lifecycle,
        type,
        type,
        lib_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

} // namespace sdfg::gpu::tensor
