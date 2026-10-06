#include "sdfg/transformations/offloading/gpu_embedding_data_transfer_extraction.h"

#include <unordered_set>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/gpu/math/tensor/embedding.h"
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

GPUEmbeddingDataTransferExtraction::GPUEmbeddingDataTransferExtraction(math::tensor::EmbeddingNode& embedding_node)
    : embedding_node_(embedding_node) {
}

bool GPUEmbeddingDataTransferExtraction::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (embedding_node_.implementation_type() != with_transfers()) {
        return false;
    }

    // Restrict to nodes in their own block
    auto& dfg = embedding_node_.get_parent();
    if (dfg.nodes().size() != dfg.in_degree(embedding_node_) + dfg.out_degree(embedding_node_) + 1) {
        return false;
    }
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (block == nullptr || dynamic_cast<structured_control_flow::Sequence*>(block->get_parent()) == nullptr) {
        return false;
    }

    // Separate device copies of one host container would lose the written Y.
    std::unordered_set<std::string> containers;
    auto& sdfg = builder.subject();
    for (auto* connector : {"Y", "W", "I"}) {
        auto* access = connected_access(embedding_node_, connector);
        if (access == nullptr || dynamic_cast<const types::Pointer*>(&sdfg.type(access->data())) == nullptr ||
            !containers.insert(access->data()).second) {
            return false;
        }
    }
    return true;
}

std::string GPUEmbeddingDataTransferExtraction::create_device_container(
    builder::StructuredSDFGBuilder& builder, const std::string& host_container, const symbolic::Expression& size
) {
    // Keep the host container's own pointer type so the offloading memlets match it.
    auto device_type = builder.subject().type(host_container).clone();
    device_type->storage_type(
        types::StorageType(
            storage(), size, types::StorageType::AllocationType::Unmanaged, types::StorageType::AllocationType::Unmanaged
        )
    );
    auto device_container = builder.find_new_name(device_prefix());
    builder.add_container(device_container, *device_type);
    return device_container;
}

void GPUEmbeddingDataTransferExtraction::
    apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dfg = embedding_node_.get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dfg.get_parent());
    auto& sequence = static_cast<structured_control_flow::Sequence&>(*block.get_parent());

    for (auto* connector : {"Y", "W", "I"}) {
        auto& access = *connected_access(embedding_node_, connector);
        auto host_container = access.data();
        auto type = builder.subject().type(host_container).clone();
        auto size = embedding_bytes(embedding_node_, dfg, connector);
        auto device_container = create_device_container(builder, host_container, size);

        auto& before = builder.add_block_before(sequence, block, block.debug_info());
        auto& after = builder.add_block_after(sequence, block, block.debug_info());
        if (std::string(connector) == "Y") {
            // Y is fully overwritten: allocate without copy-in, copy back on free.
            add_transfer(
                builder,
                before,
                device_container,
                device_container,
                offloading::DataTransferDirection::NONE,
                offloading::BufferLifecycle::ALLOC,
                *type,
                size
            );
            add_transfer(
                builder,
                after,
                host_container,
                device_container,
                offloading::DataTransferDirection::D2H,
                offloading::BufferLifecycle::FREE,
                *type,
                size
            );
        } else {
            add_transfer(
                builder,
                before,
                host_container,
                device_container,
                offloading::DataTransferDirection::H2D,
                offloading::BufferLifecycle::ALLOC,
                *type,
                size
            );
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
        access.data(device_container);
    }

    embedding_node_.set_implementation_type(without_transfers());
}

void GPUEmbeddingDataTransferExtraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["subgraph"] = {{"0", {{"element_id", embedding_node_.element_id()}, {"type", "unknown"}}}};
}

// CUDA

std::string CUDAEmbeddingDataTransferExtraction::name() const {
    return "CUDAEmbeddingDataTransferExtraction";
}

const data_flow::ImplementationType& CUDAEmbeddingDataTransferExtraction::with_transfers() const {
    return cuda::ImplementationType_CUDAWithTransfers;
}

const data_flow::ImplementationType& CUDAEmbeddingDataTransferExtraction::without_transfers() const {
    return cuda::ImplementationType_CUDAWithoutTransfers;
}

std::string CUDAEmbeddingDataTransferExtraction::storage() const {
    return "NV_Generic";
}

const std::string& CUDAEmbeddingDataTransferExtraction::device_prefix() const {
    return cuda::CUDA_DEVICE_PREFIX;
}

void CUDAEmbeddingDataTransferExtraction::add_transfer(
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
        embedding_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

// ROCm

std::string ROCMEmbeddingDataTransferExtraction::name() const {
    return "ROCMEmbeddingDataTransferExtraction";
}

const data_flow::ImplementationType& ROCMEmbeddingDataTransferExtraction::with_transfers() const {
    return rocm::ImplementationType_ROCMWithTransfers;
}

const data_flow::ImplementationType& ROCMEmbeddingDataTransferExtraction::without_transfers() const {
    return rocm::ImplementationType_ROCMWithoutTransfers;
}

std::string ROCMEmbeddingDataTransferExtraction::storage() const {
    return "AMD_Generic";
}

const std::string& ROCMEmbeddingDataTransferExtraction::device_prefix() const {
    return rocm::ROCM_DEVICE_PREFIX;
}

void ROCMEmbeddingDataTransferExtraction::add_transfer(
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
        embedding_node_.debug_info(),
        size,
        symbolic::zero()
    );
}

} // namespace sdfg::gpu::tensor
