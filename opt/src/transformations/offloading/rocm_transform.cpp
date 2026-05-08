#include "sdfg/transformations/offloading/rocm_transform.h"

#include <unordered_set>

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/transformations/transformation.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace rocm {

std::string ROCMTransform::name() const { return "ROCMTransform"; }

void ROCMTransform::add_device_buffer(
    builder::StructuredSDFGBuilder& builder,
    std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression arg_size
) {
    // Allocate device pointer
    auto& sdfg = builder.subject();
    auto& type = sdfg.type(host_arg_name);
    auto new_type = type.clone();
    new_type->storage_type(global_device_storage_type(arg_size));
    builder.add_container(device_arg_name, *new_type);
}

void ROCMTransform::allocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    Block& alloc_block,
    std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression arg_size,
    symbolic::Expression page_size
) {
    auto& sdfg = builder.subject();
    if (!builder.subject().exists(device_arg_name)) {
        auto& type = sdfg.type(host_arg_name);
        auto new_type = type.clone();
        new_type->storage_type(global_device_storage_type(arg_size));
        new_type->storage_type().allocation(types::StorageType::AllocationType::Unmanaged);
        new_type->storage_type().deallocation(types::StorageType::AllocationType::Unmanaged);
        new_type->storage_type().allocation_size(SymEngine::null);

        std::unordered_set<std::string> container_set(sdfg.containers().begin(), sdfg.containers().end());
        if (container_set.find(device_arg_name) == container_set.end()) {
            builder.add_container(device_arg_name, *new_type);
        }
    }
    auto& out_type = builder.subject().type(device_arg_name);

    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        alloc_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        out_type,
        out_type,
        this->map_.debug_info(),
        arg_size,
        symbolic::zero()
    );
}

void ROCMTransform::deallocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    Block& dealloc_block,
    std::string device_arg_name,
    symbolic::Expression arg_size,
    symbolic::Expression page_size
) {
    auto& free_type = builder.subject().type(device_arg_name);
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        dealloc_block,
        device_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        free_type,
        free_type,
        this->map_.debug_info(),
        arg_size,
        symbolic::zero()
    );
}

void ROCMTransform::copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    Block& copy_block
) {
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        copy_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->map_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

void ROCMTransform::copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    Block& copy_block
) {
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        copy_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->map_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

void ROCMTransform::copy_from_device(
    builder::StructuredSDFGBuilder& builder,
    Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        copy_out_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->map_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

void ROCMTransform::copy_from_device_with_free(
    builder::StructuredSDFGBuilder& builder,
    Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    offloading::add_offloading_node<ROCMDataOffloadingNode>(
        builder,
        copy_out_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->map_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

void ROCMTransform::to_json(nlohmann::json& j) const {
    std::string loop_type;
    if (dynamic_cast<structured_control_flow::Map*>(&map_)) {
        loop_type = "map";
    } else {
        throw std::runtime_error("Unsupported loop type for serialization of loop: " + map_.indvar()->get_name());
    }

    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", this->map_.element_id()}, {"type", loop_type}}}};
    j["parameters"] = {{"block_size", block_size_}};
};

ROCMTransform ROCMTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t block_size = desc["parameters"]["block_size"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto map = dynamic_cast<structured_control_flow::Map*>(element);

    return ROCMTransform(*map, block_size);
};


} // namespace rocm
} // namespace sdfg
