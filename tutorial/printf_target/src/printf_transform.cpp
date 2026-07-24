#include "printf_transform.h"

#include <unordered_set>

#include <sdfg/structured_control_flow/block.h>
#include "printf_data_offloading_node.h"

namespace sdfg {
namespace printf_target {

std::string PrintfTransform::name() const { return "PrintfTransform"; }

void PrintfTransform::add_device_buffer(
    builder::StructuredSDFGBuilder& builder,
    std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression arg_size
) {
    // Add a simulated device pointer container
    auto& sdfg = builder.subject();
    auto& type = sdfg.type(host_arg_name);
    auto new_type = type.clone();
    new_type->storage_type(global_device_storage_type(arg_size));
    builder.add_container(device_arg_name, *new_type);
}

void PrintfTransform::allocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& alloc_block,
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

    auto& access_node_out_device = builder.add_access(alloc_block, device_arg_name);

    auto& malloc_node = builder.add_library_node<PrintfDataOffloadingNode>(
        alloc_block,
        this->map_.debug_info(),
        arg_size,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC
    );

    auto& out_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(alloc_block, malloc_node, "_ret", access_node_out_device, {}, out_type);
}

void PrintfTransform::deallocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& dealloc_block,
    std::string device_arg_name,
    symbolic::Expression arg_size,
    symbolic::Expression page_size
) {
    auto& access_node_in_device = builder.add_access(dealloc_block, device_arg_name);
    auto& access_node_out_device = builder.add_access(dealloc_block, device_arg_name);

    auto& free_node = builder.add_library_node<PrintfDataOffloadingNode>(
        dealloc_block,
        this->map_.debug_info(),
        arg_size,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE
    );

    auto& free_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(dealloc_block, access_node_in_device, free_node, "_ptr", {}, free_type);
    builder.add_computational_memlet(dealloc_block, free_node, "_ptr", access_node_out_device, {}, free_type);
}

void PrintfTransform::copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    structured_control_flow::Block& copy_block
) {
    auto& access_node_host = builder.add_access(copy_block, host_arg_name);
    auto& access_node_device = builder.add_access(copy_block, device_arg_name);

    auto& memcpy_node = builder.add_library_node<PrintfDataOffloadingNode>(
        copy_block,
        this->map_.debug_info(),
        size,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type(host_arg_name);
    builder.add_computational_memlet(copy_block, access_node_host, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(copy_block, memcpy_node, "_dst", access_node_device, {}, out_type);
}

void PrintfTransform::copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    structured_control_flow::Block& copy_block
) {
    auto& access_node_host = builder.add_access(copy_block, host_arg_name);
    auto& access_node_device = builder.add_access(copy_block, device_arg_name);

    auto& memcpy_node = builder.add_library_node<PrintfDataOffloadingNode>(
        copy_block,
        this->map_.debug_info(),
        size,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC
    );

    auto& in_type = builder.subject().type(host_arg_name);
    builder.add_computational_memlet(copy_block, access_node_host, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(copy_block, memcpy_node, "_dst", access_node_device, {}, out_type);
}

void PrintfTransform::copy_from_device(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    auto& access_node_device = builder.add_access(copy_out_block, device_arg_name);
    auto& access_node_host = builder.add_access(copy_out_block, host_arg_name);

    auto& memcpy_node = builder.add_library_node<PrintfDataOffloadingNode>(
        copy_out_block,
        this->map_.debug_info(),
        size,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE
    );

    auto& in_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(copy_out_block, access_node_device, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type(host_arg_name);
    builder.add_computational_memlet(copy_out_block, memcpy_node, "_dst", access_node_host, {}, out_type);
}

void PrintfTransform::copy_from_device_with_free(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    auto& access_node_device = builder.add_access(copy_out_block, device_arg_name);
    auto& access_node_host = builder.add_access(copy_out_block, host_arg_name);

    auto& memcpy_node = builder.add_library_node<PrintfDataOffloadingNode>(
        copy_out_block,
        this->map_.debug_info(),
        size,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE
    );

    auto& in_type = builder.subject().type(device_arg_name);
    builder.add_computational_memlet(copy_out_block, access_node_device, memcpy_node, "_src", {}, in_type);

    auto& out_type = builder.subject().type(host_arg_name);
    builder.add_computational_memlet(copy_out_block, memcpy_node, "_dst", access_node_host, {}, out_type);
}

void PrintfTransform::to_json(nlohmann::json& j) const {
    j["type"] = "PrintfTransform";
    j["map_element_id"] = map_.element_id();
}

PrintfTransform PrintfTransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto map_element_id = desc["map_element_id"].get<size_t>();

    auto map = dyn_cast<structured_control_flow::Map*>(builder.find_element_by_id(map_element_id));

    return PrintfTransform(*map);
}

} // namespace printf_target
} // namespace sdfg
