#include "sdfg/transformations/offloading/cuda_transform.h"

#include <unordered_set>

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/transformations/transformation.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace cuda {

std::string CUDATransform::name() const { return "CUDATransform"; }

bool CUDATransform::can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!OffloadTransform::can_be_applied(builder, analysis_manager)) {
        return false;
    }

    // Condition: Resulting CUDA grid X-dimension must not exceed hardware limits.
    // X grid dimension is limited to 2^31 - 1.
    auto num_iters = this->map_.num_iterations();
    if (!num_iters.is_null() && SymEngine::is_a<SymEngine::Integer>(*num_iters)) {
        int64_t iters = SymEngine::down_cast<const SymEngine::Integer&>(*num_iters).as_int();
        int64_t block = static_cast<int64_t>(block_size_);
        int64_t grid_size = (iters + block - 1) / block;

        constexpr int64_t max_grid_dim_x = 2147483647; // 2^31 - 1
        if (grid_size > max_grid_dim_x) {
            return false;
        }
    }

    return true;
}

void CUDATransform::add_device_buffer(
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

void CUDATransform::allocate_device_arg(
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

    offloading::add_offloading_node<CUDADataOffloadingNode>(
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

void CUDATransform::deallocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    Block& dealloc_block,
    std::string device_arg_name,
    symbolic::Expression arg_size,
    symbolic::Expression page_size
) {
    auto& free_type = builder.subject().type(device_arg_name);
    offloading::add_offloading_node<CUDADataOffloadingNode>(
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

void CUDATransform::copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    Block& copy_block
) {
    offloading::add_offloading_node<CUDADataOffloadingNode>(
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

void CUDATransform::copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    Block& copy_block
) {
    offloading::add_offloading_node<CUDADataOffloadingNode>(
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

void CUDATransform::copy_from_device(
    builder::StructuredSDFGBuilder& builder,
    Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    offloading::add_offloading_node<CUDADataOffloadingNode>(
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

void CUDATransform::copy_from_device_with_free(
    builder::StructuredSDFGBuilder& builder,
    Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    offloading::add_offloading_node<CUDADataOffloadingNode>(
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

void CUDATransform::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["block_size"] = block_size_;

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], map_);
};

CUDATransform CUDATransform::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    size_t block_size = desc["parameters"]["block_size"].get<size_t>();
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto map = dynamic_cast<structured_control_flow::Map*>(element);

    return CUDATransform(*map, block_size);
};


} // namespace cuda
} // namespace sdfg
