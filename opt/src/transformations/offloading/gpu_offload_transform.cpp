#include "sdfg/transformations/offloading/gpu_offload_transform.h"

#include <unordered_set>

#include "sdfg/structured_control_flow/block.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/transformations/transformation.h"
#include "symengine/symengine_rcp.h"

namespace sdfg {
namespace gpu {

template<typename OffloaderNodeType>
bool GPUOffloadTransform<OffloaderNodeType>::
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (!OffloadTransform::can_be_applied(builder, analysis_manager)) {
        return false;
    }

    // Condition: Resulting CUDA grid X-dimension must not exceed hardware limits.
    // X grid dimension is limited to 2^31 - 1.
    if (target_level_ == gpu::TargetLevel::X_GRID) {
        constexpr int64_t max_grid_dim_x = 2147483647; // 2^31 - 1
        if (parallel_size_->as_int() > max_grid_dim_x) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::Y_GRID) {
        constexpr int64_t max_grid_dim_y = 65535; // 2^16 - 1
        if (parallel_size_->as_int() > max_grid_dim_y) {
            return false;
        }
    } else if (target_level_ == gpu::TargetLevel::Z_GRID) {
        constexpr int64_t max_grid_dim_z = 65535; // 2^16 - 1
        if (parallel_size_->as_int() > max_grid_dim_z) {
            return false;
        }
    } else {
        // Unsupported outermost level
        return false;
    }

    return true;
};

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::add_device_buffer(
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

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::allocate_device_arg(
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

    offloading::add_offloading_node<OffloaderNodeType>(
        builder,
        alloc_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::ALLOC,
        out_type,
        out_type,
        this->loop_.debug_info(),
        arg_size,
        symbolic::zero()
    );
}

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::deallocate_device_arg(
    builder::StructuredSDFGBuilder& builder,
    Block& dealloc_block,
    std::string device_arg_name,
    symbolic::Expression arg_size,
    symbolic::Expression page_size
) {
    auto& free_type = builder.subject().type(device_arg_name);
    offloading::add_offloading_node<OffloaderNodeType>(
        builder,
        dealloc_block,
        device_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::NONE,
        offloading::BufferLifecycle::FREE,
        free_type,
        free_type,
        this->loop_.debug_info(),
        arg_size,
        symbolic::zero()
    );
}

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::copy_to_device(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    Block& copy_block
) {
    offloading::add_offloading_node<OffloaderNodeType>(
        builder,
        copy_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::NO_CHANGE,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->loop_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::copy_to_device_with_allocation(
    builder::StructuredSDFGBuilder& builder,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size,
    Block& copy_block
) {
    offloading::add_offloading_node<OffloaderNodeType>(
        builder,
        copy_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::H2D,
        offloading::BufferLifecycle::ALLOC,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->loop_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::copy_from_device(
    builder::StructuredSDFGBuilder& builder,
    Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    offloading::add_offloading_node<OffloaderNodeType>(
        builder,
        copy_out_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::NO_CHANGE,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->loop_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::copy_from_device_with_free(
    builder::StructuredSDFGBuilder& builder,
    Block& copy_out_block,
    const std::string host_arg_name,
    std::string device_arg_name,
    symbolic::Expression size,
    symbolic::Expression page_size
) {
    offloading::add_offloading_node<OffloaderNodeType>(
        builder,
        copy_out_block,
        host_arg_name,
        device_arg_name,
        offloading::DataTransferDirection::D2H,
        offloading::BufferLifecycle::FREE,
        builder.subject().type(host_arg_name),
        builder.subject().type(device_arg_name),
        this->loop_.debug_info(),
        size,
        symbolic::integer(0)
    );
}

template<typename OffloaderNodeType>
void GPUOffloadTransform<OffloaderNodeType>::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["parameters"] = nlohmann::json::object();
    j["parameters"]["target_level"] = to_string(target_level_);
    j["parameters"]["parallel_size"] = serializer::JSONSerializer::expression(parallel_size_);

    serializer::JSONSerializer ser_flat(false);
    j["subgraph"] = nlohmann::json::object();
    j["subgraph"]["0"] = nlohmann::json::object();
    ser_flat.serialize_node(j["subgraph"]["0"], loop_);
};

template<typename OffloaderNodeType>
GPUOffloadTransform<OffloaderNodeType> GPUOffloadTransform<
    OffloaderNodeType>::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc) {
    auto loop_id = desc["subgraph"]["0"]["element_id"].get<size_t>();
    auto target_level = target_level_from_string(desc["parameters"]["target_level"].get<std::string>());
    symbolic::Integer parallel_size =
        SymEngine::rcp_static_cast<const SymEngine::Integer>(symbolic::parse(desc["parameters"]["parallel_size"]));
    auto element = builder.find_element_by_id(loop_id);
    if (!element) {
        throw transformations::
            InvalidTransformationDescriptionException("Element with ID " + std::to_string(loop_id) + " not found.");
    }
    auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(element);
    if (!loop) {
        throw transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(loop_id) + " is not a StructuredLoop."
        );
    }

    return GPUOffloadTransform(*loop, target_level, parallel_size);
};


// Explicit instantiation of the out-of-line template members for the concrete node types.
// The whole-class form (`template class ...`) cannot be used: the class is abstract and
// from_json returns it by value, so members are instantiated individually instead.
template bool GPUOffloadTransform<
    cuda::CUDADataOffloadingNode>::can_be_applied(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::
    add_device_buffer(builder::StructuredSDFGBuilder&, std::string, std::string, symbolic::Expression);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::allocate_device_arg(
    builder::StructuredSDFGBuilder&, Block&, std::string, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::deallocate_device_arg(
    builder::StructuredSDFGBuilder&, Block&, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::
    copy_to_device(builder::StructuredSDFGBuilder&, std::string, std::string, symbolic::Expression, symbolic::Expression, Block&);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::
    copy_to_device_with_allocation(builder::StructuredSDFGBuilder&, std::string, std::string, symbolic::Expression, symbolic::Expression, Block&);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::copy_from_device(
    builder::StructuredSDFGBuilder&, Block&, std::string, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::copy_from_device_with_free(
    builder::StructuredSDFGBuilder&, Block&, std::string, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<cuda::CUDADataOffloadingNode>::to_json(nlohmann::json&) const;

template bool GPUOffloadTransform<
    rocm::ROCMDataOffloadingNode>::can_be_applied(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::
    add_device_buffer(builder::StructuredSDFGBuilder&, std::string, std::string, symbolic::Expression);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::allocate_device_arg(
    builder::StructuredSDFGBuilder&, Block&, std::string, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::deallocate_device_arg(
    builder::StructuredSDFGBuilder&, Block&, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::
    copy_to_device(builder::StructuredSDFGBuilder&, std::string, std::string, symbolic::Expression, symbolic::Expression, Block&);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::
    copy_to_device_with_allocation(builder::StructuredSDFGBuilder&, std::string, std::string, symbolic::Expression, symbolic::Expression, Block&);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::copy_from_device(
    builder::StructuredSDFGBuilder&, Block&, std::string, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::copy_from_device_with_free(
    builder::StructuredSDFGBuilder&, Block&, std::string, std::string, symbolic::Expression, symbolic::Expression
);
template void GPUOffloadTransform<rocm::ROCMDataOffloadingNode>::to_json(nlohmann::json&) const;


} // namespace gpu
} // namespace sdfg
