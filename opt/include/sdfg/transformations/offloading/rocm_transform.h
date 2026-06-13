#pragma once

#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/transformations/offloading/offload_transform.h"

namespace sdfg {
namespace rocm {

/**
 * @brief Offloads a top-level map to a ROCm/HIP kernel (X grid dimension).
 *
 * This transformation does not perform blocking or tiling on its own. It expects
 * the scheduler to have already identified a suitable map for GPU offloading.
 * The transformation assigns the map to the ROCm X grid dimension and handles
 * data transfers between host and device.
 *
 * The resulting grid X-dimension is validated against ROCm/HIP hardware limits
 * (2^31 - 1 blocks). If the grid would exceed this limit, the transformation
 * is rejected (can_be_applied returns false).
 */
class ROCMTransform : public transformations::OffloadTransform {
public:
    explicit ROCMTransform(structured_control_flow::Map& map, int block_size = 64, bool allow_dynamic_sizes = false)
        : OffloadTransform(map, allow_dynamic_sizes), block_size_(block_size) {};

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static ROCMTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc);

protected:
    types::StorageType local_device_storage_type() override {
        return types::StorageType(
            "AMD_Generic",
            SymEngine::null,
            types::StorageType::AllocationType::Unmanaged,
            types::StorageType::AllocationType::Unmanaged
        );
    }

    types::StorageType global_device_storage_type(symbolic::Expression arg_size) override {
        return types::StorageType(
            "AMD_Generic",
            arg_size,
            types::StorageType::AllocationType::Unmanaged,
            types::StorageType::AllocationType::Unmanaged
        );
    }

    ScheduleType transformed_schedule_type() override {
        auto schedule = ScheduleType_ROCM::create();
        if (block_size_ != 0) {
            ScheduleType_ROCM::block_size(schedule, symbolic::integer(block_size_));
        }
        return schedule;
    }

    std::string copy_prefix() override { return ROCM_DEVICE_PREFIX; }

    void add_device_buffer(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size
    ) override;

    void allocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        Block& alloc_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) override;

    void deallocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        Block& dealloc_block,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) override;

    void copy_to_device(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        Block& copy_block
    ) override;

    void copy_to_device_with_allocation(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        Block& copy_block
    ) override;

    void copy_from_device(
        builder::StructuredSDFGBuilder& builder,
        Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) override;

    void copy_from_device_with_free(
        builder::StructuredSDFGBuilder& builder,
        Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) override;

    void setup_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) override {}
    void teardown_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) override {}

private:
    int block_size_;
};

} // namespace rocm
} // namespace sdfg
