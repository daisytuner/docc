#pragma once

#include <sdfg/transformations/offloading/offload_transform.h>
#include "printf_target.h"

namespace sdfg {
namespace printf_target {

/**
 * @brief Transform that converts a loop to use the printf debug target
 *
 * This transform follows the same pattern as CUDATransform but instead
 * of generating actual GPU kernels and memory transfers, it generates
 * printf statements for debugging and tracing purposes.
 */
class PrintfTransform : public transformations::OffloadTransform {
public:
    explicit PrintfTransform(structured_control_flow::StructuredLoop& loop, bool allow_dynamic_sizes = false)
        : OffloadTransform(loop, allow_dynamic_sizes) {};

    std::string name() const override;

    void to_json(nlohmann::json& j) const override;

    static PrintfTransform from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc);

protected:
    /// Storage type for local device variables (simulated)
    types::StorageType local_device_storage_type() override {
        return types::StorageType(
            "Printf_Local",
            SymEngine::null,
            types::StorageType::AllocationType::Unmanaged,
            types::StorageType::AllocationType::Unmanaged
        );
    }

    /// Storage type for global device buffers (simulated)
    types::StorageType global_device_storage_type(symbolic::Expression arg_size) override {
        return types::StorageType(
            "Printf_Global",
            arg_size,
            types::StorageType::AllocationType::Unmanaged,
            types::StorageType::AllocationType::Unmanaged
        );
    }

    /// Returns the schedule type for printf target
    structured_control_flow::ScheduleType transformed_schedule_type() override { return ScheduleType_Printf::create(); }

    /// Prefix for device buffer copies
    std::string copy_prefix() override { return PRINTF_DEVICE_PREFIX; }

    /// Adds a device buffer container (simulated)
    void add_device_buffer(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size
    ) override;

    /// Generates printf for device allocation
    void allocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& alloc_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) override;

    /// Generates printf for device deallocation
    void deallocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& dealloc_block,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) override;

    /// Generates printf for host-to-device copy
    void copy_to_device(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        structured_control_flow::Block& copy_block
    ) override;

    /// Generates printf for host-to-device copy with allocation
    void copy_to_device_with_allocation(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        structured_control_flow::Block& copy_block
    ) override;

    /// Generates printf for device-to-host copy
    void copy_from_device(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) override;

    /// Generates printf for device-to-host copy with free
    void copy_from_device_with_free(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) override;

    /// No device setup needed for printf target
    void setup_device(builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& global_alloc_block)
        override {}

    /// No device teardown needed for printf target
    void teardown_device(builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& global_alloc_block)
        override {}
};

} // namespace printf_target
} // namespace sdfg
