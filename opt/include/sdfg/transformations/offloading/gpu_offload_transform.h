#pragma once

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/transformations/offloading/offload_transform.h"

namespace sdfg {
namespace gpu {

template<typename OffloaderNodeType>
class GPUOffloadTransform : public transformations::OffloadTransform {
public:
    explicit GPUOffloadTransform(
        structured_control_flow::StructuredLoop& loop,
        symbolic::Integer parallel_size = symbolic::integer(32),
        TargetLevel target_level = TargetLevel::X_GRID,
        bool allow_dynamic_sizes = false
    )
        : OffloadTransform(loop, allow_dynamic_sizes), parallel_size_(parallel_size), target_level_(target_level) {};

    virtual std::string name() const override = 0;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;

    static GPUOffloadTransform<OffloaderNodeType>
    from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& desc);

protected:
    const symbolic::Integer parallel_size_;
    gpu::TargetLevel target_level_;

    types::StorageType local_device_storage_type() override {
        return types::StorageType(
            "NV_Generic",
            SymEngine::null,
            types::StorageType::AllocationType::Unmanaged,
            types::StorageType::AllocationType::Unmanaged
        );
    }

    types::StorageType global_device_storage_type(symbolic::Expression arg_size) override {
        return types::StorageType(
            "NV_Generic",
            arg_size,
            types::StorageType::AllocationType::Unmanaged,
            types::StorageType::AllocationType::Unmanaged
        );
    }

    virtual ScheduleType transformed_schedule_type() override = 0;

    virtual std::string copy_prefix() override = 0;

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
};

} // namespace gpu
} // namespace sdfg
