#pragma once

#include <map>

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class OffloadTransform : public transformations::Transformation {
protected:
    structured_control_flow::Map& map_;
    bool allow_dynamic_sizes_ = false;

public:
    explicit OffloadTransform(structured_control_flow::Map& map, bool allow_dynamic_sizes = false);

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;


protected:
    virtual types::StorageType local_device_storage_type() = 0;

    virtual types::StorageType global_device_storage_type(symbolic::Expression arg_size) = 0;

    virtual ScheduleType transformed_schedule_type() = 0;

    virtual std::string copy_prefix() = 0;

    template<class Container>
    void allocate_locals_on_device_stack(
        builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, Container& locals
    ) {
        auto& sdfg = builder.subject();

        for (auto& local : locals) {
            const types::IType& type = sdfg.type(local);
            type.storage_type() = local_device_storage_type(); // TODO useless
        }
    }

    virtual void setup_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) = 0;

    virtual void teardown_device(builder::StructuredSDFGBuilder& builder, Block& global_alloc_block) = 0;

    void handle_device_setup_and_teardown(
        builder::StructuredSDFGBuilder& builder,
        const std::map<std::string, analysis::RegionArgument>& arguments,
        const std::unordered_map<std::string, symbolic::Expression>& argument_sizes,
        std::string prefix
    );

    void update_map_containers(const std::map<std::string, analysis::RegionArgument>& arguments, std::string prefix);

    virtual void add_device_buffer(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size
    ) = 0;

    virtual void allocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        Block& alloc_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) = 0;

    virtual void deallocate_device_arg(
        builder::StructuredSDFGBuilder& builder,
        Block& dealloc_block,
        std::string device_arg_name,
        symbolic::Expression arg_size,
        symbolic::Expression page_size
    ) = 0;

    virtual void copy_to_device(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        Block& copy_block
    ) = 0;
    virtual void copy_to_device_with_allocation(
        builder::StructuredSDFGBuilder& builder,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size,
        Block& copy_block
    ) = 0;

    virtual void copy_from_device(
        builder::StructuredSDFGBuilder& builder,
        Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) = 0;
    virtual void copy_from_device_with_free(
        builder::StructuredSDFGBuilder& builder,
        Block& copy_out_block,
        std::string host_arg_name,
        std::string device_arg_name,
        symbolic::Expression size,
        symbolic::Expression page_size
    ) = 0;
};

} // namespace transformations
} // namespace sdfg
