#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace rocm {

class ROCBLASOffloadingExpansion : public transformations::Transformation {
private:
    math::blas::BLASNode& blas_node_;

    std::string create_device_container(
        builder::StructuredSDFGBuilder& builder, const types::Pointer& type, const symbolic::Expression& size
    );

    void create_allocate(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block,
        const std::string& device_container,
        const symbolic::Expression& size,
        const types::Pointer& type
    );
    void create_deallocate(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block,
        const std::string& device_container,
        const types::Pointer& type
    );

    void create_copy_to_device(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block,
        const std::string& host_container,
        const std::string& device_container,
        const symbolic::Expression& size,
        const types::Pointer& type
    );
    void create_copy_from_device(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block,
        const std::string& host_container,
        const std::string& device_container,
        const symbolic::Expression& size,
        const types::Pointer& type
    );

    void create_copy_to_device_with_allocation(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block,
        const std::string& host_container,
        const std::string& device_container,
        const symbolic::Expression& size,
        const types::Pointer& type
    );
    void create_copy_from_device_with_deallocation(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Sequence& sequence,
        structured_control_flow::Block& block,
        const std::string& host_container,
        const std::string& device_container,
        const symbolic::Expression& size,
        const types::Pointer& type
    );

public:
    ROCBLASOffloadingExpansion(math::blas::BLASNode& blas_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& json) const override;

    static ROCBLASOffloadingExpansion from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace rocm
} // namespace sdfg
