#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace cuda {

class CUDASoftmaxDataTransferExtraction : public transformations::Transformation {
private:
    math::tensor::SoftmaxNode& softmax_node_;

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
    explicit CUDASoftmaxDataTransferExtraction(math::tensor::SoftmaxNode& softmax_node);

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;
};

} // namespace cuda
} // namespace sdfg
