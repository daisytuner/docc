#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_conv_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace cuda {

/**
 * @brief Extract the host<->device data transfers of a fused FFTConv node.
 *
 * Turns a `ImplementationType_CUDAWithTransfers` FFTConvNode into a
 * `ImplementationType_CUDAWithoutTransfers` node surrounded by explicit offloading
 * nodes: H2D copies (with allocation) for the read-only operands `X`, `W` and the
 * optional bias `B`, plus a device allocation and D2H copy (with deallocation) for
 * the output `Y`. This makes the fused convolution device-resident so subsequent
 * offloading passes can keep the surrounding tensors on the GPU.
 */
class CUDAFFTConvDataTransferExtraction : public transformations::Transformation {
private:
    math::tensor::FFTConvNode& fft_conv_node_;

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
    CUDAFFTConvDataTransferExtraction(math::tensor::FFTConvNode& fft_conv_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& json) const override;
};

} // namespace cuda
} // namespace sdfg
