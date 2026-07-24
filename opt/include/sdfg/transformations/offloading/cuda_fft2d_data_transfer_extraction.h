#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace cuda {

/**
 * @brief Extract the host<->device data transfers of an R2CFFT2D / C2RFFT2D node.
 *
 * Turns a `ImplementationType_CUDAWithTransfers` FFT2D node into a
 * `ImplementationType_CUDAWithoutTransfers` node surrounded by explicit offloading
 * nodes: an H2D copy (with allocation) for the read-only input `X`, plus a device
 * allocation and D2H copy (with deallocation) for the output `Y`. Shared by the
 * forward (R2C) and inverse (C2R) nodes; `forward_` selects which operand is the
 * complex half spectrum. Mirrors @ref CUDAFFTConvDataTransferExtraction.
 */
class CUDAFFT2DDataTransferExtraction : public transformations::Transformation {
private:
    data_flow::LibraryNode& node_;
    std::vector<symbolic::Expression> real_shape_; ///< [matrices, fftH, fftW]
    bool forward_; ///< true = R2C (X real, Y complex); false = C2R (X complex, Y real)

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
    CUDAFFT2DDataTransferExtraction(
        data_flow::LibraryNode& node, const std::vector<symbolic::Expression>& real_shape, bool forward
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;
};

} // namespace cuda
} // namespace sdfg
