#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu::tensor {

/**
 * @brief Moves the transfers of a `WithTransfers` GPU tensor library node into explicit offloading blocks.
 *
 * Each host container bound to the node (see @ref tensor_buffers) gets one device
 * container, allocated before and freed after the node. Contents are copied in or
 * out as the node's pointer access types require. The node is rewired to the
 * device containers and switched to the `WithoutTransfers` implementation, so
 * later passes can minimize the transfers.
 */
class GPUTensorDataTransferExtraction : public transformations::Transformation {
protected:
    data_flow::LibraryNode& lib_node_;

    virtual const data_flow::ImplementationType& with_transfers() const = 0;
    virtual const data_flow::ImplementationType& without_transfers() const = 0;
    virtual std::string storage() const = 0;
    virtual const std::string& device_prefix() const = 0;

    virtual void add_transfer(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        const std::string& host_container,
        const std::string& device_container,
        offloading::DataTransferDirection direction,
        offloading::BufferLifecycle lifecycle,
        const types::IType& type,
        const symbolic::Expression& size
    ) = 0;

public:
    explicit GPUTensorDataTransferExtraction(data_flow::LibraryNode& lib_node);

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;
};

class CUDATensorDataTransferExtraction : public GPUTensorDataTransferExtraction {
protected:
    const data_flow::ImplementationType& with_transfers() const override;
    const data_flow::ImplementationType& without_transfers() const override;
    std::string storage() const override;
    const std::string& device_prefix() const override;

    void add_transfer(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        const std::string& host_container,
        const std::string& device_container,
        offloading::DataTransferDirection direction,
        offloading::BufferLifecycle lifecycle,
        const types::IType& type,
        const symbolic::Expression& size
    ) override;

public:
    using GPUTensorDataTransferExtraction::GPUTensorDataTransferExtraction;

    std::string name() const override;
};

class ROCMTensorDataTransferExtraction : public GPUTensorDataTransferExtraction {
protected:
    const data_flow::ImplementationType& with_transfers() const override;
    const data_flow::ImplementationType& without_transfers() const override;
    std::string storage() const override;
    const std::string& device_prefix() const override;

    void add_transfer(
        builder::StructuredSDFGBuilder& builder,
        structured_control_flow::Block& block,
        const std::string& host_container,
        const std::string& device_container,
        offloading::DataTransferDirection direction,
        offloading::BufferLifecycle lifecycle,
        const types::IType& type,
        const symbolic::Expression& size
    ) override;

public:
    using GPUTensorDataTransferExtraction::GPUTensorDataTransferExtraction;

    std::string name() const override;
};

} // namespace sdfg::gpu::tensor
