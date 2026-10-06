#pragma once

#include <nlohmann/json.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/embedding_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/type.h"

namespace sdfg::gpu::tensor {

/**
 * @brief Moves the transfers of a `WithTransfers` embedding node into explicit offloading blocks.
 *
 * `W` and `I` are copied to device containers, the fully overwritten `Y` is only
 * allocated and copied back. The node is rewired to the device containers and
 * switched to the `WithoutTransfers` implementation, so later passes can minimize
 * the transfers.
 */
class GPUEmbeddingDataTransferExtraction : public transformations::Transformation {
protected:
    math::tensor::EmbeddingNode& embedding_node_;

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

    std::string create_device_container(
        builder::StructuredSDFGBuilder& builder, const std::string& host_container, const symbolic::Expression& size
    );

public:
    explicit GPUEmbeddingDataTransferExtraction(math::tensor::EmbeddingNode& embedding_node);

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;
};

class CUDAEmbeddingDataTransferExtraction : public GPUEmbeddingDataTransferExtraction {
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
    using GPUEmbeddingDataTransferExtraction::GPUEmbeddingDataTransferExtraction;

    std::string name() const override;
};

class ROCMEmbeddingDataTransferExtraction : public GPUEmbeddingDataTransferExtraction {
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
    using GPUEmbeddingDataTransferExtraction::GPUEmbeddingDataTransferExtraction;

    std::string name() const override;
};

} // namespace sdfg::gpu::tensor
