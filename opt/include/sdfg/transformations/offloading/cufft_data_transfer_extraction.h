#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/fft_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"

namespace sdfg {
namespace cuda {

/**
 * @brief Extract the host<->device data transfers of a cuFFT FFT/IFFT node.
 *
 * Turns a `ImplementationType_CUDAWithTransfers` FFT/IFFT node into a
 * `ImplementationType_CUDAWithoutTransfers` node surrounded by explicit
 * offloading nodes (device alloc, H2D of the input `__X`, D2H of the output
 * `__Y`, free). Buffer sizes use the Hermitian layout for the complex side.
 */
class CUFFTDataTransferExtraction : public transformations::Transformation {
private:
    math::tensor::FFTNodeBase& fft_node_;

public:
    CUFFTDataTransferExtraction(math::tensor::FFTNodeBase& fft_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& json) const override;
};

} // namespace cuda
} // namespace sdfg
