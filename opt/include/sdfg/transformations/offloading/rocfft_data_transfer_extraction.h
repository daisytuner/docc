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
namespace rocm {

/**
 * @brief Extract the host<->device data transfers of a hipFFT FFT/IFFT node.
 *
 * ROCm counterpart of @ref sdfg::cuda::CUFFTDataTransferExtraction: rewrites a
 * `ImplementationType_ROCMWithTransfers` FFT/IFFT node into a
 * `ImplementationType_ROCMWithoutTransfers` node surrounded by explicit
 * offloading nodes (device alloc, H2D of `__X`, D2H of `__Y`, free), sizing the
 * complex side with the Hermitian layout.
 */
class ROCFFTDataTransferExtraction : public transformations::Transformation {
private:
    math::tensor::FFTNodeBase& fft_node_;

public:
    ROCFFTDataTransferExtraction(math::tensor::FFTNodeBase& fft_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& json) const override;
};

} // namespace rocm
} // namespace sdfg
