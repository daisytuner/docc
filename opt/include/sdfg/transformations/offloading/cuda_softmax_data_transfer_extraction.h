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

public:
    explicit CUDASoftmaxDataTransferExtraction(math::tensor::SoftmaxNode& softmax_node);

    std::string name() const override;

    bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    void to_json(nlohmann::json& j) const override;
};

} // namespace cuda
} // namespace sdfg
