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
namespace cuda {

class CUBLASDataTransferExtraction : public transformations::Transformation {
private:
    math::blas::BLASNode& blas_node_;

public:
    CUBLASDataTransferExtraction(math::blas::BLASNode& blas_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& json) const override;

    static CUBLASDataTransferExtraction from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace cuda
} // namespace sdfg
