#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class HIPParallelizeNestedMap : public Transformation {
    structured_control_flow::Map& loop_;
    size_t block_size_;

public:
    HIPParallelizeNestedMap(structured_control_flow::Map& loop, size_t block_size);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static HIPParallelizeNestedMap from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};


} // namespace transformations
} // namespace sdfg
