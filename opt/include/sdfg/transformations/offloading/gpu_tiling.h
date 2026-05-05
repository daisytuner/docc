#pragma once

#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class GPUTiling : public Transformation {
    structured_control_flow::StructuredLoop& loop_;
    size_t size_;
    bool applied_ = false;

    structured_control_flow::StructuredLoop* inner_loop_ = nullptr;
    structured_control_flow::StructuredLoop* outer_loop_ = nullptr;
    std::set<std::string> target_containers_;

public:
    GPUTiling(structured_control_flow::StructuredLoop& loop, size_t size);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static GPUTiling from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    structured_control_flow::StructuredLoop* inner_loop();
    structured_control_flow::StructuredLoop* outer_loop();
};

} // namespace transformations
} // namespace sdfg
