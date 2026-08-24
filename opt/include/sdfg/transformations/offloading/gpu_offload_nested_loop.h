#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

template<typename GPUType>
class GPUOffloadNestedLoop : public Transformation {
    gpu::TargetLevel target_level_;
    structured_control_flow::StructuredLoop& loop_;
    symbolic::Integer parallel_size_;

public:
    GPUOffloadNestedLoop(
        structured_control_flow::StructuredLoop& loop, gpu::TargetLevel target_level, symbolic::Integer parallel_size
    );

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static GPUOffloadNestedLoop<GPUType> from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};


} // namespace transformations
} // namespace sdfg
