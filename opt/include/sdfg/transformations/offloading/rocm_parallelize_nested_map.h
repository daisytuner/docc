#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Assigns a nested sequential map to the next available ROCm grid dimension.
 *
 * This transformation does not perform blocking or tiling on its own. It expects
 * the scheduler to have already identified a suitable nested map within an existing
 * ROCm-scheduled map. The transformation simply promotes the map's schedule from
 * sequential to ROCm, assigning the next available dimension (X->Y or Y->Z).
 *
 * The resulting grid dimension is validated against ROCm/HIP hardware limits:
 * Y and Z dimensions are limited to 65535 blocks. If the grid would exceed this
 * limit, the transformation is rejected (can_be_applied returns false).
 */
class ROCMParallelizeNestedMap : public Transformation {
    structured_control_flow::Map& loop_;
    size_t block_size_;

public:
    ROCMParallelizeNestedMap(structured_control_flow::Map& loop, size_t block_size);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static ROCMParallelizeNestedMap from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};


} // namespace transformations
} // namespace sdfg
