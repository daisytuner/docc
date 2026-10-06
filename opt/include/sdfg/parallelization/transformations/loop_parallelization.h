#pragma once

#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Converts a single For loop into a sequential Map
 *
 * Applies the legality criteria of the AutoParallelization to one loop: the
 * iterations must be independent (no loop-carried hazards, false dependencies
 * confined to loop-local storage). Unlike the pass, it targets a single loop, so
 * it can be recorded and replayed, e.g. after a wavefront interchange.
 */
class LoopParallelization : public Transformation {
    structured_control_flow::For& loop_;
    structured_control_flow::Map* map_ = nullptr;

public:
    LoopParallelization(structured_control_flow::For& loop);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static LoopParallelization from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// The created Map (only valid after apply)
    structured_control_flow::Map* map() const;
};

} // namespace transformations
} // namespace sdfg
