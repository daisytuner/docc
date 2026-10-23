#pragma once

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

/**
 * @brief Eliminates a trivial map, i.e., a map executing exactly one iteration.
 *
 * Original:
 *   map (i = init; i < init + 1; i++)
 *       body(i)
 *
 * After TrivialLoopElimination:
 *   body(init)
 *
 * Symbolic uses of the induction variable are replaced by init. Data accesses
 * to the induction variable container are replaced by a constant (integer init),
 * renamed (symbolic init) or kept and fed by an assignment `i = init` otherwise.
 *
 * Prerequisites:
 * - Sequential schedule (non-sequential maps carry semantics beyond their
 *   iterations, e.g., a single-wave GPU map defines 64 lanes)
 * - Unit stride
 * - Trip count is exactly 1
 * - Parent is a sequence
 */
class TrivialLoopElimination : public Transformation {
    structured_control_flow::Map& loop_;

public:
    explicit TrivialLoopElimination(structured_control_flow::Map& loop);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static TrivialLoopElimination from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
