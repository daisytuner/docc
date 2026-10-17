#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace fp_math {

/**
 * @brief Splits a fused multiply-add into a multiplication and an addition.
 *
 * Replaces an `fp_fma` tasklet `d = fma(a, b, c)` by `t = a * b` and `d = t + c`, where `t` is a new transient
 * scalar of the result's precision. This is the inverse of @ref FuseFMA and reintroduces the intermediate rounding
 * of the product, e.g. so that a later transformation can hand `t` on to another computation.
 */
class UnfuseFMA : public transformations::Transformation {
private:
    data_flow::Tasklet& fma_;

    data_flow::Tasklet* multiply_ = nullptr;
    data_flow::Tasklet* add_ = nullptr;
    std::string product_;

public:
    explicit UnfuseFMA(data_flow::Tasklet& fma);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static UnfuseFMA from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// The `fp_mul` tasklet created by apply(), or nullptr before.
    data_flow::Tasklet* multiply() const;

    /// The `fp_add` tasklet created by apply(), or nullptr before.
    data_flow::Tasklet* add() const;

    /// The transient scalar holding the product, empty before apply().
    const std::string& product() const;
};

} // namespace fp_math
} // namespace sdfg
