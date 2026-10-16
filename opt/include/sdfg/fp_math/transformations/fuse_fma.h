#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace fp_math {

/**
 * @brief Contracts a floating-point multiplication feeding an addition into a fused multiply-add.
 *
 * Anchored at an `fp_add` tasklet `d = t + c` (or `d = c + t`) whose operand `t` is a transient scalar that is
 * written only by an `fp_mul` tasklet `t = a * b` of the same block and read only by this addition. The pair is
 * replaced by a single `fp_fma` tasklet `d = fma(a, b, c)`; the product container is removed.
 *
 * If both operands of the addition are such products, the one on `_in1` is contracted.
 * All operands, the product and the result must share the same floating-point precision.
 *
 * Contraction skips the intermediate rounding of the product, so it is only valid under `-ffp-contract=fast`
 * semantics. @see UnfuseFMA for the inverse.
 */
class FuseFMA : public transformations::Transformation {
private:
    data_flow::Tasklet& add_;
    data_flow::Tasklet* fma_ = nullptr;
    bool invalidate_analysis_;

    struct Match {
        data_flow::Tasklet* multiply = nullptr;
        data_flow::AccessNode* product = nullptr;
        data_flow::Memlet* product_edge = nullptr;
        data_flow::Memlet* addend_edge = nullptr;
    };

    bool match(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, Match& out);

public:
    explicit FuseFMA(data_flow::Tasklet& add, bool invalidate_analysis = true);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static FuseFMA from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    /// The fused multiply-add tasklet created by apply(), or nullptr before.
    data_flow::Tasklet* fma() const;
};

} // namespace fp_math
} // namespace sdfg
