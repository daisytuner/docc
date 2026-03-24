#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class EinsumExpand : public Transformation {
private:
    einsum::EinsumNode& einsum_node_;

    symbolic::Expression cnf_to_upper_bound(const symbolic::CNF& cnf, const symbolic::Symbol indvar);

    bool subset_contains_symbol(const data_flow::Subset& subset, const symbolic::Symbol& symbol);

public:
    EinsumExpand(einsum::EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static EinsumExpand from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
