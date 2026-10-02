#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/einsum/einsum_node.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace einsum {

class Einsum2Dot : public transformations::Transformation {
private:
    EinsumNode& einsum_node_;

public:
    Einsum2Dot(EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Einsum2Dot from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace einsum
} // namespace sdfg
