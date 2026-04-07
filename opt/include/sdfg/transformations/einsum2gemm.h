#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/einsum/einsum.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace transformations {

class Einsum2Gemm : public Transformation {
private:
    einsum::EinsumNode& einsum_node_;

    bool check_matrix_indices(long long mat, const symbolic::Symbol& indvar1, const symbolic::Symbol& indvar2);

public:
    Einsum2Gemm(einsum::EinsumNode& einsum_node);

    virtual std::string name() const override;

    virtual bool can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager)
        override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static Einsum2Gemm from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);
};

} // namespace transformations
} // namespace sdfg
