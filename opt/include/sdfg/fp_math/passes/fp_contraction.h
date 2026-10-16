#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace fp_math {

/**
 * @brief Floating-point contraction (`-ffp-contract=fast`): applies @ref FuseFMA in every block until no
 * multiplication feeding an addition is left to contract.
 */
class FPContraction : public visitor::NonStoppingStructuredSDFGVisitor {
public:
    FPContraction(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager);

    static std::string name() {
        return "FPContraction";
    }

    virtual bool visit() override;

    virtual bool accept(structured_control_flow::Block& block) override;
};

typedef passes::VisitorPass<FPContraction> FPContractionPass;

} // namespace fp_math
} // namespace sdfg
