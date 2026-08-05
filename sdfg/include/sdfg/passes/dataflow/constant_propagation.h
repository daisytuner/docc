#pragma once

#include "sdfg/passes/pass.h"

namespace sdfg {
namespace passes {

/**
 * @brief Forward-propagates floating-point constant scalars into their uses.
 *
 * A scalar container that is defined by a single constant assignment
 * (`ConstantNode -> assign -> scalar`) is treated as holding that constant. Every downstream
 * computational read of the scalar is rewritten to read the constant directly, which in turn
 * enables library-node simplifications (e.g. GEMM alpha == 1 / beta == 0) and later folding.
 */
class ConstantPropagation : public Pass {
public:
    ConstantPropagation();

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace passes
} // namespace sdfg
