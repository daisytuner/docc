#pragma once

#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/fp_math/transformations/ozaki_contraction.h"
#include "sdfg/passes/pass.h"

namespace sdfg {
namespace fp_math {

/**
 * @brief How contractions (MatMul and GEMM nodes) of Float and Double compute, as users select it.
 *
 * | Spec           | Computation                                                             |
 * |----------------|-------------------------------------------------------------------------|
 * | `native`       | unchanged                                                               |
 * | `int8`         | QuantizeContraction with 8 bits (lossy); `int2` ... `int7` use fewer      |
 * | `ozaki`        | OzakiContraction, modular, as many moduli as the output precision needs  |
 * | `ozaki1[:s]`   | OzakiContraction, slicing with s slices (default: by precision)          |
 * | `ozaki2[:L]`   | OzakiContraction, modular with L moduli (default: by precision)          |
 */
struct ContractionPrecision {
    enum class Kind { Native, Quantize, Ozaki };

    Kind kind = Kind::Native;
    OzakiVariant variant = OzakiVariant::Modular;
    /// Bits for Quantize, slices or moduli for Ozaki; 0 picks the default.
    int parameter = 0;

    /// @throws std::invalid_argument for a malformed spec
    static ContractionPrecision parse(const std::string& spec);

    std::string str() const;
};

/// Applies the transformation @p precision selects to every contraction it can; others stay native.
class ContractionPrecisionPass : public passes::Pass {
private:
    ContractionPrecision precision_;

public:
    explicit ContractionPrecisionPass(const ContractionPrecision& precision);

    virtual std::string name() override;

    virtual bool run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;
};

} // namespace fp_math
} // namespace sdfg
