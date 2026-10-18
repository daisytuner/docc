#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace fp_math {

/// How QuantizeContraction applies the scales of A's rows and B's columns to the Int32 products.
enum class Rescaling {
    /// Through the scales' power-of-two exponents in one step, so that no intermediate overflows where the result does
    /// not (needs floor and exact powers of two).
    Exponents,
    /// Multiplied by `s_A[i] / q` and then `s_B[j] / q`, which needs nothing beyond multiplication but overflows for a
    /// row whose magnitude lies within a factor K q of the type's largest value, about 10^33 for Float and K = 768.
    Direct,
};

/**
 * @brief Computes a floating-point matmul (MatMul or GEMM node) on integers, quantized with a scale per row of A and
 * per column of B.
 *
 * With `q = 2^(bits-1) - 1`, `s_A[i] = max_k |A[i, k]|` and `s_B[j] = max_k |B[k, j]|`:
 *
 *     A_q[i, k] = rint(q * A[i, k] / s_A[i])          (Int8)
 *     B_q[k, j] = rint(q * B[k, j] / s_B[j])          (Int8)
 *     P         = A_q @ B_q                            (Int8 -> Int32 MatMul, exact)
 *     Y[i, j]   = P[i, j] * (s_A[i] / q) * (s_B[j] / q)
 *
 * The scales are computed at run time, so the result changes with every rounding of A and B: about 1% relative error
 * for 8 bits. The Int8 matmul stays a library node for the target to map, e.g. onto integer matrix units.
 *
 * Applies to a Float or Double MatMul, also one repeating over batch dimensions with scales per matrix, or a row-major
 * GEMM (whose alpha and beta it applies to the result), with a constant K small enough for exact Int32 accumulation.
 */
class QuantizeContraction : public transformations::Transformation {
private:
    data_flow::LibraryNode& node_;
    int bits_;
    Rescaling rescaling_;

public:
    explicit QuantizeContraction(data_flow::LibraryNode& node, int bits = 8, Rescaling rescaling = Rescaling::Exponents);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static QuantizeContraction from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    int bits() const {
        return bits_;
    }

    Rescaling rescaling() const {
        return rescaling_;
    }
};

} // namespace fp_math
} // namespace sdfg
