#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/transformations/transformation.h"

namespace sdfg {
namespace fp_math {

/// How the Ozaki scheme represents the operands in Int8.
enum class OzakiVariant {
    /// Ozaki-I: split each operand into 7-bit slices and sum the s(s+1)/2 leading products of slices.
    Slicing,
    /// Ozaki-II: scale the operands to integers, multiply their residues modulo coprime moduli and reconstruct the
    /// product with the Chinese remainder theorem; one product per modulus.
    Modular,
};

/**
 * @brief Emulates a floating-point matmul (MatMul or GEMM node) with exact Int8 matmuls (Ozaki scheme), at about the
 * input's precision.
 *
 * Every row of A and column of B is scaled by a power of two, which is exact, and represented in Int8 matrices whose
 * products the target computes exactly with Int32 accumulation:
 *
 * - **Slicing (Ozaki-I).** `A[i, :] / 2^e_i = sum_t 2^-7t A_t[i, :]` with Int8 slices A_t, and likewise for B.
 *   `Y = 2^(e_i + f_j) sum_{t+u <= s+1} 2^-7(t+u) (A_t @ B_u)`, dropping the products below the precision.
 * - **Modular (Ozaki-II).** `A' = trunc(2^q A[i, :] / 2^e_i)` and `B'` are integers below 2^q; their product is exact
 *   modulo `P = prod m_l` as long as `K 2^2q < P / 2`. Each modulus takes one product of the Int8 residues; Garner's
 *   algorithm turns the residues of the result into mixed-radix digits, which are summed in Double.
 *
 * Accuracy is relative to the magnitudes of each row of A and column of B: elements much smaller than their row's
 * largest lose bits, as in all Ozaki schemes. The Int8 matmuls stay library nodes for the target to map.
 *
 * Applies to a 2-D Float or Double MatMul, or a row-major GEMM (whose alpha and beta it applies to the result), with a
 * constant K small enough for exact Int32 accumulation.
 */
class OzakiContraction : public transformations::Transformation {
private:
    data_flow::LibraryNode& node_;
    OzakiVariant variant_;
    int splits_;

public:
    static constexpr int MAX_SLICES = 10;

    /// Odd primes below 256, largest first; the first MAX_MODULI serve as the moduli of the modular variant.
    static const std::vector<int64_t>& moduli();
    static constexpr int MAX_MODULI = 20;

    /**
     * @param splits slices (Slicing) or moduli (Modular); 0 picks as many as the output precision needs
     */
    explicit OzakiContraction(data_flow::LibraryNode& node, OzakiVariant variant = OzakiVariant::Modular, int splits = 0);

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static OzakiContraction from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    OzakiVariant variant() const {
        return variant_;
    }

    /// The slices or moduli the transformation uses for @p node, resolving 0 to the default.
    int effective_splits() const;

    /// Bits each of A and B keeps per row or column in the modular variant with @p moduli and reduction @p k.
    static int modular_bits(int moduli, int64_t k);

    /// Int8 matmuls the transformation computes.
    int matmuls() const;
};

} // namespace fp_math
} // namespace sdfg
