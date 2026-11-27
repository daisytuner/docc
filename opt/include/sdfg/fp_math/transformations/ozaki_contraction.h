#pragma once

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/fp_math/transformations/quantize_contraction.h"
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
 * Both run in Double, the scales applied through their exponents (Rescaling::Exponents). With
 * Rescaling::Direct, slicing instead runs in the contraction's own type with nothing beyond the arithmetic
 * QuantizeContraction's direct rescaling needs, for targets without Double, floor or exact powers of two: with
 * `q = 2^7 - 1`, `s` the largest magnitude of a row of A (or column of B) and `r_1 = q x / s`, the slices are
 * `d_t = rint(r_t)` and `r_{t+1} = 2q (r_t - d_t)`, each in [-q, q], so `x = (s / q) sum_t (2q)^(1-t) d_t` up to the
 * last remainder, and `Y = (s_A[i] / q) (s_B[j] / q) sum_{t+u <= s+1} (2q)^(2-t-u) (A_t @ B_u)`. Each slice is computed
 * by a nest of its own that reads the slices before it back from memory, as a target needs that rounds only when it
 * stores Int8. Its accuracy is that of the contraction's type, and it applies to batched MatMuls too, with scales per
 * matrix.
 *
 * Accuracy is relative to the magnitudes of each row of A and column of B: elements much smaller than their row's
 * largest lose bits, as in all Ozaki schemes. The Int8 matmuls stay library nodes for the target to map.
 *
 * Without a given number of slices, each operand gets as many as its own precision needs, at most as many as the
 * result keeps (see operand_splits): an operand of fewer significand bits than the result, such as bfloat16 or FP8
 * weights, needs fewer slices, and a target that computes the result in fewer bits than its type (`max_bits`) needs
 * fewer for both. With direct rescaling, A and B then take slices of their own and the products (t, u) with t <= s_A,
 * u <= s_B and t + u <= max(s_A, s_B) + 1.
 *
 * Applies to a 2-D Float or Double MatMul, or a row-major GEMM (whose alpha and beta it applies to the result), with a
 * constant K small enough for exact Int32 accumulation; with direct rescaling, slicing only, also to a MatMul that
 * repeats over batch dimensions.
 */
class OzakiContraction : public transformations::Transformation {
private:
    data_flow::LibraryNode& node_;
    OzakiVariant variant_;
    int splits_;
    Rescaling rescaling_;
    int max_bits_;
    double min_normal_;

public:
    static constexpr int MAX_SLICES = 10;

    /// Odd primes below 256, largest first; the first MAX_MODULI serve as the moduli of the modular variant.
    static const std::vector<int64_t>& moduli();
    static constexpr int MAX_MODULI = 20;

    /**
     * @param splits slices (Slicing) or moduli (Modular); 0 picks as many as the operands' and the output's precision
     * need
     * @param rescaling Direct computes the slices and the result in the contraction's own type; Slicing only
     * @param max_bits the significand bits, guard bits included, the target computes the result in; 0 for the result
     * type's own
     * @param min_normal the smallest normal value of the float the target computes direct rescaling in, the least
     * scale of a row or column, so that the scales of all-zero ones stay invertible there; 0 for the type's own
     */
    explicit OzakiContraction(
        data_flow::LibraryNode& node,
        OzakiVariant variant = OzakiVariant::Modular,
        int splits = 0,
        Rescaling rescaling = Rescaling::Exponents,
        int max_bits = 0,
        double min_normal = 0
    );

    virtual std::string name() const override;

    virtual bool
    can_be_applied(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) override;

    virtual void to_json(nlohmann::json& j) const override;

    static OzakiContraction from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j);

    OzakiVariant variant() const {
        return variant_;
    }

    Rescaling rescaling() const {
        return rescaling_;
    }

    /// The slices or moduli the transformation uses for @p node, resolving 0 to the default: for slicing, the larger
    /// of operand_splits().
    int effective_splits() const;
    /**
     * The slices of A and of B, slicing: the given number for both, else per operand ceil(bits / 7) with
     * bits = min(significand of the operand's type + 2 guard bits, the result's bits), the result's bits being its
     * type's significand + 2, or max_bits if smaller. Integer operands of 8 bits or less need one slice.
     */
    std::pair<int, int> operand_splits() const;
    /// The bits operand @p connector ("A" or "B") needs represented, before the result's bounds; 0 if unknown.
    int operand_bits(const std::string& connector) const;

    /// Bits each of A and B keeps per row or column in the modular variant with @p moduli and reduction @p k.
    static int modular_bits(int moduli, int64_t k);

    /// Int8 matmuls the transformation computes.
    int matmuls() const;
};

} // namespace fp_math
} // namespace sdfg
