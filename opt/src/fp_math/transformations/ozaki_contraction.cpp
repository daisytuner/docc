#include "sdfg/fp_math/transformations/ozaki_contraction.h"

#include <cmath>
#include <functional>
#include <optional>
#include <tuple>

#include <nlohmann/json.hpp>

#include "../contractions/contraction_kernels.h"
#include "sdfg/fp_math/contractions/row_major_operand.h"

namespace sdfg {
namespace fp_math {

using data_flow::TaskletCode;
using math::cmath::CMathFunction;
using types::PrimitiveType;

namespace {

constexpr int SLICE_BITS = 7;

/// Bits of the output's significand plus guard bits, which the emulation keeps per row and column.
int target_bits(PrimitiveType type) {
    return type == PrimitiveType::Float ? 26 : 55;
}

/// The inverse of @p value modulo @p modulus.
int64_t inverse_mod(int64_t value, int64_t modulus) {
    int64_t r0 = modulus, r1 = ((value % modulus) + modulus) % modulus;
    int64_t t0 = 0, t1 = 1;
    while (r1 != 0) {
        int64_t quotient = r0 / r1;
        std::tie(r0, r1) = std::make_pair(r1, r0 - quotient * r1);
        std::tie(t0, t1) = std::make_pair(t1, t0 - quotient * t1);
    }
    return ((t0 % modulus) + modulus) % modulus;
}

symbolic::Expression
flat(const symbolic::Expression& row, const symbolic::Expression& columns, const symbolic::Expression& column) {
    return symbolic::add(symbolic::mul(row, columns), column);
}

} // namespace

const std::vector<int64_t>& OzakiContraction::moduli() {
    static const std::vector<int64_t> primes = {251, 241, 239, 233, 229, 227, 223, 211, 199, 197,
                                                193, 191, 181, 179, 173, 167, 163, 157, 151, 149};
    return primes;
}

OzakiContraction::OzakiContraction(data_flow::LibraryNode& node, OzakiVariant variant, int splits, Rescaling rescaling)
    : node_(node), variant_(variant), splits_(splits), rescaling_(rescaling) {
}

std::string OzakiContraction::name() const {
    return "OzakiContraction";
}

int OzakiContraction::modular_bits(int moduli, int64_t k) {
    double log2_p = 0;
    for (int l = 0; l < moduli; l++) {
        log2_p += std::log2(static_cast<double>(OzakiContraction::moduli().at(l)));
    }
    // |sum_k A' B'| < K 2^2q must stay below P / 4, inside the symmetric range (-P/2, P/2); A' fits Int64.
    int bits = static_cast<int>(std::floor((log2_p - 2 - std::log2(static_cast<double>(k))) / 2));
    return std::min(bits, 62);
}

int OzakiContraction::effective_splits() const {
    if (splits_ != 0) {
        return splits_;
    }
    auto target = target_bits(detail::contraction_type(node_));
    if (variant_ == OzakiVariant::Slicing) {
        return (target + SLICE_BITS - 1) / SLICE_BITS;
    }
    auto k_expression = detail::contraction_k(node_);
    if (k_expression.is_null() || !SymEngine::is_a<SymEngine::Integer>(*k_expression)) {
        return MAX_MODULI;
    }
    auto k = detail::constant_k(node_);
    for (int moduli = 2; moduli < MAX_MODULI; moduli++) {
        if (modular_bits(moduli, k) >= target) {
            return moduli;
        }
    }
    return MAX_MODULI;
}

int OzakiContraction::matmuls() const {
    auto splits = effective_splits();
    return variant_ == OzakiVariant::Slicing ? splits * (splits + 1) / 2 : splits;
}

bool OzakiContraction::can_be_applied(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&) {
    bool direct = rescaling_ == Rescaling::Direct;
    if (direct && variant_ != OzakiVariant::Slicing) {
        return false;
    }
    int64_t magnitude = variant_ == OzakiVariant::Slicing ? (1 << SLICE_BITS) - 1 : (moduli().front() - 1) / 2;
    if (!detail::unsupported_contraction(node_, detail::exact_int32_reduction(magnitude), direct).empty()) {
        return false;
    }
    auto splits = effective_splits();
    if (variant_ == OzakiVariant::Slicing) {
        return splits >= 1 && splits <= MAX_SLICES;
    }
    return splits >= 2 && splits <= MAX_MODULI && modular_bits(splits, detail::constant_k(node_)) >= 1;
}

namespace {

/// Row scales of A and column scales of B: exponents `E = e` and inverse powers `I = 2^-e` with every |x| of the row
/// or column below 2^e.
struct OperandScales {
    std::string a_exponent, a_inverse, b_exponent, b_inverse;
};

OperandScales operand_scales(KernelBuilder& kb, detail::ContractionOperands& operands) {
    auto tiny = detail::smallest_normal(operands.element);
    auto read_a = [&](BlockBuilder& block, auto& entry, auto& row, auto& column) {
        return operands.a(block, entry, row, column);
    };
    auto read_b_t = [&](BlockBuilder& block, auto& entry, auto& column, auto& row) {
        return operands.b(block, entry, row, column);
    };
    auto magnitudes_a = detail::absmax(kb, "_oz_max_a", read_a, operands.m, operands.k, tiny, PrimitiveType::Double);
    auto magnitudes_b = detail::absmax(kb, "_oz_max_b", read_b_t, operands.n, operands.k, tiny, PrimitiveType::Double);
    auto [a_exponent, a_inverse] =
        detail::power_of_two_scales(kb, "_oz_a", magnitudes_a, operands.m, PrimitiveType::Double);
    auto [b_exponent, b_inverse] =
        detail::power_of_two_scales(kb, "_oz_b", magnitudes_b, operands.n, PrimitiveType::Double);
    return {a_exponent, a_inverse, b_exponent, b_inverse};
}

/// For every element x of A (or B) in [-1, 1) after scaling by its inverse power, @p emit derives Int8 values.
void split_operands(
    KernelBuilder& kb,
    detail::ContractionOperands& operands,
    const OperandScales& scales,
    const std::function<void(BlockBuilder&, const KernelValue&, bool, const symbolic::Expression&)>& emit
) {
    auto m = operands.m, n = operands.n, k = operands.k;
    {
        auto& rows = kb.map(kb.scope(), "_i", m);
        auto& columns = kb.map(rows.root(), "_k", k);
        auto i = rows.indvar(), c = columns.indvar();
        auto body = kb.block(columns.root());
        auto x = body.convert(operands.a(body, {}, i, c), PrimitiveType::Double);
        auto normalized = body.op(TaskletCode::fp_mul, {x, body.read(scales.a_inverse, {i})}, PrimitiveType::Double);
        emit(body, normalized, true, flat(i, k, c));
    }
    {
        auto& rows = kb.map(kb.scope(), "_k", k);
        auto& columns = kb.map(rows.root(), "_j", n);
        auto r = rows.indvar(), j = columns.indvar();
        auto body = kb.block(columns.root());
        auto x = body.convert(operands.b(body, {}, r, j), PrimitiveType::Double);
        auto normalized = body.op(TaskletCode::fp_mul, {x, body.read(scales.b_inverse, {j})}, PrimitiveType::Double);
        emit(body, normalized, false, flat(r, n, j));
    }
}

/// `Y[i, j] = value(i, j) * 2^(E_A[i] + E_B[j] + shift)`, undoing the power-of-two scaling in one step, so that no
/// intermediate overflows where the result does not.
void store_scaled(
    KernelBuilder& kb,
    detail::ContractionOperands& operands,
    const OperandScales& scales,
    const std::function<KernelValue(BlockBuilder&, const symbolic::Expression&, const symbolic::Expression&)>& value,
    int shift = 0
) {
    auto& rows = kb.map(kb.scope(), "_i", operands.m);
    auto& columns = kb.map(rows.root(), "_j", operands.n);
    auto i = rows.indvar(), j = columns.indvar();
    auto body = kb.block(columns.root());
    auto exponent = body
                        .op(TaskletCode::fp_add,
                            {body.read(scales.a_exponent, {i}), body.read(scales.b_exponent, {j})},
                            PrimitiveType::Double);
    if (shift != 0) {
        exponent =
            body.op(TaskletCode::fp_add, {exponent, body.constant(shift, PrimitiveType::Double)}, PrimitiveType::Double);
    }
    operands
        .store(body, detail::scale_by_power_of_two(body, value(body, i, j), exponent, PrimitiveType::Double), {}, i, j);
}

void build_slicing(KernelBuilder& kb, detail::ContractionOperands& operands, int slices) {
    auto m = operands.m, n = operands.n, k = operands.k;
    auto scales = operand_scales(kb, operands);

    std::vector<std::string> a_slices, b_slices;
    for (int t = 0; t < slices; t++) {
        a_slices.push_back(kb.buffer("_oz_a_slice", PrimitiveType::Int8, symbolic::mul(m, k)));
        b_slices.push_back(kb.buffer("_oz_b_slice", PrimitiveType::Int8, symbolic::mul(k, n)));
    }
    // Peel 7 bits per slice: x = 2^7 r, d = trunc(x) in [-127, 127], r = x - d; all exact in Double.
    split_operands(kb, operands, scales, [&](BlockBuilder& body, const KernelValue& normalized, bool is_a, auto& index) {
        auto rest = normalized;
        for (int t = 0; t < slices; t++) {
            auto shifted = body
                               .op(TaskletCode::fp_mul,
                                   {rest, body.constant(1 << SLICE_BITS, PrimitiveType::Double)},
                                   PrimitiveType::Double);
            auto digit = body.cmath(CMathFunction::trunc, {shifted}, PrimitiveType::Double);
            body.store(digit, (is_a ? a_slices : b_slices).at(t), {index});
            if (t + 1 < slices) {
                rest = body.op(TaskletCode::fp_sub, {shifted, digit}, PrimitiveType::Double);
            }
        }
    });

    auto sums = kb.buffer("_oz_sum", PrimitiveType::Double, symbolic::mul(m, n));
    {
        auto& rows = kb.map(kb.scope(), "_i", m);
        auto& columns = kb.map(rows.root(), "_j", n);
        auto body = kb.block(columns.root());
        body.store(body.constant(0.0, PrimitiveType::Double), sums, {flat(rows.indvar(), n, columns.indvar())});
    }

    // Smallest contributions first: slice pairs (t, u), 1-based, by descending t + u up to s + 1.
    auto products = kb.buffer("_oz_p", PrimitiveType::Int32, symbolic::mul(m, n));
    for (int g = slices + 1; g >= 2; g--) {
        for (int t = std::max(1, g - slices); t <= std::min(slices, g - 1); t++) {
            detail::int8_matmul(kb, a_slices.at(t - 1), b_slices.at(g - t - 1), products, m, n, k);
            auto& rows = kb.map(kb.scope(), "_i", m);
            auto& columns = kb.map(rows.root(), "_j", n);
            auto index = flat(rows.indvar(), n, columns.indvar());
            auto body = kb.block(columns.root());
            auto product = body.convert(body.read(products, {index}), PrimitiveType::Double);
            auto weighted = body
                                .op(TaskletCode::fp_mul,
                                    {product, body.constant(std::ldexp(1.0, -SLICE_BITS * g), PrimitiveType::Double)},
                                    PrimitiveType::Double);
            body.store(
                body.op(TaskletCode::fp_add, {body.read(sums, {index}), weighted}, PrimitiveType::Double), sums, {index}
            );
        }
    }

    store_scaled(kb, operands, scales, [&](BlockBuilder& body, auto& i, auto& j) {
        return body.read(sums, {flat(i, n, j)});
    });
}

void build_modular(KernelBuilder& kb, detail::ContractionOperands& operands, int moduli_count) {
    auto m = operands.m, n = operands.n, k = operands.k;
    auto bits =
        OzakiContraction::modular_bits(moduli_count, SymEngine::down_cast<const SymEngine::Integer&>(*k).as_int());
    std::vector<int64_t> moduli(OzakiContraction::moduli().begin(), OzakiContraction::moduli().begin() + moduli_count);
    auto scales = operand_scales(kb, operands);

    std::vector<std::string> a_residues, b_residues;
    for (int l = 0; l < moduli_count; l++) {
        a_residues.push_back(kb.buffer("_oz_a_res", PrimitiveType::Int8, symbolic::mul(m, k)));
        b_residues.push_back(kb.buffer("_oz_b_res", PrimitiveType::Int8, symbolic::mul(k, n)));
    }
    // x' = trunc(2^q x) is an integer below 2^q, exact in Double and Int64; its residues fit Int8.
    split_operands(kb, operands, scales, [&](BlockBuilder& body, const KernelValue& normalized, bool is_a, auto& index) {
        auto scaled = body
                          .op(TaskletCode::fp_mul,
                              {normalized, body.constant(std::ldexp(1.0, bits), PrimitiveType::Double)},
                              PrimitiveType::Double);
        auto integer =
            body.convert(body.cmath(CMathFunction::trunc, {scaled}, PrimitiveType::Double), PrimitiveType::Int64);
        for (int l = 0; l < moduli_count; l++) {
            auto residue = detail::symmetric_mod(body, integer, moduli.at(l), PrimitiveType::Int64);
            body.store(residue, (is_a ? a_residues : b_residues).at(l), {index});
        }
    });

    // Residues of the integer product, one Int8 matmul per modulus.
    auto products = kb.buffer("_oz_p", PrimitiveType::Int32, symbolic::mul(m, n));
    std::vector<std::string> residues;
    for (int l = 0; l < moduli_count; l++) {
        detail::int8_matmul(kb, a_residues.at(l), b_residues.at(l), products, m, n, k);
        residues.push_back(kb.buffer("_oz_y_res", PrimitiveType::Int8, symbolic::mul(m, n)));
        auto& rows = kb.map(kb.scope(), "_i", m);
        auto& columns = kb.map(rows.root(), "_j", n);
        auto index = flat(rows.indvar(), n, columns.indvar());
        auto body = kb.block(columns.root());
        auto residue = detail::symmetric_mod(body, body.read(products, {index}), moduli.at(l), PrimitiveType::Int32);
        body.store(residue, residues.back(), {index});
    }

    // Garner: symmetric mixed-radix digits v_l of the product, y = v_1 + m_1 (v_2 + m_2 (v_3 + ...)), which covers
    // (-P/2, P/2) exactly; the Horner sum in Double is accurate to a few ulps.
    store_scaled(
        kb,
        operands,
        scales,
        [&](BlockBuilder& body, auto& i, auto& j) {
            auto index = flat(i, n, j);
            std::vector<KernelValue> digits;
            for (int l = 0; l < moduli_count; l++) {
                auto digit = body.convert(body.read(residues.at(l), {index}), PrimitiveType::Int32);
                for (int p = 0; p < l; p++) {
                    auto difference = body.op(TaskletCode::int_sub, {digit, digits.at(p)}, PrimitiveType::Int32);
                    auto scaled =
                        body
                            .op(TaskletCode::int_mul,
                                {difference,
                                 body.constant(inverse_mod(moduli.at(p), moduli.at(l)), PrimitiveType::Int32)},
                                PrimitiveType::Int32);
                    digit = detail::symmetric_mod(body, scaled, moduli.at(l), PrimitiveType::Int32);
                }
                digits.push_back(digit);
            }
            auto value = body.convert(digits.back(), PrimitiveType::Double);
            for (int l = moduli_count - 2; l >= 0; l--) {
                auto shifted = body
                                   .op(TaskletCode::fp_mul,
                                       {value, body.constant(moduli.at(l), PrimitiveType::Double)},
                                       PrimitiveType::Double);
                value = body
                            .op(TaskletCode::fp_add,
                                {body.convert(digits.at(l), PrimitiveType::Double), shifted},
                                PrimitiveType::Double);
            }
            return value;
        },
        -2 * bits
    );
}

/// The slices of an operand in its storage order, each from x and the slices before it, read back from memory, as
/// slicing with direct rescaling computes them; see OzakiContraction.
std::vector<std::string> slice_direct(
    KernelBuilder& kb,
    const detail::Batch& batch,
    PrimitiveType type,
    int slices,
    const std::string& prefix,
    const symbolic::Expression& rows,
    const symbolic::Expression& columns,
    bool transposed,
    const detail::MatrixRead& read,
    const std::string& scales,
    bool per_row
) {
    const double q = (1 << SLICE_BITS) - 1;
    auto size = symbolic::mul(rows, columns);
    std::vector<std::string> digits;
    for (int t = 0; t < slices; t++) {
        auto digit = kb.buffer(prefix, PrimitiveType::Int8, symbolic::mul(batch.count(), size));
        auto matrices = batch.maps(kb, kb.scope());
        auto& entry = matrices.second;
        auto position = batch.position(entry);
        auto& outer = kb.map(*matrices.first, "_o", transposed ? columns : rows);
        auto& inner = kb.map(outer.root(), "_e", transposed ? rows : columns);
        auto r = transposed ? inner.indvar() : outer.indvar();
        auto c = transposed ? outer.indvar() : inner.indvar();
        auto body = kb.block(inner.root());
        auto scale = symbolic::add(symbolic::mul(position, per_row ? rows : columns), per_row ? r : c);
        auto index = symbolic::
            add(symbolic::mul(position, size),
                transposed ? symbolic::add(symbolic::mul(c, rows), r) : symbolic::add(symbolic::mul(r, columns), c));
        // r_1 = q x / s, dividing first, which never overflows, unlike q / s for small s.
        auto ratio = body
                         .op(TaskletCode::fp_div,
                             {body.convert(read(body, entry, r, c), type), body.read(scales, {scale})},
                             type);
        auto rest = body.op(TaskletCode::fp_mul, {ratio, body.constant(q, type)}, type);
        for (auto& previous : digits) {
            auto difference =
                body.op(TaskletCode::fp_sub, {rest, body.convert(body.read(previous, {index}), type)}, type);
            rest = body.op(TaskletCode::fp_mul, {difference, body.constant(2 * q, type)}, type);
        }
        body.store(body.cmath(CMathFunction::rint, {rest}, type), digit, {index});
        digits.push_back(digit);
    }
    return digits;
}

/// The Int8 products of the slices of A and B, rescaled into the result; see OzakiContraction.
void direct_products(
    KernelBuilder& kb,
    detail::ContractionOperands& operands,
    int slices,
    const std::vector<std::string>& a_slices,
    const std::string& scales_a,
    const std::vector<std::string>& b_slices,
    const std::string& scales_b,
    bool b_transposed
) {
    auto m = operands.m, n = operands.n, k = operands.k;
    auto& batch = operands.batch;
    auto type = operands.element;
    const double q = (1 << SLICE_BITS) - 1;

    // The products of slice pairs (t, u), 1-based, t + u <= s + 1, each with its weight (2q)^(2-t-u); smallest first.
    auto size = symbolic::mul(batch.count(), symbolic::mul(m, n));
    std::vector<std::pair<std::string, double>> products;
    for (int g = slices + 1; g >= 2; g--) {
        for (int t = std::max(1, g - slices); t <= std::min(slices, g - 1); t++) {
            auto product = kb.buffer("_oz_p", PrimitiveType::Int32, size);
            detail::int8_matmul(
                kb,
                a_slices.at(t - 1),
                b_slices.at(g - t - 1),
                product,
                m,
                n,
                k,
                operands.a_transposed,
                b_transposed,
                batch
            );
            products.emplace_back(product, std::pow(2 * q, 2 - g));
        }
    }

    auto matrices = batch.maps(kb, kb.scope());
    auto position = batch.position(matrices.second);
    auto& rows = kb.map(*matrices.first, "_i", m);
    auto& columns = kb.map(rows.root(), "_j", n);
    auto i = rows.indvar(), j = columns.indvar();
    auto body = kb.block(columns.root());
    auto at = symbolic::add(symbolic::mul(position, symbolic::mul(m, n)), flat(i, n, j));
    std::optional<KernelValue> sum;
    for (auto& [product, weight] : products) {
        auto value = body.convert(body.read(product, {at}), type);
        if (weight != 1.0) {
            value = body.op(TaskletCode::fp_mul, {value, body.constant(weight, type)}, type);
        }
        sum = sum ? body.op(TaskletCode::fp_add, {*sum, value}, type) : value;
    }
    auto inverse_q = body.constant(1.0 / q, type);
    auto row_scale = body
                         .op(TaskletCode::fp_mul,
                             {body.read(scales_a, {symbolic::add(symbolic::mul(position, m), i)}), inverse_q},
                             type);
    auto column_scale = body
                            .op(TaskletCode::fp_mul,
                                {body.read(scales_b, {symbolic::add(symbolic::mul(position, n), j)}), inverse_q},
                                type);
    auto scaled = body.op(TaskletCode::fp_mul, {*sum, row_scale}, type);
    operands.store(body, body.op(TaskletCode::fp_mul, {scaled, column_scale}, type), matrices.second, i, j);
}

/// Slicing with direct rescaling, in the contraction's own type; see OzakiContraction.
void build_slicing_direct(KernelBuilder& kb, detail::ContractionOperands& operands, int slices) {
    auto m = operands.m, n = operands.n, k = operands.k;
    // Each matrix of a batch gets scales of its own; every buffer holds one matrix after the other.
    auto& batch = operands.batch;
    auto type = operands.element;
    auto tiny = detail::smallest_normal(type);
    auto read_a = [&](BlockBuilder& block, auto& entry, auto& row, auto& column) {
        return operands.a(block, entry, row, column);
    };
    auto read_b = [&](BlockBuilder& block, auto& entry, auto& row, auto& column) {
        return operands.b(block, entry, row, column);
    };
    auto read_b_t = [&](BlockBuilder& block, auto& entry, auto& column, auto& row) {
        return operands.b(block, entry, row, column);
    };
    auto scales_a = detail::absmax(kb, "_oz_scale_a", read_a, m, k, tiny, type, batch);
    auto scales_b = detail::absmax(kb, "_oz_scale_b", read_b_t, n, k, tiny, type, batch);
    auto a_slices =
        slice_direct(kb, batch, type, slices, "_oz_a_slice", m, k, operands.a_transposed, read_a, scales_a, true);
    auto b_slices =
        slice_direct(kb, batch, type, slices, "_oz_b_slice", k, n, operands.b_transposed, read_b, scales_b, false);
    direct_products(kb, operands, slices, a_slices, scales_a, b_slices, scales_b, operands.b_transposed);
}

} // namespace

namespace detail {

void build_slicing_direct_prepared(
    KernelBuilder& kb,
    ContractionOperands& operands,
    const std::vector<std::string>& b_slices,
    const std::string& scales_b
) {
    auto m = operands.m, k = operands.k;
    auto type = operands.element;
    auto read_a = [&](BlockBuilder& block, auto& entry, auto& row, auto& column) {
        return operands.a(block, entry, row, column);
    };
    auto scales_a = absmax(kb, "_oz_scale_a", read_a, m, k, smallest_normal(type), type, operands.batch);
    auto slices = static_cast<int>(b_slices.size());
    auto a_slices =
        slice_direct(kb, operands.batch, type, slices, "_oz_a_slice", m, k, operands.a_transposed, read_a, scales_a, true);
    direct_products(kb, operands, slices, a_slices, scales_a, b_slices, scales_b, false);
}

} // namespace detail

void OzakiContraction::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto splits = effective_splits();
    auto variant = variant_;
    auto direct = rescaling_ == Rescaling::Direct;
    // Scales of B's columns then reduce along rows, and the slices are row-major, as targets of direct rescaling need.
    auto& node = direct && b_stored_transposed(node_) ? read_b_row_major(builder, node_).contraction : node_;
    detail::expand_contraction(builder, node, [&](KernelBuilder& kb, detail::ContractionOperands& operands) {
        if (direct) {
            build_slicing_direct(kb, operands, splits);
        } else if (variant == OzakiVariant::Slicing) {
            build_slicing(kb, operands, splits);
        } else {
            build_modular(kb, operands, splits);
        }
    });
    analysis_manager.invalidate_all();
}

void OzakiContraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", node_.element_id()}, {"type", "library_node"}}}};
    j["parameters"] = {
        {"variant", variant_ == OzakiVariant::Slicing ? "slicing" : "modular"},
        {"splits", splits_},
        {"rescaling", rescaling_ == Rescaling::Direct ? "direct" : "exponents"},
    };
}

OzakiContraction OzakiContraction::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto parameters = j.contains("parameters") ? j["parameters"] : nlohmann::json::object();
    auto variant = parameters.value("variant", std::string("modular")) == "slicing" ? OzakiVariant::Slicing
                                                                                    : OzakiVariant::Modular;
    auto rescaling = parameters.value("rescaling", std::string("exponents")) == "direct" ? Rescaling::Direct
                                                                                         : Rescaling::Exponents;
    return OzakiContraction(detail::contraction_from_json(builder, j), variant, parameters.value("splits", 0), rescaling);
}

} // namespace fp_math
} // namespace sdfg
