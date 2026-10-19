#include "sdfg/fp_math/transformations/quantize_contraction.h"

#include <nlohmann/json.hpp>

#include "../contractions/contraction_kernels.h"

namespace sdfg {
namespace fp_math {

using data_flow::TaskletCode;
using math::cmath::CMathFunction;
using types::PrimitiveType;

QuantizeContraction::QuantizeContraction(data_flow::LibraryNode& node, int bits, Rescaling rescaling)
    : node_(node), bits_(bits), rescaling_(rescaling) {
}

std::string QuantizeContraction::name() const {
    return "QuantizeContraction";
}

bool QuantizeContraction::can_be_applied(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&) {
    if (bits_ < 2 || bits_ > 8) {
        return false;
    }
    int64_t q = (int64_t{1} << (bits_ - 1)) - 1;
    return detail::unsupported_contraction(node_, detail::exact_int32_reduction(q), true).empty();
}

void QuantizeContraction::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    const double q = static_cast<double>((int64_t{1} << (bits_ - 1)) - 1);
    detail::expand_contraction(builder, node_, [&](KernelBuilder& kb, detail::ContractionOperands& operands) {
        auto m = operands.m, n = operands.n, k = operands.k;
        // Each matrix of a batch gets scales of its own; every buffer holds one matrix after the other.
        auto& batch = operands.batch;
        // Rounding in the contraction's own type is far below the quantization error.
        auto type = operands.element;
        auto tiny = detail::smallest_normal(type);
        auto read_a = [&](BlockBuilder& block, auto& entry, auto& row, auto& column) {
            return operands.a(block, entry, row, column);
        };
        auto read_b_t = [&](BlockBuilder& block, auto& entry, auto& column, auto& row) {
            return operands.b(block, entry, row, column);
        };
        auto scales_a = detail::absmax(kb, "_q_scale_a", read_a, m, k, tiny, type, batch);
        auto scales_b = detail::absmax(kb, "_q_scale_b", read_b_t, n, k, tiny, type, batch);

        // X_q = rint(x / s * q), with |x| <= s, lies in [-q, q]; dividing first never overflows, unlike q / s for
        // small s.
        auto quantize = [&](BlockBuilder& block, const KernelValue& x, const KernelValue& scale) {
            auto ratio = block.op(TaskletCode::fp_div, {block.convert(x, type), scale}, type);
            auto scaled = block.op(TaskletCode::fp_mul, {ratio, block.constant(q, type)}, type);
            return block.cmath(CMathFunction::rint, {scaled}, type);
        };
        // Each operand is quantized in its storage order, its inner loop running along memory, and the Int8 matmul
        // reads it in that order.
        auto quantize_operand = [&](const std::string& prefix,
                                    const symbolic::Expression& rows,
                                    const symbolic::Expression& columns,
                                    bool transposed,
                                    const detail::MatrixRead& read,
                                    const std::string& scales,
                                    bool per_row) {
            auto size = symbolic::mul(rows, columns);
            auto quantized = kb.buffer(prefix, PrimitiveType::Int8, symbolic::mul(batch.count(), size));
            auto matrices = batch.maps(kb, kb.scope());
            auto& entry = matrices.second;
            auto position = batch.position(entry);
            auto& outer = kb.map(*matrices.first, "_o", transposed ? columns : rows);
            auto& inner = kb.map(outer.root(), "_e", transposed ? rows : columns);
            auto r = transposed ? inner.indvar() : outer.indvar();
            auto c = transposed ? outer.indvar() : inner.indvar();
            auto body = kb.block(inner.root());
            auto scale = symbolic::add(symbolic::mul(position, per_row ? rows : columns), per_row ? r : c);
            auto value = quantize(body, read(body, entry, r, c), body.read(scales, {scale}));
            auto index = transposed ? symbolic::add(symbolic::mul(c, rows), r)
                                    : symbolic::add(symbolic::mul(r, columns), c);
            body.store(value, quantized, {symbolic::add(symbolic::mul(position, size), index)});
            return quantized;
        };
        auto a_q = quantize_operand("_q_a", m, k, operands.a_transposed, read_a, scales_a, true);
        auto read_b = [&](BlockBuilder& block, auto& entry, auto& row, auto& column) {
            return operands.b(block, entry, row, column);
        };
        auto b_q = quantize_operand("_q_b", k, n, operands.b_transposed, read_b, scales_b, false);

        auto products = kb.buffer("_q_p", PrimitiveType::Int32, symbolic::mul(batch.count(), symbolic::mul(m, n)));
        // The nest scaling the products of each matrix: its entry, block, and the row and column it is at.
        struct Rescale {
            symbolic::MultiExpression entry;
            BlockBuilder body;
            symbolic::Expression i, j, p, row_scale, column_scale;
        };
        auto rescale = [&]() {
            auto matrices = batch.maps(kb, kb.scope());
            auto position = batch.position(matrices.second);
            auto& rows = kb.map(*matrices.first, "_i", m);
            auto& columns = kb.map(rows.root(), "_j", n);
            auto i = rows.indvar(), j = columns.indvar();
            return Rescale{
                matrices.second,
                kb.block(columns.root()),
                i,
                j,
                symbolic::add(symbolic::mul(position, symbolic::mul(m, n)), symbolic::add(symbolic::mul(i, n), j)),
                symbolic::add(symbolic::mul(position, m), i),
                symbolic::add(symbolic::mul(position, n), j),
            };
        };
        if (rescaling_ == Rescaling::Direct) {
            detail::int8_matmul(kb, a_q, b_q, products, m, n, k, operands.a_transposed, operands.b_transposed, batch);
            auto nest = rescale();
            auto& body = nest.body;
            auto p = body.convert(body.read(products, {nest.p}), type);
            auto inverse_q = body.constant(1.0 / q, type);
            auto row_scale = body.op(TaskletCode::fp_mul, {body.read(scales_a, {nest.row_scale}), inverse_q}, type);
            auto column_scale =
                body.op(TaskletCode::fp_mul, {body.read(scales_b, {nest.column_scale}), inverse_q}, type);
            auto scaled = body.op(TaskletCode::fp_mul, {p, row_scale}, type);
            operands.store(body, body.op(TaskletCode::fp_mul, {scaled, column_scale}, type), nest.entry, nest.i, nest.j);
            return;
        }

        auto [exponents_a, inverses_a] = detail::power_of_two_scales(kb, "_q_a", scales_a, m, type, batch);
        auto [exponents_b, inverses_b] = detail::power_of_two_scales(kb, "_q_b", scales_b, n, type, batch);

        detail::int8_matmul(kb, a_q, b_q, products, m, n, k, operands.a_transposed, operands.b_transposed, batch);

        auto nest = rescale();
        auto& body = nest.body;
        // Y = P * (s_A / q) * (s_B / q) with s = m 2^e and m in [1/2, 1): the mantissas keep the product within
        // K q^2, and the exponents apply in one step, so that no intermediate overflows where Y does not.
        auto p = body.convert(body.read(products, {nest.p}), type);
        auto inverse_q = body.constant(1.0 / q, type);
        auto mantissa = [&](const std::string& scales, const std::string& inverses, const symbolic::Expression& at) {
            auto normalized = body.op(TaskletCode::fp_mul, {body.read(scales, {at}), body.read(inverses, {at})}, type);
            return body.op(TaskletCode::fp_mul, {normalized, inverse_q}, type);
        };
        auto scaled = body.op(TaskletCode::fp_mul, {p, mantissa(scales_a, inverses_a, nest.row_scale)}, type);
        auto value = body.op(TaskletCode::fp_mul, {scaled, mantissa(scales_b, inverses_b, nest.column_scale)}, type);
        auto exponent = body
                            .op(TaskletCode::fp_add,
                                {body.read(exponents_a, {nest.row_scale}), body.read(exponents_b, {nest.column_scale})},
                                type);
        operands.store(body, detail::scale_by_power_of_two(body, value, exponent, type), nest.entry, nest.i, nest.j);
    });
    analysis_manager.invalidate_all();
}

void QuantizeContraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", node_.element_id()}, {"type", "library_node"}}}};
    j["parameters"] = {{"bits", bits_}, {"rescaling", rescaling_ == Rescaling::Direct ? "direct" : "exponents"}};
}

QuantizeContraction QuantizeContraction::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    int bits = 8;
    auto rescaling = Rescaling::Exponents;
    if (j.contains("parameters")) {
        bits = j["parameters"].value("bits", 8);
        rescaling = j["parameters"].value("rescaling", "exponents") == "direct" ? Rescaling::Direct
                                                                                : Rescaling::Exponents;
    }
    return QuantizeContraction(detail::contraction_from_json(builder, j), bits, rescaling);
}

} // namespace fp_math
} // namespace sdfg
