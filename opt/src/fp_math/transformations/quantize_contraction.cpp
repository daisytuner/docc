#include "sdfg/fp_math/transformations/quantize_contraction.h"

#include <nlohmann/json.hpp>

#include "../contractions/contraction_kernels.h"

namespace sdfg {
namespace fp_math {

using data_flow::TaskletCode;
using math::cmath::CMathFunction;
using types::PrimitiveType;

QuantizeContraction::QuantizeContraction(data_flow::LibraryNode& node, int bits) : node_(node), bits_(bits) {
}

std::string QuantizeContraction::name() const {
    return "QuantizeContraction";
}

bool QuantizeContraction::can_be_applied(builder::StructuredSDFGBuilder&, analysis::AnalysisManager&) {
    if (bits_ < 2 || bits_ > 8) {
        return false;
    }
    int64_t q = (int64_t{1} << (bits_ - 1)) - 1;
    return detail::unsupported_contraction(node_, detail::exact_int32_reduction(q)).empty();
}

void QuantizeContraction::apply(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    const double q = static_cast<double>((int64_t{1} << (bits_ - 1)) - 1);
    detail::expand_contraction(builder, node_, [&](KernelBuilder& kb, detail::ContractionOperands& operands) {
        auto m = operands.m, n = operands.n, k = operands.k;
        // Rounding in the contraction's own type is far below the quantization error.
        auto type = operands.element;
        auto tiny = detail::smallest_normal(type);
        auto read_a = [&](BlockBuilder& block, auto& row, auto& column) {
            return operands.a(block, row, column);
        };
        auto read_b_t = [&](BlockBuilder& block, auto& column, auto& row) {
            return operands.b(block, row, column);
        };
        auto scales_a = detail::absmax(kb, "_q_scale_a", read_a, m, k, tiny, type);
        auto scales_b = detail::absmax(kb, "_q_scale_b", read_b_t, n, k, tiny, type);

        // X_q = rint(x * q / s), with |x| <= s, lies in [-q, q].
        auto quantize = [&](BlockBuilder& block, const KernelValue& x, const KernelValue& scale) {
            auto factor = block.op(TaskletCode::fp_div, {block.constant(q, type), scale}, type);
            auto scaled = block.op(TaskletCode::fp_mul, {block.convert(x, type), factor}, type);
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
            auto quantized = kb.buffer(prefix, PrimitiveType::Int8, symbolic::mul(rows, columns));
            auto& outer = kb.map(kb.scope(), "_o", transposed ? columns : rows);
            auto& inner = kb.map(outer.root(), "_e", transposed ? rows : columns);
            auto r = transposed ? inner.indvar() : outer.indvar();
            auto c = transposed ? outer.indvar() : inner.indvar();
            auto body = kb.block(inner.root());
            auto value = quantize(body, read(body, r, c), body.read(scales, {per_row ? r : c}));
            auto index = transposed ? symbolic::add(symbolic::mul(c, rows), r)
                                    : symbolic::add(symbolic::mul(r, columns), c);
            body.store(value, quantized, {index});
            return quantized;
        };
        auto a_q = quantize_operand("_q_a", m, k, operands.a_transposed, read_a, scales_a, true);
        auto read_b = [&](BlockBuilder& block, auto& row, auto& column) {
            return operands.b(block, row, column);
        };
        auto b_q = quantize_operand("_q_b", k, n, operands.b_transposed, read_b, scales_b, false);

        auto products = kb.buffer("_q_p", PrimitiveType::Int32, symbolic::mul(m, n));
        detail::int8_matmul(kb, a_q, b_q, products, m, n, k, operands.a_transposed, operands.b_transposed);

        auto& rows = kb.map(kb.scope(), "_i", m);
        auto& columns = kb.map(rows.root(), "_j", n);
        auto i = rows.indvar(), j = columns.indvar();
        auto body = kb.block(columns.root());
        auto p = body.convert(body.read(products, {symbolic::add(symbolic::mul(i, n), j)}), type);
        auto inverse_q = body.constant(1.0 / q, type);
        auto row_scale = body.op(TaskletCode::fp_mul, {body.read(scales_a, {i}), inverse_q}, type);
        auto column_scale = body.op(TaskletCode::fp_mul, {body.read(scales_b, {j}), inverse_q}, type);
        auto scaled = body.op(TaskletCode::fp_mul, {p, row_scale}, type);
        auto value = body.op(TaskletCode::fp_mul, {scaled, column_scale}, type);
        operands.store(body, value, i, j);
    });
    analysis_manager.invalidate_all();
}

void QuantizeContraction::to_json(nlohmann::json& j) const {
    j["transformation_type"] = this->name();
    j["subgraph"] = {{"0", {{"element_id", node_.element_id()}, {"type", "library_node"}}}};
    j["parameters"] = {{"bits", bits_}};
}

QuantizeContraction QuantizeContraction::from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    int bits = j.contains("parameters") ? j["parameters"].value("bits", 8) : 8;
    return QuantizeContraction(detail::contraction_from_json(builder, j), bits);
}

} // namespace fp_math
} // namespace sdfg
