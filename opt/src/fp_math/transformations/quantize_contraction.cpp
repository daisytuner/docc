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
        auto tiny = detail::smallest_normal(operands.element);
        auto read_a = [&](BlockBuilder& block, auto& row, auto& column) {
            return operands.a(block, row, column);
        };
        auto read_b_t = [&](BlockBuilder& block, auto& column, auto& row) {
            return operands.b(block, row, column);
        };
        auto scales_a = detail::absmax(kb, "_q_scale_a", read_a, m, k, tiny);
        auto scales_b = detail::absmax(kb, "_q_scale_b", read_b_t, n, k, tiny);

        // X_q = rint(x * q / s), with |x| <= s, lies in [-q, q].
        auto quantize = [&](BlockBuilder& block, const KernelValue& x, const KernelValue& scale) {
            auto factor =
                block.op(TaskletCode::fp_div, {block.constant(q, PrimitiveType::Double), scale}, PrimitiveType::Double);
            auto scaled =
                block.op(TaskletCode::fp_mul, {block.convert(x, PrimitiveType::Double), factor}, PrimitiveType::Double);
            return block.cmath(CMathFunction::rint, {scaled}, PrimitiveType::Double);
        };
        auto a_q = kb.buffer("_q_a", PrimitiveType::Int8, symbolic::mul(m, k));
        {
            auto& rows = kb.map(kb.scope(), "_i", m);
            auto& columns = kb.map(rows.root(), "_k", k);
            auto i = rows.indvar(), c = columns.indvar();
            auto body = kb.block(columns.root());
            auto value = quantize(body, operands.a(body, i, c), body.read(scales_a, {i}));
            body.store(value, a_q, {symbolic::add(symbolic::mul(i, k), c)});
        }
        auto b_q = kb.buffer("_q_b", PrimitiveType::Int8, symbolic::mul(k, n));
        {
            auto& rows = kb.map(kb.scope(), "_k", k);
            auto& columns = kb.map(rows.root(), "_j", n);
            auto r = rows.indvar(), j = columns.indvar();
            auto body = kb.block(columns.root());
            auto value = quantize(body, operands.b(body, r, j), body.read(scales_b, {j}));
            body.store(value, b_q, {symbolic::add(symbolic::mul(r, n), j)});
        }

        auto products = kb.buffer("_q_p", PrimitiveType::Int32, symbolic::mul(m, n));
        detail::int8_matmul(kb, a_q, b_q, products, m, n, k);

        auto& rows = kb.map(kb.scope(), "_i", m);
        auto& columns = kb.map(rows.root(), "_j", n);
        auto i = rows.indvar(), j = columns.indvar();
        auto body = kb.block(columns.root());
        auto p = body.convert(body.read(products, {symbolic::add(symbolic::mul(i, n), j)}), PrimitiveType::Double);
        auto inverse_q = body.constant(1.0 / q, PrimitiveType::Double);
        auto row_scale = body.op(TaskletCode::fp_mul, {body.read(scales_a, {i}), inverse_q}, PrimitiveType::Double);
        auto column_scale = body.op(TaskletCode::fp_mul, {body.read(scales_b, {j}), inverse_q}, PrimitiveType::Double);
        auto scaled = body.op(TaskletCode::fp_mul, {p, row_scale}, PrimitiveType::Double);
        auto value = body.op(TaskletCode::fp_mul, {scaled, column_scale}, PrimitiveType::Double);
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
