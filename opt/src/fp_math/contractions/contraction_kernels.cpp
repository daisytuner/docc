#include "contraction_kernels.h"

#include <climits>
#include <cmath>
#include <limits>

#include <nlohmann/json.hpp>

#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/transformations/transformation.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace fp_math {
namespace detail {

using data_flow::TaskletCode;
using math::blas::BLAS_Layout;
using math::blas::BLAS_Transpose;
using math::blas::GEMMNode;
using math::cmath::CMathFunction;
using math::tensor::MatMulNode;
using types::PrimitiveType;

int64_t exact_int32_reduction(int64_t magnitude) {
    return static_cast<int64_t>(INT32_MAX) / (magnitude * magnitude);
}

bool is_contraction(const data_flow::LibraryNode& node) {
    return dynamic_cast<const MatMulNode*>(&node) || dynamic_cast<const GEMMNode*>(&node);
}

symbolic::Expression contraction_k(const data_flow::LibraryNode& node) {
    if (auto* matmul = dynamic_cast<const MatMulNode*>(&node)) {
        return matmul->k();
    }
    if (auto* gemm = dynamic_cast<const GEMMNode*>(&node)) {
        return gemm->k();
    }
    return SymEngine::null;
}

types::PrimitiveType contraction_type(const data_flow::LibraryNode& node) {
    if (auto* matmul = dynamic_cast<const MatMulNode*>(&node)) {
        return matmul->output_type();
    }
    if (auto* gemm = dynamic_cast<const GEMMNode*>(&node)) {
        return gemm->scalar_primitive();
    }
    return PrimitiveType::Void;
}

namespace {

std::string unsupported_matmul(const MatMulNode& node) {
    if (node.get_parent().in_degree(node) != 3 || node.get_parent().out_degree(node) != 0) {
        return "the matmul is not connected to Y, A and B";
    }
    if (node.layout_a().dims() != 2 || node.layout_b().dims() != 2 || node.layout_y().dims() != 2) {
        return "batched matmuls are not supported";
    }
    if (node.input_type() != node.output_type() || node.accumulator_type() != node.output_type()) {
        return "inputs, accumulator and output must share their type";
    }
    return "";
}

std::string unsupported_gemm(const GEMMNode& node) {
    if (node.get_parent().in_degree(node) != 5 || node.get_parent().out_degree(node) != 0) {
        return "the GEMM is not connected to A, B, C, alpha and beta";
    }
    if (node.layout() != BLAS_Layout::RowMajor) {
        return "only row-major GEMMs are supported";
    }
    if (node.trans_a() == BLAS_Transpose::ConjTrans || node.trans_b() == BLAS_Transpose::ConjTrans) {
        return "conjugate transposes are not supported";
    }
    // BLAS ignores C if beta is 0 and A @ B if alpha is 0, even NaNs or overflowing products. Only constants tell which
    // operands the emulation may read.
    auto& dataflow = node.get_parent();
    auto* alpha = dataflow.in_edge_for_connector(node, node.input(GEMMNode::ALPHA_INPUT_IDX));
    auto* beta = dataflow.in_edge_for_connector(node, node.input(GEMMNode::BETA_INPUT_IDX));
    if (!dynamic_cast<const data_flow::ConstantNode*>(&alpha->src()) ||
        !dynamic_cast<const data_flow::ConstantNode*>(&beta->src())) {
        return "alpha and beta must be constants";
    }
    if (alpha->is_src_constant(0.0)) {
        return "alpha = 0 leaves no product to compute";
    }
    return "";
}

} // namespace

std::string unsupported_contraction(const data_flow::LibraryNode& node, int64_t max_k) {
    if (!dynamic_cast<const structured_control_flow::Block*>(node.get_parent().get_parent())) {
        return "the contraction is not in a block";
    }
    std::string why;
    if (auto* matmul = dynamic_cast<const MatMulNode*>(&node)) {
        why = unsupported_matmul(*matmul);
    } else if (auto* gemm = dynamic_cast<const GEMMNode*>(&node)) {
        why = unsupported_gemm(*gemm);
    } else {
        return "only MatMul and GEMM nodes are supported";
    }
    if (!why.empty()) {
        return why;
    }
    auto type = contraction_type(node);
    if (type != PrimitiveType::Float && type != PrimitiveType::Double) {
        return "only Float and Double contractions are supported";
    }
    if (!SymEngine::is_a<SymEngine::Integer>(*contraction_k(node))) {
        return "the reduction K must be a constant";
    }
    auto k = constant_k(node);
    if (k < 1 || k > max_k) {
        return "the reduction K = " + std::to_string(k) + " exceeds " + std::to_string(max_k) +
               ", beyond which Int32 accumulation is not exact";
    }
    return "";
}

int64_t constant_k(const data_flow::LibraryNode& node) {
    return SymEngine::down_cast<const SymEngine::Integer&>(*contraction_k(node)).as_int();
}

double smallest_normal(PrimitiveType type) {
    return type == PrimitiveType::Float ? std::numeric_limits<float>::min() : std::numeric_limits<double>::min();
}

namespace {

/// Addresses the operands through the node's layouts, which hold their shapes, strides and offsets.
class MatMulOperands : public ContractionOperands {
    passes::LibNodeExpander::AccessNodeExpand& standalone_;
    math::tensor::TensorLayout layout_a_, layout_b_, layout_y_;
    types::Pointer pointer_;

public:
    MatMulOperands(passes::LibNodeExpander::AccessNodeExpand& standalone, const MatMulNode& node)
        : standalone_(standalone), layout_a_(node.layout_a()), layout_b_(node.layout_b()), layout_y_(node.layout_y()),
          pointer_(types::Scalar(node.output_type())) {
        m = node.m();
        n = node.n();
        k = node.k();
        element = node.output_type();
        using math::tensor::TensorLayout;
        a_transposed = layout_a_.is_2d_col_or_row_major() == TensorLayout::LAYOUT_COL_MAJOR;
        b_transposed = layout_b_.is_2d_col_or_row_major() == TensorLayout::LAYOUT_COL_MAJOR;
    }

    static std::vector<passes::LibNodeExpander::InputUse> uses() {
        using Use = passes::LibNodeExpander::InputUse;
        return {Use::IndirectWrite, Use::IndirectRead, Use::IndirectRead};
    }

    KernelValue a(BlockBuilder& block, const symbolic::Expression& row, const symbolic::Expression& column) override {
        auto& access = standalone_.add_indirect_read_access(block.block(), MatMulNode::A_INPUT_IDX);
        return block.read(access, {layout_a_.resolve_element({row, column})}, pointer_);
    }

    KernelValue b(BlockBuilder& block, const symbolic::Expression& row, const symbolic::Expression& column) override {
        auto& access = standalone_.add_indirect_read_access(block.block(), MatMulNode::B_INPUT_IDX);
        return block.read(access, {layout_b_.resolve_element({row, column})}, pointer_);
    }

    void store(
        BlockBuilder& block,
        const KernelValue& value,
        const symbolic::Expression& row,
        const symbolic::Expression& column
    ) override {
        auto& access = standalone_.add_indirect_write_access(block.block(), MatMulNode::Y_INPUT_IDX);
        block.store(block.convert(value, element), access, {layout_y_.resolve_element({row, column})}, pointer_);
    }
};

class GemmOperands : public ContractionOperands {
    passes::LibNodeExpander::AccessNodeExpand& standalone_;
    const GEMMNode& node_;
    std::unique_ptr<types::IType> a_type_, b_type_, c_type_;
    bool alpha_one_, beta_zero_;

public:
    GemmOperands(passes::LibNodeExpander::AccessNodeExpand& standalone, const GEMMNode& node)
        : standalone_(standalone), node_(node) {
        m = node.m();
        n = node.n();
        k = node.k();
        element = node.scalar_primitive();
        a_transposed = node.trans_a() == BLAS_Transpose::Trans;
        b_transposed = node.trans_b() == BLAS_Transpose::Trans;
        auto& dataflow = node.get_parent();
        auto edge = [&](int idx) {
            return dataflow.in_edge_for_connector(node, node.input(idx));
        };
        a_type_ = edge(GEMMNode::A_INPUT_IDX)->base_type().clone();
        b_type_ = edge(GEMMNode::B_INPUT_IDX)->base_type().clone();
        c_type_ = edge(GEMMNode::C_INPUT_IDX)->base_type().clone();
        alpha_one_ = edge(GEMMNode::ALPHA_INPUT_IDX)->is_src_constant(1.0);
        beta_zero_ = edge(GEMMNode::BETA_INPUT_IDX)->is_src_constant(0.0);
    }

    static std::vector<passes::LibNodeExpander::InputUse> uses(const GEMMNode& node) {
        using Use = passes::LibNodeExpander::InputUse;
        auto& dataflow = node.get_parent();
        // With beta = 0, C is only written: BLAS ignores its old values, even NaNs.
        bool beta_zero =
            dataflow.in_edge_for_connector(node, node.input(GEMMNode::BETA_INPUT_IDX))->is_src_constant(0.0);
        return {
            Use::IndirectRead,
            Use::IndirectRead,
            beta_zero ? Use::IndirectWrite : Use::IndirectReadWrite,
            Use::Scalar,
            Use::Scalar
        };
    }

    KernelValue a(BlockBuilder& block, const symbolic::Expression& row, const symbolic::Expression& column) override {
        auto index = node_.trans_a() == BLAS_Transpose::No ? symbolic::add(symbolic::mul(node_.lda(), row), column)
                                                           : symbolic::add(symbolic::mul(node_.lda(), column), row);
        auto& access = standalone_.add_indirect_read_access(block.block(), GEMMNode::A_INPUT_IDX);
        return block.read(access, {index}, *a_type_);
    }

    KernelValue b(BlockBuilder& block, const symbolic::Expression& row, const symbolic::Expression& column) override {
        auto index = node_.trans_b() == BLAS_Transpose::No ? symbolic::add(symbolic::mul(node_.ldb(), row), column)
                                                           : symbolic::add(symbolic::mul(node_.ldb(), column), row);
        auto& access = standalone_.add_indirect_read_access(block.block(), GEMMNode::B_INPUT_IDX);
        return block.read(access, {index}, *b_type_);
    }

    void store(
        BlockBuilder& block,
        const KernelValue& value,
        const symbolic::Expression& row,
        const symbolic::Expression& column
    ) override {
        auto index = symbolic::add(symbolic::mul(node_.ldc(), row), column);
        auto type = value.base_type->primitive_type();
        auto result = value;
        if (!alpha_one_) {
            auto alpha = block.read(
                standalone_.add_scalar_input_access(block.block(), GEMMNode::ALPHA_INPUT_IDX),
                {},
                types::Scalar(element)
            );
            result = block.op(TaskletCode::fp_mul, {block.convert(alpha, type), result}, type);
        }
        if (!beta_zero_) {
            auto beta = block.read(
                standalone_.add_scalar_input_access(block.block(), GEMMNode::BETA_INPUT_IDX), {}, types::Scalar(element)
            );
            auto c =
                block
                    .read(standalone_.add_indirect_read_access(block.block(), GEMMNode::C_INPUT_IDX), {index}, *c_type_);
            auto scaled = block.op(TaskletCode::fp_mul, {block.convert(beta, type), block.convert(c, type)}, type);
            result = block.op(TaskletCode::fp_add, {result, scaled}, type);
        }
        auto& access = standalone_.add_indirect_write_access(block.block(), GEMMNode::C_INPUT_IDX);
        block.store(block.convert(result, element), access, {index}, *c_type_);
    }
};

} // namespace

std::string absmax(
    KernelBuilder& kb,
    const std::string& prefix,
    const MatrixRead& read,
    const symbolic::Expression& outer,
    const symbolic::Expression& inner,
    double tiny,
    PrimitiveType type
) {
    auto result = kb.buffer(prefix, type, outer);
    auto accumulator = kb.scalar(prefix + "_acc", type);

    auto& rows = kb.map(kb.scope(), "_o", outer);
    {
        // Starting from the smallest normal keeps all-zero rows from dividing by zero later.
        auto init = kb.block(rows.root());
        init.store(init.constant(tiny, type), accumulator);
    }
    auto& elements = kb.reduce(rows.root(), "_e", inner, structured_control_flow::ReductionOperation::Max, accumulator);
    {
        auto body = kb.block(elements.root());
        auto x = body.convert(read(body, rows.indvar(), elements.indvar()), type);
        auto magnitude = body.cmath(CMathFunction::fabs, {x}, type);
        auto max = body.cmath(CMathFunction::fmax, {magnitude, body.read(accumulator)}, type);
        body.store(max, accumulator);
    }
    {
        auto store = kb.block(rows.root());
        store.store(store.read(accumulator), result, {rows.indvar()});
    }
    return result;
}

std::pair<std::string, std::string> power_of_two_scales(
    KernelBuilder& kb,
    const std::string& prefix,
    const std::string& magnitudes,
    const symbolic::Expression& count,
    PrimitiveType type
) {
    auto exponents = kb.buffer(prefix + "_exp", type, count);
    auto inverses = kb.buffer(prefix + "_inv", type, count);
    auto& loop = kb.map(kb.scope(), "_o", count);
    auto body = kb.block(loop.root());
    auto log = body.cmath(CMathFunction::log2, {body.read(magnitudes, {loop.indvar()})}, type);
    auto floor = body.cmath(CMathFunction::floor, {log}, type);
    auto exponent = body.op(TaskletCode::fp_add, {floor, body.constant(1.0, type)}, type);
    auto negated = body.op(TaskletCode::fp_neg, {exponent}, type);
    body.store(exponent, exponents, {loop.indvar()});
    // 2^-e stays finite: magnitudes are at least the smallest normal value, so e > the smallest normal exponent.
    body.store(body.cmath(CMathFunction::exp2, {negated}, type), inverses, {loop.indvar()});
    return {exponents, inverses};
}

KernelValue
scale_by_power_of_two(BlockBuilder& block, const KernelValue& value, const KernelValue& exponent, PrimitiveType type) {
    // value * 2^e in two halves: each power stays representable for any e a representable result can have, and an
    // intermediate lies between value and the result.
    auto half = block.cmath(
        CMathFunction::floor, {block.op(TaskletCode::fp_mul, {exponent, block.constant(0.5, type)}, type)}, type
    );
    auto rest = block.op(TaskletCode::fp_sub, {exponent, half}, type);
    auto first = block.op(TaskletCode::fp_mul, {value, block.cmath(CMathFunction::exp2, {half}, type)}, type);
    return block.op(TaskletCode::fp_mul, {first, block.cmath(CMathFunction::exp2, {rest}, type)}, type);
}

KernelValue symmetric_mod(BlockBuilder& block, const KernelValue& value, int64_t modulus, PrimitiveType type) {
    // ((x % m) + m + h) % m - h lies in [-h, h] for h = (m - 1) / 2 and is congruent to x.
    int64_t half = (modulus - 1) / 2;
    auto remainder = block.op(TaskletCode::int_srem, {value, block.constant(modulus, type)}, type);
    auto shifted = block.op(TaskletCode::int_add, {remainder, block.constant(modulus + half, type)}, type);
    auto wrapped = block.op(TaskletCode::int_srem, {shifted, block.constant(modulus, type)}, type);
    return block.op(TaskletCode::int_sub, {wrapped, block.constant(half, type)}, type);
}

void int8_matmul(
    KernelBuilder& kb,
    const std::string& a,
    const std::string& b,
    const std::string& p,
    const symbolic::Expression& m,
    const symbolic::Expression& n,
    const symbolic::Expression& k,
    bool a_transposed,
    bool b_transposed
) {
    auto& builder = kb.builder();
    auto& debug_info = kb.debug_info();
    auto& block = builder.add_block(kb.scope(), {}, debug_info);
    auto& a_access = builder.add_access(block, a, debug_info);
    auto& b_access = builder.add_access(block, b, debug_info);
    auto& p_access = builder.add_access(block, p, debug_info);
    using math::tensor::TensorLayout;
    auto one = symbolic::one();
    auto& matmul = builder.add_library_node<MatMulNode>(
        block,
        debug_info,
        a_transposed ? TensorLayout({m, k}, {one, m}) : TensorLayout({m, k}),
        b_transposed ? TensorLayout({k, n}, {one, k}) : TensorLayout({k, n}),
        math::tensor::ContractionTypes{PrimitiveType::Int8, PrimitiveType::Int32, PrimitiveType::Int32}
    );
    types::Pointer int8_pointer{types::Scalar(PrimitiveType::Int8)};
    types::Pointer int32_pointer{types::Scalar(PrimitiveType::Int32)};
    builder.add_computational_memlet(block, a_access, matmul, "A", {}, int8_pointer, debug_info);
    builder.add_computational_memlet(block, b_access, matmul, "B", {}, int8_pointer, debug_info);
    builder.add_computational_memlet(block, p_access, matmul, "Y", {}, int32_pointer, debug_info);
}

data_flow::LibraryNode& contraction_from_json(builder::StructuredSDFGBuilder& builder, const nlohmann::json& j) {
    auto id = j.at("subgraph").at("0").at("element_id").get<size_t>();
    auto* node = dynamic_cast<data_flow::LibraryNode*>(builder.find_element_by_id(id));
    if (!node || !is_contraction(*node)) {
        throw transformations::InvalidTransformationDescriptionException(
            "Element with ID " + std::to_string(id) + " is not a MatMul or GEMM node"
        );
    }
    return *node;
}

namespace {

class CallbackExpander : public passes::LibNodeExpander {
    const std::function<void(KernelBuilder&, ContractionOperands&)>& build_;

public:
    explicit CallbackExpander(const std::function<void(KernelBuilder&, ContractionOperands&)>& build) : build_(build) {
    }

    ExpandOutcome handle_expand(
        ExpandContext& context, structured_control_flow::Block& block, data_flow::LibraryNode& node
    ) const override {
        auto* matmul = dynamic_cast<MatMulNode*>(&node);
        auto* gemm = dynamic_cast<GEMMNode*>(&node);
        auto standalone =
            context.replacement_requires_access_nodes(matmul ? MatMulOperands::uses() : GemmOperands::uses(*gemm));
        if (!standalone) {
            return context.unable();
        }
        std::unique_ptr<ContractionOperands> operands;
        if (matmul) {
            operands = std::make_unique<MatMulOperands>(*standalone, *matmul);
        } else {
            operands = std::make_unique<GemmOperands>(*standalone, *gemm);
        }
        auto& sequence = standalone->replace_with_sequence();
        KernelBuilder kb(standalone->builder(), sequence, node.debug_info());
        build_(kb, *operands);
        kb.finish();
        return standalone->successfully_expanded();
    }
};

} // namespace

void expand_contraction(
    builder::StructuredSDFGBuilder& builder,
    data_flow::LibraryNode& node,
    const std::function<void(KernelBuilder&, ContractionOperands&)>& build
) {
    auto& block = static_cast<structured_control_flow::Block&>(*node.get_parent().get_parent());
    CallbackExpander expander(build);
    auto outcome = passes::expansion::expand_single_node(builder, block, node, expander);
    if (!outcome.expanded) {
        throw transformations::InvalidTransformationException(
            "the contraction's operands could not be isolated from the rest of its block"
        );
    }
}

} // namespace detail
} // namespace fp_math
} // namespace sdfg
