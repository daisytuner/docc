#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include <cstddef>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/control_flow_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

bool MatMulNode::has_basic_strides(symbolic::MultiExpression shape, symbolic::MultiExpression strides) {
    auto basic_strides = types::Tensor::strides_from_shape(shape);
    if (basic_strides.size() != strides.size()) {
        return false;
    }
    for (size_t i = 0; i < strides.size(); i++) {
        if (!symbolic::eq(basic_strides[i], strides[i])) {
            return false;
        }
    }
    return true;
}

bool MatMulNode::has_transposed_strides(symbolic::MultiExpression shape, symbolic::MultiExpression strides) {
    if (shape.size() < 2) {
        return false;
    }
    symbolic::MultiExpression new_shape;
    new_shape.reserve(shape.size());
    for (size_t i = 0; i < shape.size() - 2; i++) {
        new_shape.push_back(shape[i]);
    }
    new_shape.push_back(shape[shape.size() - 1]);
    new_shape.push_back(shape[shape.size() - 2]);
    symbolic::MultiExpression transposed_strides(strides);
    transposed_strides[strides.size() - 2] = strides[strides.size() - 1];
    transposed_strides[strides.size() - 1] = strides[strides.size() - 2];
    return MatMulNode::has_basic_strides(new_shape, transposed_strides);
}

MatMulNode::MatMulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& layout_a,
    const TensorLayout& layout_b,
    QuantizationType quantization,
    const TensorLayout* layout_y,
    const data_flow::ImplementationType& impl_type
)
    : MatMulNode(
          element_id,
          debug_info,
          vertex,
          parent,
          layout_a,
          layout_b,
          ContractionTypes::uniform(quantization),
          layout_y,
          impl_type
      ) {
}

MatMulNode::MatMulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& layout_a,
    const TensorLayout& layout_b,
    const ContractionTypes& types,
    const TensorLayout* layout_y,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_MatMul, {}, {"Y", "A", "B"}, impl_type),
      types_(types), layout_a_(layout_a), layout_b_(layout_b),
      layout_y_(layout_y ? *layout_y : get_linear_result_layout(layout_a, layout_b)) {
    if (layout_a.dims() < 2) {
        throw std::invalid_argument("MatMulNode: Input A must have at least 2 dimensions");
    }
    if (layout_b.dims() < 2) {
        throw std::invalid_argument("MatMulNode: Input B must have at least 2 dimensions");
    }
}

symbolic::Expression MatMulNode::m() const {
    // M is the second-to-last dimension of A
    return layout_a_.get_dim_innermost(1);
}

symbolic::Expression MatMulNode::n() const {
    // N is the last dimension of B
    return layout_b_.get_dim_innermost(0);
}

symbolic::Expression MatMulNode::k() const {
    // K is the last dimension of A (and second-to-last of B)
    return layout_a_.get_dim_innermost(0);
}

const TensorLayout& MatMulNode::layout_a() const {
    return layout_a_;
}

const TensorLayout& MatMulNode::layout_b() const {
    return layout_b_;
}

const TensorLayout& MatMulNode::layout_y() const {
    return layout_y_;
}

TensorLayout MatMulNode::get_linear_result_layout(const TensorLayout& layout_a, const TensorLayout& layout_b) {
    // The trailing two dimensions follow the standard matmul rule:
    //   [..., M, K] x [..., K, N] -> [..., M, N]
    auto m = layout_a.get_dim_innermost(1);
    auto n = layout_b.get_dim_innermost(0);

    // Batch dimensions are everything but the trailing two matrix dimensions. A and B may
    // have different batch ranks and/or size-1 batch dims that broadcast, so the result's
    // batch shape is the element-wise broadcast of A's and B's right-aligned batch dims.
    const size_t batch_a = layout_a.dims() - 2;
    const size_t batch_b = layout_b.dims() - 2;
    const size_t out_batch = std::max(batch_a, batch_b);

    symbolic::MultiExpression shape;
    shape.reserve(out_batch + 2);
    for (size_t p = 0; p < out_batch; ++p) {
        // Walk batch dims outermost-to-innermost. Innermost indices are 0 = N, 1 = M,
        // 2.. = batch (inner-to-outer), so this output batch dim sits at innermost index:
        const size_t inner = out_batch + 1 - p;
        const bool has_a = (inner - 2) < batch_a;
        const bool has_b = (inner - 2) < batch_b;

        if (has_a && has_b) {
            auto a_dim = layout_a.get_dim_innermost(inner);
            auto b_dim = layout_b.get_dim_innermost(inner);
            // Broadcast: a size-1 dimension takes the other operand's extent.
            shape.push_back(symbolic::eq(a_dim, symbolic::one()) ? b_dim : a_dim);
        } else if (has_a) {
            shape.push_back(layout_a.get_dim_innermost(inner));
        } else {
            shape.push_back(layout_b.get_dim_innermost(inner));
        }
    }
    shape.push_back(m);
    shape.push_back(n);

    return TensorLayout(shape); // always in row-major, linear strides, no padding
}

void MatMulNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    // Check that we have exactly 2 inputs and 1 output
    if (graph.in_degree(*this) != 3) {
        throw InvalidSDFGException("MatMulNode: Expected exactly 3 inputs (Y, A, B)");
    }
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("MatMulNode: Expected no outputs");
    }

    // Validate K dimension matches between A and B
    auto m_a = layout_a_.get_dim_innermost(1);
    auto k_a = layout_a_.get_dim_innermost(0);
    auto n_b = layout_b_.get_dim_innermost(0);
    auto k_b = layout_b_.get_dim_innermost(1);
    if (!symbolic::eq(k_a, k_b)) {
        throw InvalidSDFGException(
            "MatMulNode: K dimension mismatch. A has K=" + k_a->__str__() + ", B has K=" + k_b->__str__()
        );
    }
    auto m_y = layout_y_.get_dim_innermost(1);
    auto n_y = layout_y_.get_dim_innermost(0);

    auto m_match = symbolic::eq(m_a, m_y);
    auto n_match = symbolic::eq(n_b, n_y);
    if (!m_match || !n_match) {
        throw InvalidSDFGException(
            "MatMulNode: Output dimensions mismatch. Expected (" + m_a->__str__() + ", " + n_b->__str__() + "), got (" +
            m_y->__str__() + ", " + n_y->__str__() + ")"
        );
    }
}

namespace {

types::PrimitiveType edge_type(const data_flow::DataFlowGraph& graph, const MatMulNode& node, size_t input_idx) {
    auto* edge = graph.in_edge_for_connector(node, node.input(input_idx));
    return edge ? edge->base_type().primitive_type() : types::PrimitiveType::Void;
}

void check_type(types::PrimitiveType expected, types::PrimitiveType actual, const std::string& operand) {
    if (expected != types::PrimitiveType::Void && actual != types::PrimitiveType::Void && expected != actual) {
        throw InvalidSDFGException(
            "MatMulNode: " + operand + " memlet type mismatch. Expected " +
            std::string(types::primitive_type_to_string(expected)) + ", got " +
            std::string(types::primitive_type_to_string(actual))
        );
    }
}

} // namespace

void MatMulNode::verify_data_types(const data_flow::DataFlowGraph& graph) const {
    auto a = edge_type(graph, *this, A_INPUT_IDX);
    auto b = edge_type(graph, *this, B_INPUT_IDX);
    auto y = edge_type(graph, *this, Y_INPUT_IDX);

    // A and B always share a type, the configured one if any.
    auto input = types_.input != types::PrimitiveType::Void ? types_.input : a;
    check_type(input, a, "Input A");
    check_type(input, b, "Input B");
    check_type(types_.output, y, "Output");
}

symbolic::SymbolSet MatMulNode::symbols() const {
    symbolic::SymbolSet syms;
    layout_a_.collect_symbols(syms);
    layout_b_.collect_symbols(syms);
    return syms;
}

data_flow::PointerAccessType MatMulNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(layout_y_.total_elements(), true, layout_y_);
    } else if (input_idx == A_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(layout_a_.total_elements(), true, layout_a_);
    } else if (input_idx == B_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(layout_b_.total_elements(), true, layout_b_);
    }
    return TensorNode::pointer_access_type(input_idx);
}

bool MatMulNode::relocalize_operand(int input_idx, const TensorLayout& packed) {
    if (!can_relocalize_operand(input_idx, packed)) {
        return false;
    }
    if (input_idx == Y_INPUT_IDX) {
        layout_y_ = packed;
    } else if (input_idx == A_INPUT_IDX) {
        layout_a_ = packed;
    } else {
        layout_b_ = packed;
    }
    return true;
}

bool MatMulNode::can_relocalize_operand(int input_idx, const TensorLayout&) const {
    return input_idx == Y_INPUT_IDX || input_idx == A_INPUT_IDX || input_idx == B_INPUT_IDX;
}

void MatMulNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_a_.replace_symbols(old_expression, new_expression);
    layout_b_.replace_symbols(old_expression, new_expression);
}

void MatMulNode::replace(const symbolic::ExpressionMapping& replacements) {
    layout_a_.replace_symbols(replacements);
    layout_b_.replace_symbols(replacements);
}

std::unique_ptr<data_flow::DataFlowNode> MatMulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new MatMulNode(
        element_id, debug_info(), vertex, parent, layout_a_, layout_b_, types_, &layout_y_, implementation_type_
    ));
}

const ContractionTypes& MatMulNode::types() const {
    return types_;
}

void MatMulNode::set_types(const ContractionTypes& types) {
    types_ = types;
}

types::PrimitiveType MatMulNode::input_type() const {
    if (types_.input != types::PrimitiveType::Void) {
        return types_.input;
    }
    auto a = edge_type(get_parent(), *this, A_INPUT_IDX);
    auto b = edge_type(get_parent(), *this, B_INPUT_IDX);
    if (a == types::PrimitiveType::Void) {
        return b;
    }
    if (b != types::PrimitiveType::Void && a != b) {
        return types::PrimitiveType::Void;
    }
    return a;
}

types::PrimitiveType MatMulNode::output_type() const {
    if (types_.output != types::PrimitiveType::Void) {
        return types_.output;
    }
    return edge_type(get_parent(), *this, Y_INPUT_IDX);
}

types::PrimitiveType MatMulNode::accumulator_type() const {
    if (types_.accumulator != types::PrimitiveType::Void) {
        return types_.accumulator;
    }
    return output_type();
}

std::optional<types::PrimitiveType> MatMulNode::uniform_quantization(const data_flow::DataFlowGraph&) const {
    // Unknown (Void) types adopt the known ones; Void if none is known.
    auto uniform = types::PrimitiveType::Void;
    for (auto type : {input_type(), accumulator_type(), output_type()}) {
        if (type == types::PrimitiveType::Void) {
            continue;
        }
        if (uniform != types::PrimitiveType::Void && uniform != type) {
            return std::nullopt;
        }
        uniform = type;
    }
    return uniform;
}

types::PrimitiveType MatMulNode::quantization(const data_flow::DataFlowGraph& dataflow) const {
    return uniform_quantization(dataflow).value_or(types::PrimitiveType::Void);
}

types::PrimitiveType MatMulNode::input_quantization() const {
    return input_type();
}

types::PrimitiveType MatMulNode::output_quantization() const {
    return output_type();
}

types::PrimitiveType MatMulNode::fixed_quantization() const {
    if (types_.input == types_.output &&
        (types_.accumulator == types::PrimitiveType::Void || types_.accumulator == types_.output)) {
        return types_.output;
    }
    return types::PrimitiveType::Void;
}

void MatMulNode::set_fixed_quantization(const QuantizationType quant) {
    types_ = ContractionTypes::uniform(quant);
}

std::string MatMulNode::toStr() const {
    std::stringstream ss;
    ss << "MatMul(";
    if (types_.input == types_.output && types_.accumulator == types::PrimitiveType::Void) {
        ss << types::primitive_type_to_string(types_.output) << ", ";
    } else {
        ss << types::primitive_type_to_string(types_.input) << " -> "
           << types::primitive_type_to_string(types_.accumulator) << " -> "
           << types::primitive_type_to_string(types_.output) << ", ";
    }
    ss << "A: " << layout_a_;
    ss << ", B: " << layout_b_;
    if (!layout_y_.has_linear_accesses()) {
        ss << ", Y: " << layout_y_;
    }
    if (implementation_type_ != data_flow::ImplementationType_NONE) {
        ss << ", impl: " << implementation_type_.value();
    }
    ss << ")";
    return ss.str();
}

symbolic::Expression MatMulNode::flop() const {
    auto res_elems = symbolic::mul(this->m(), this->n());
    auto k = this->k();

    auto mm_mul_ops = symbolic::mul(res_elems, k);
    auto mm_sum_ops = symbolic::mul(res_elems, symbolic::sub(k, symbolic::one()));

    auto mul_ops = mm_mul_ops;
    auto add_ops = mm_sum_ops;
    auto per_mat = symbolic::add(mul_ops, add_ops);
    int a_dims = layout_a_.dims();
    int b_dims = layout_b_.dims();
    if (a_dims > 2 || b_dims > 2) {
        std::vector<symbolic::Expression> factors{per_mat};
        auto max_dims = std::max(a_dims, b_dims);
        for (int i = 2; i < max_dims; ++i) {
            symbolic::Expression dim_a, dim_b;
            if (i < a_dims) {
                dim_a = layout_a_.get_dim_innermost(i);
            }
            if (i < b_dims) {
                dim_b = layout_b_.get_dim_innermost(i);
            }
            if (dim_a.is_null() & !dim_b.is_null()) {
                factors.push_back(dim_b);
            } else if (!dim_a.is_null() & dim_b.is_null()) {
                factors.push_back(dim_a);
            } else if (!dim_a.is_null() & !dim_b.is_null()) {
                if (!symbolic::eq(dim_a, dim_b)) {
                    throw InvalidSDFGException(
                        "Batch dimension " + std::to_string(i) + " mismatch between A and B. A has " +
                        dim_a->__str__() + ", B has " + dim_b->__str__()
                    );
                } else {
                    factors.push_back(dim_a);
                }
            } else {
                return SymEngine::null;
            }
        }
        return SymEngine::mul(factors);
    } else {
        return per_mat;
    }
}

void free_after_copy(
    const std::string& copy_name, builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& parent
) {
    auto& block = builder.add_block(parent, {}, DebugInfo());
    auto& access_in = builder.add_access(block, copy_name);
    auto& free_node = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
    builder.add_computational_memlet(
        block, access_in, free_node, "_ptr", {}, types::Pointer(types::Scalar(types::PrimitiveType::Void))
    );
}

passes::LibNodeExpander::ExpandOutcome MatMulNode::expand(passes::LibNodeExpander::ExpandContext& context, Block& block) {
    MatMulNaiveExpander exp;
    return exp.handle_expand(context, block, *this);
}

nlohmann::json MatMulNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MatMulNode& matmul_node = static_cast<const MatMulNode&>(library_node);
    nlohmann::json j;

    j["code"] = matmul_node.code().value();

    serializer::JSONSerializer serializer;

    matmul_node.layout_a().serialize_to_json(j["layout_a"]);
    matmul_node.layout_b().serialize_to_json(j["layout_b"]);
    matmul_node.layout_y().serialize_to_json(j["layout_y"]);

    // Older readers only know a single type, Void if the types differ.
    auto& types = matmul_node.types();
    bool uniform = types.input == types.output &&
                   (types.accumulator == types::PrimitiveType::Void || types.accumulator == types.output);
    j["result_quant"] = uniform ? types.output : types::PrimitiveType::Void;
    types.serialize_to_json(j);

    return j;
}

data_flow::LibraryNode& MatMulNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    std::optional<TensorLayout> layout_a;
    std::optional<TensorLayout> layout_b;

    auto layout_a_it = j.find("layout_a");
    if (layout_a_it != j.end()) {
        layout_a = TensorLayout::deserialize_from_json(*layout_a_it);
        layout_b = TensorLayout::deserialize_from_json(j.at("layout_b"));

    } else {
        assert(j.contains("shape_a"));
        assert(j.contains("shape_b"));

        symbolic::MultiExpression shape_a;
        for (const auto& dim : j["shape_a"]) {
            shape_a.push_back(symbolic::parse(dim.get<std::string>()));
        }

        symbolic::MultiExpression shape_b;
        for (const auto& dim : j["shape_b"]) {
            shape_b.push_back(symbolic::parse(dim.get<std::string>()));
        }

        symbolic::MultiExpression strides_a;
        if (j.contains("strides_a")) {
            for (const auto& stride : j["strides_a"]) {
                strides_a.push_back(symbolic::parse(stride.get<std::string>()));
            }
        }

        symbolic::MultiExpression strides_b;
        if (j.contains("strides_b")) {
            for (const auto& stride : j["strides_b"]) {
                strides_b.push_back(symbolic::parse(stride.get<std::string>()));
            }
        }

        symbolic::Expression offset_a = symbolic::integer(0);
        if (j.contains("offset_a")) {
            offset_a = symbolic::parse(j["offset_a"].get<std::string>());
        }

        symbolic::Expression offset_b = symbolic::integer(0);
        if (j.contains("offset_b")) {
            offset_b = symbolic::parse(j["offset_b"].get<std::string>());
        }

        layout_a = TensorLayout(shape_a, strides_a, offset_a);
        layout_b = TensorLayout(shape_b, strides_b, offset_b);
    }

    auto layout_y_it = j.find("layout_y");
    std::optional<TensorLayout> layout_y;
    if (layout_y_it != j.end()) {
        layout_y = TensorLayout::deserialize_from_json(*layout_y_it);
    }

    auto types = ContractionTypes::deserialize_from_json(j, "result_quant");

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto* layout_y_ptr = layout_y.has_value() ? &*layout_y : nullptr;

    return builder
        .add_library_node<MatMulNode>(parent, debug_info, layout_a.value(), layout_b.value(), types, layout_y_ptr);
}

using Dir = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome MatMulNaiveExpander::handle_expand(
    passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, MatMulNode& node
) const {
    auto& dataflow = node.get_parent();

    // Expect exactly the three tensor inputs (Y, A, B) and no outputs.
    if (dataflow.in_degree(node) != 3 || dataflow.out_degree(node) != 0) {
        return context.unable();
    }

    auto* iedge_y = dataflow.in_edge_for_connector(node, node.input(MatMulNode::Y_INPUT_IDX));
    auto* iedge_a = dataflow.in_edge_for_connector(node, node.input(MatMulNode::A_INPUT_IDX));
    auto* iedge_b = dataflow.in_edge_for_connector(node, node.input(MatMulNode::B_INPUT_IDX));
    if (!iedge_y || !iedge_a || !iedge_b) {
        return context.unable();
    }

    const auto& a_shape = node.layout_a().shape();
    const auto& b_shape = node.layout_b().shape();
    const auto& y_shape = node.layout_y().shape();

    if (a_shape.size() < 2 || b_shape.size() < 2 || y_shape.size() < 2) {
        return context.unable();
    }

    // Batch dimensions are everything but the trailing two matrix dimensions.
    const size_t batch_a = a_shape.size() - 2;
    const size_t batch_b = b_shape.size() - 2;
    const size_t out_batch = y_shape.size() - 2;

    // The output must have enough rank to broadcast both inputs' batch dims.
    if (out_batch < batch_a || out_batch < batch_b) {
        return context.unable();
    }

    const auto m = node.m();
    const auto n = node.n();
    const auto k = node.k();

    // Accumulate in place in the output, in its element type; the C/C++ backend inserts the implicit conversions
    // for the (possibly differing) input element types. A wider accumulator would need a separate buffer.
    const auto y_prim = node.output_type();
    if (node.accumulator_type() != y_prim) {
        return context.unable();
    }
    const bool fp = types::is_floating_point(y_prim);
    types::Scalar compute_scalar(y_prim);

    auto standalone =
        context.replacement_requires_access_nodes({Dir::IndirectReadWrite, Dir::IndirectRead, Dir::IndirectRead});
    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();
    auto& new_sequence = standalone->replace_with_sequence();

    // Create a loop (Map for parallel dims, For for the sequential K reduction).
    auto add_loop = [&](structured_control_flow::Sequence& scope,
                        const symbolic::Expression& bound,
                        const std::string& prefix,
                        bool as_map) -> structured_control_flow::StructuredLoop& {
        std::string iv = builder.find_new_name(prefix);
        builder.add_container(iv, types::Scalar(types::get_primitive_type_to_hold_upper_bound(bound)));
        auto sym = symbolic::symbol(iv);
        auto cond = symbolic::Lt(sym, bound);
        auto update = symbolic::add(sym, symbolic::one());
        if (as_map) {
            return builder.add_map(
                scope,
                sym,
                cond,
                symbolic::zero(),
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                block.debug_info()
            );
        }
        return builder.add_for(scope, sym, cond, symbolic::zero(), update, block.debug_info());
    };

    // Build the per-dimension logical index into an operand, aligning its batch
    // dims to the right of the output's batch dims and honoring size-1 broadcasts.
    auto build_operand_idx = [&](const symbolic::MultiExpression& batch_vars,
                                 const symbolic::MultiExpression& operand_shape,
                                 size_t batch_dims,
                                 const symbolic::Expression& row,
                                 const symbolic::Expression& col) {
        symbolic::MultiExpression idx;
        idx.reserve(batch_dims + 2);
        for (size_t j = 0; j < batch_dims; ++j) {
            const size_t out_pos = out_batch - batch_dims + j;
            if (symbolic::eq(operand_shape[j], symbolic::one())) {
                idx.push_back(symbolic::zero()); // broadcast
            } else {
                idx.push_back(batch_vars[out_pos]);
            }
        }
        idx.push_back(row);
        idx.push_back(col);
        return idx;
    };

    auto build_y_idx = [&](const symbolic::MultiExpression& batch_vars,
                           const symbolic::Expression& row,
                           const symbolic::Expression& col) {
        symbolic::MultiExpression idx(batch_vars);
        idx.push_back(row);
        idx.push_back(col);
        return idx;
    };

    // Address every operand through the node's layout as a pointer, whether its memlet is typed as a pointer or a
    // (deprecated) tensor.
    auto element_pointer = [](const data_flow::Memlet& edge) {
        return types::Pointer(types::Scalar(edge.base_type().primitive_type()));
    };
    const auto a_pointer = element_pointer(*iedge_a);
    const auto b_pointer = element_pointer(*iedge_b);
    const auto y_pointer = element_pointer(*iedge_y);
    auto linear = [](const TensorLayout& layout, const symbolic::MultiExpression& idx) {
        return data_flow::Subset{layout.resolve_element(idx)};
    };

    // ---- Initialization nest: for (batch, i, j) Y[...] = 0 ----
    {
        structured_control_flow::Sequence* scope = &new_sequence;
        symbolic::MultiExpression batch_vars;
        for (size_t i = 0; i < out_batch; ++i) {
            auto& loop = add_loop(*scope, y_shape[i], "_b", true);
            scope = &loop.root();
            batch_vars.push_back(loop.indvar());
        }
        auto& i_loop = add_loop(*scope, m, "_i", true);
        auto& j_loop = add_loop(i_loop.root(), n, "_j", true);
        auto& init_blk = builder.add_block(j_loop.root(), block.debug_info());

        auto y_idx = linear(node.layout_y(), build_y_idx(batch_vars, i_loop.indvar(), j_loop.indvar()));

        auto& zero_node = builder.add_constant(init_blk, fp ? "0.0" : "0", compute_scalar, block.debug_info());
        auto& assign_t =
            builder.add_tasklet(init_blk, data_flow::TaskletCode::assign, "_out", {"_in"}, block.debug_info());
        auto& y_write = standalone->add_indirect_write_access(init_blk, MatMulNode::Y_INPUT_IDX);
        builder.add_computational_memlet(init_blk, zero_node, assign_t, "_in", {}, compute_scalar, block.debug_info());
        builder.add_computational_memlet(init_blk, assign_t, "_out", y_write, y_idx, y_pointer, iedge_y->debug_info());
    }

    // ---- Accumulation nest: for (batch, i, j) for k: Y[...] += A[...] * B[...] ----
    {
        structured_control_flow::Sequence* scope = &new_sequence;
        symbolic::MultiExpression batch_vars;
        for (size_t i = 0; i < out_batch; ++i) {
            auto& loop = add_loop(*scope, y_shape[i], "_b", true);
            scope = &loop.root();
            batch_vars.push_back(loop.indvar());
        }
        auto& i_loop = add_loop(*scope, m, "_i", true);
        auto& j_loop = add_loop(i_loop.root(), n, "_j", true);
        auto& k_loop = add_loop(j_loop.root(), k, "_k", false);
        auto& code_block = builder.add_block(k_loop.root(), block.debug_info());

        const auto i_sym = i_loop.indvar();
        const auto j_sym = j_loop.indvar();
        const auto k_sym = k_loop.indvar();

        auto a_idx = linear(node.layout_a(), build_operand_idx(batch_vars, a_shape, batch_a, i_sym, k_sym));
        auto b_idx = linear(node.layout_b(), build_operand_idx(batch_vars, b_shape, batch_b, k_sym, j_sym));
        auto y_idx = linear(node.layout_y(), build_y_idx(batch_vars, i_sym, j_sym));

        auto& a_node = standalone->add_indirect_read_access(code_block, MatMulNode::A_INPUT_IDX);
        auto& b_node = standalone->add_indirect_read_access(code_block, MatMulNode::B_INPUT_IDX);
        auto& y_in = standalone->add_indirect_read_access(code_block, MatMulNode::Y_INPUT_IDX);
        auto& y_out = standalone->add_indirect_write_access(code_block, MatMulNode::Y_INPUT_IDX);

        // An operand of another type than the accumulator is converted first: the backend multiplies in the
        // operands' types, so e.g. Float inputs overflow even though they accumulate in Double.
        struct Factor {
            data_flow::AccessNode* node;
            data_flow::Subset subset;
            const types::IType* type;
        };
        auto factor = [&](data_flow::AccessNode& node,
                          const data_flow::Subset& subset,
                          const types::Pointer& pointer,
                          const data_flow::Memlet& edge) -> Factor {
            if (pointer.pointee_type().primitive_type() == y_prim) {
                return {&node, subset, &pointer};
            }
            auto& convert =
                builder.add_tasklet(code_block, data_flow::TaskletCode::assign, "_out", {"_in"}, block.debug_info());
            builder.add_computational_memlet(code_block, node, convert, "_in", subset, pointer, edge.debug_info());
            std::string converted = builder.find_new_name("_operand");
            builder.add_container(converted, compute_scalar);
            auto& converted_node = builder.add_access(code_block, converted, block.debug_info());
            builder.add_computational_memlet(
                code_block, convert, "_out", converted_node, {}, compute_scalar, block.debug_info()
            );
            return {&converted_node, {}, &compute_scalar};
        };
        auto a = factor(a_node, a_idx, a_pointer, *iedge_a);
        auto b = factor(b_node, b_idx, b_pointer, *iedge_b);

        if (fp) {
            // Single fused multiply-add: Y[...] = A[...] * B[...] + Y[...].
            auto& fma =
                builder.add_tasklet(code_block, data_flow::fp_fma, "_out", {"_in1", "_in2", "_in3"}, block.debug_info());
            builder.add_computational_memlet(code_block, *a.node, fma, "_in1", a.subset, *a.type, iedge_a->debug_info());
            builder.add_computational_memlet(code_block, *b.node, fma, "_in2", b.subset, *b.type, iedge_b->debug_info());
            builder.add_computational_memlet(code_block, y_in, fma, "_in3", y_idx, y_pointer, iedge_y->debug_info());
            builder.add_computational_memlet(code_block, fma, "_out", y_out, y_idx, y_pointer, iedge_y->debug_info());
        } else {
            // Integer path (no integer FMA): prod = A * B; Y = prod + Y.
            auto& mul_t = builder.add_tasklet(
                code_block, data_flow::TaskletCode::int_mul, "_out", {"_in1", "_in2"}, block.debug_info()
            );
            builder
                .add_computational_memlet(code_block, *a.node, mul_t, "_in1", a.subset, *a.type, iedge_a->debug_info());
            builder
                .add_computational_memlet(code_block, *b.node, mul_t, "_in2", b.subset, *b.type, iedge_b->debug_info());

            std::string prod = builder.find_new_name("_prod");
            builder.add_container(prod, compute_scalar);
            auto& prod_node = builder.add_access(code_block, prod, block.debug_info());
            builder
                .add_computational_memlet(code_block, mul_t, "_out", prod_node, {}, compute_scalar, block.debug_info());

            auto& add_t = builder.add_tasklet(
                code_block, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"}, block.debug_info()
            );
            builder
                .add_computational_memlet(code_block, prod_node, add_t, "_in1", {}, compute_scalar, block.debug_info());
            builder.add_computational_memlet(code_block, y_in, add_t, "_in2", y_idx, y_pointer, iedge_y->debug_info());
            builder.add_computational_memlet(code_block, add_t, "_out", y_out, y_idx, y_pointer, iedge_y->debug_info());
        }
    }

    return standalone->successfully_expanded();
}

passes::LibNodeExpander::ExpandOutcome MatMulBlasExpander::handle_expand(
    passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, MatMulNode& node
) const {
    auto& dataflow = node.get_parent();

    if (dataflow.in_degree(node) != 3 || dataflow.out_degree(node) != 0) {
        return context.unable();
    }

    auto& parent = static_cast<structured_control_flow::Sequence&>(*block.get_parent());
    int index = parent.index(block);

    // Determine BLAS precision from primitive type
    auto prim_type = node.uniform_quantization(dataflow);
    if (!prim_type) {
        return context.unable();
    }
    blas::BLAS_Precision precision;
    switch (prim_type.value()) {
        case types::PrimitiveType::Half:
            if (!this->supports_half_) {
                DEBUG_PRINTLN("MatMulBlasExpander: No support for BLAS call with half precision");
                return context.unable();
            }
            precision = blas::BLAS_Precision::h;
            break;
        case types::PrimitiveType::Float:
            precision = blas::BLAS_Precision::s;
            break;
        case types::PrimitiveType::Double:
            precision = blas::BLAS_Precision::d;
            break;
        default:
            // GEMM only supports floating point types, fall back to naive expansion
            return context.unable();
    };

    if (!node.layout_y().is_last_dims_expressible_as_row_major()) {
        DEBUG_PRINTLN(
            "MatMulBlasExpander: Matmul #" << node.element_id()
                                           << " uses a custom result layout, that does not fulfill GEMM requirements"
        );
        return context.unable();
    }

    // Check if A and B have basic strides and whether they are transposed in the last dimension
    blas::BLAS_Transpose trans_a, trans_b;
    if (node.layout_a().has_linear_accesses_no_padding()) {
        trans_a = blas::BLAS_Transpose::No;
    } else if (node.layout_a().has_transposed_strides_no_padding()) {
        trans_a = blas::BLAS_Transpose::Trans;
    } else {
        DEBUG_PRINTLN("MatMulBlasExpander: A must be in c-order but got: " + node.layout_a().toStr());
        return context.unable();
    }
    if (node.layout_b().has_linear_accesses_no_padding()) {
        trans_b = blas::BLAS_Transpose::No;
    } else if (node.layout_b().has_transposed_strides_no_padding()) {
        trans_b = blas::BLAS_Transpose::Trans;
    } else {
        DEBUG_PRINTLN("MatMulBlasExpander: B must be in c-order but got: " + node.layout_b().toStr());
        return context.unable();
    }

    auto standalone =
        context.replacement_requires_access_nodes({Dir::IndirectReadWrite, Dir::IndirectRead, Dir::IndirectRead});

    if (standalone) {
        auto& builder = standalone->builder();

        // Add new graph after the current block
        auto& new_sequence = standalone->replace_with_sequence();

        // Create maps for batch dimensions and M, N dimensions
        structured_control_flow::Sequence* last_scope = &new_sequence;
        structured_control_flow::Map* last_map = nullptr;
        symbolic::MultiExpression batch_vars;

        // Compute batch dimensions (all except last 2)
        size_t batch_dims_a = node.layout_a().dims() - 2;
        size_t batch_dims_b = node.layout_b().dims() - 2;
        size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

        // Create maps for batch dimensions (using broadcasting)
        for (size_t i = 0; i < max_batch_dims; ++i) {
            // Determine the bound for this batch dimension (max of A and B for broadcasting)
            symbolic::Expression bound;
            size_t a_idx = batch_dims_a >= (max_batch_dims - i) ? i - (max_batch_dims - batch_dims_a) : SIZE_MAX;
            size_t b_idx = batch_dims_b >= (max_batch_dims - i) ? i - (max_batch_dims - batch_dims_b) : SIZE_MAX;

            if (a_idx != SIZE_MAX && b_idx != SIZE_MAX) {
                // Both have this dimension - they should be equal or one should be 1 (broadcasting)
                bound = node.layout_a().get_dim(a_idx); // Assume they match or broadcasting is handled
            } else if (a_idx != SIZE_MAX) {
                bound = node.layout_a().get_dim(a_idx);
            } else {
                bound = node.layout_b().get_dim(b_idx);
            }

            std::string indvar_str = builder.find_new_name("_b");
            builder.add_container(indvar_str, types::Scalar(TensorLayout::get_tensor_indvar_type_for_shape({bound})));

            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());

            auto condition = symbolic::Lt(indvar, bound);
            last_map = &builder.add_map(
                *last_scope,
                indvar,
                condition,
                init,
                update,
                structured_control_flow::ScheduleType_Sequential::create(),
                block.debug_info()
            );
            last_scope = &last_map->root();
            batch_vars.push_back(indvar);
        }

        auto& ref_block = builder.add_block(*last_scope, {}, block.debug_info());

        auto scalar_type = types::Scalar(prim_type.value());

        // Compute offsets for this batch iteration
        // For A: base_offset_a = offset_a + sum_i(batch_idx_i * batch_stride_a_i)
        symbolic::Expression a_batch_offset = node.layout_a().offset();
        for (size_t i = 0; i < batch_dims_a; ++i) {
            size_t batch_idx = max_batch_dims - batch_dims_a + i;
            a_batch_offset =
                symbolic::add(a_batch_offset, symbolic::mul(batch_vars[batch_idx], node.layout_a().get_stride(i)));
        }

        // For B: base_offset_b = offset_b + sum_i(batch_idx_i * batch_stride_b_i)
        symbolic::Expression b_batch_offset = node.layout_b().offset();
        for (size_t i = 0; i < batch_dims_b; ++i) {
            size_t batch_idx = max_batch_dims - batch_dims_b + i;
            b_batch_offset =
                symbolic::add(b_batch_offset, symbolic::mul(batch_vars[batch_idx], node.layout_b().get_stride(i)));
        }

        // Compute output batch offset (same as batch_vars pattern for Y)
        symbolic::Expression c_batch_offset = symbolic::integer(0);
        for (size_t i = 0; i < batch_vars.size(); ++i) {
            // Output has shape [batch..., M, N] with row-major strides
            // Stride for batch dim i is: M * N * product of remaining batch dims
            symbolic::Expression c_stride = symbolic::mul(node.m(), node.n());
            for (size_t j = i + 1; j < batch_vars.size(); ++j) {
                // Multiply by subsequent batch dimensions
                if (j < batch_dims_a) {
                    c_stride = symbolic::mul(c_stride, node.layout_a().get_dim(j));
                } else if (j - batch_dims_a < batch_dims_b) {
                    c_stride = symbolic::mul(c_stride, node.layout_b().get_dim(j - batch_dims_a));
                }
            }
            c_batch_offset = symbolic::add(c_batch_offset, symbolic::mul(batch_vars[i], c_stride));
        }

        // Create input access nodes
        auto& a_access = standalone->add_scalar_input_access(ref_block, MatMulNode::A_INPUT_IDX);
        auto& b_access = standalone->add_scalar_input_access(ref_block, MatMulNode::B_INPUT_IDX);
        auto& c_access_in = standalone->add_indirect_read_access(ref_block, MatMulNode::Y_INPUT_IDX);

        auto copy_name_a = a_access.data();
        auto copy_name_b = b_access.data();
        auto output_name = c_access_in.data();

        std::string ref_name_a = builder.find_new_name(copy_name_a + "_ref");
        builder.add_container(ref_name_a, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
        auto& a_access_ref = builder.add_access(ref_block, ref_name_a, node.debug_info());
        std::string ref_name_b = builder.find_new_name(copy_name_b + "_ref");
        builder.add_container(ref_name_b, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
        auto& b_access_ref = builder.add_access(ref_block, ref_name_b, node.debug_info());
        std::string ref_name_c = builder.find_new_name(output_name + "_ref");
        builder.add_container(ref_name_c, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
        auto& c_access_ref_in = builder.add_access(ref_block, ref_name_c, node.debug_info());

        builder.add_reference_memlet(
            ref_block, a_access, a_access_ref, {a_batch_offset}, ::sdfg::types::Pointer(scalar_type), node.debug_info()
        );
        builder.add_reference_memlet(
            ref_block, b_access, b_access_ref, {b_batch_offset}, ::sdfg::types::Pointer(scalar_type), node.debug_info()
        );
        builder.add_reference_memlet(
            ref_block,
            c_access_in,
            c_access_ref_in,
            {c_batch_offset},
            ::sdfg::types::Pointer(scalar_type),
            node.debug_info()
        );

        // Create block with GEMM library node
        auto& gemm_block = builder.add_block(*last_scope, {}, node.debug_info());

        // Leading dimensions: stride of the row dimension (second-to-last dim)
        symbolic::Expression lda, ldb;
        if (trans_a == blas::BLAS_Transpose::No) {
            // For row-major A [m * k] -> lda = k
            lda = node.layout_a().get_stride_innermost(1);
        } else {
            // For row-major A [m * k] -> lda = m
            lda = node.layout_a().get_stride_innermost(0);
        }
        if (trans_b == blas::BLAS_Transpose::No) {
            // For row-major B [k * n] -> ldb = n
            ldb = node.layout_b().get_stride_innermost(1);
        } else {
            // For row-major B [k * n] -> ldb = k
            ldb = node.layout_b().get_stride_innermost(0);
        }
        // For row-major C [m * n] -> ldc = n
        auto ldc = node.n();

        // Add GEMM node: C = alpha * A * B + beta * C
        // With alpha = 1.0, beta = 0.0: C = A * B
        auto& gemm_node = builder.add_library_node<blas::GEMMNode>(
            gemm_block,
            node.debug_info(),
            blas::ImplementationType_BLAS,
            precision,
            blas::BLAS_Layout::RowMajor,
            trans_a,
            trans_b,
            node.m(),
            node.n(),
            node.k(),
            lda,
            ldb,
            ldc
        );

        auto& a_access_ref_in_gemm = builder.add_access(gemm_block, ref_name_a, node.debug_info());
        auto& b_access_ref_in_gemm = builder.add_access(gemm_block, ref_name_b, node.debug_info());
        auto& c_access_ref_in_gemm = builder.add_access(gemm_block, ref_name_c, node.debug_info());

        // Create alpha and beta constants
        auto& alpha_const = builder.add_constant(gemm_block, "1.0", scalar_type, node.debug_info());
        auto& beta_const = builder.add_constant(gemm_block, "0.0", scalar_type, node.debug_info());

        // Connect memlets with batch offsets
        // Input A with offset
        builder.add_computational_memlet(
            gemm_block,
            a_access_ref_in_gemm,
            gemm_node,
            "__A",
            {},
            ::sdfg::types::Pointer(scalar_type),
            node.debug_info()
        );
        // Input B with offset
        builder.add_computational_memlet(
            gemm_block,
            b_access_ref_in_gemm,
            gemm_node,
            "__B",
            {},
            ::sdfg::types::Pointer(scalar_type),
            node.debug_info()
        );
        // Input C (for beta * C, but beta=0 so just needs to be connected)
        builder.add_computational_memlet(
            gemm_block,
            c_access_ref_in_gemm,
            gemm_node,
            "__C",
            {},
            ::sdfg::types::Pointer(scalar_type),
            node.debug_info()
        );
        // Alpha constant
        builder
            .add_computational_memlet(gemm_block, alpha_const, gemm_node, "__alpha", {}, scalar_type, node.debug_info());
        // Beta constant
        builder
            .add_computational_memlet(gemm_block, beta_const, gemm_node, "__beta", {}, scalar_type, node.debug_info());

        return standalone->successfully_expanded();
    } else {
        // lib node was not "standalone" in its block (all inputs and outputs come from access nodes solely used with
        // this libnode) or could not be transformed into a form that can be used as if
        return context.unable();
    }
}


} // namespace tensor
} // namespace math
} // namespace sdfg
