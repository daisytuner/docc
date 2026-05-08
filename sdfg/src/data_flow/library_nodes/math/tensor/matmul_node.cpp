#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include <cstddef>
#include <string>

#include "sdfg/analysis/scope_analysis.h"
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
    types::PrimitiveType quantization
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_MatMul,
          {},
          {"Y", "A", "B"},
          data_flow::ImplementationType_NONE
      ),
      fixed_quantization_(quantization), layout_a_(layout_a), layout_b_(layout_b) {
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

const TensorLayout& MatMulNode::layout_a() const { return layout_a_; }

const TensorLayout& MatMulNode::layout_b() const { return layout_b_; }

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
    auto k_a = layout_a_.get_dim_innermost(0);
    auto k_b = layout_b_.get_dim_innermost(1);
    if (!symbolic::eq(k_a, k_b)) {
        throw InvalidSDFGException(
            "MatMulNode: K dimension mismatch. A has K=" + k_a->__str__() + ", B has K=" + k_b->__str__()
        );
    }
}

symbolic::SymbolSet MatMulNode::symbols() const {
    symbolic::SymbolSet syms;
    layout_a_.collect_symbols(syms);
    layout_b_.collect_symbols(syms);
    return syms;
}

void MatMulNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_a_.replace_symbols(old_expression, new_expression);
    layout_b_.replace_symbols(old_expression, new_expression);
}

std::unique_ptr<data_flow::DataFlowNode> MatMulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(
        new MatMulNode(element_id, debug_info(), vertex, parent, layout_a_, layout_b_, fixed_quantization_)
    );
}

types::PrimitiveType MatMulNode::fixed_quantization() const { return fixed_quantization_; }

types::PrimitiveType MatMulNode::quantization(const data_flow::DataFlowGraph& data_flow_graph) const {
    if (fixed_quantization_ != QUANTIZATION_MATCH_INPUTS) {
        return fixed_quantization_;
    } else {
        return this->primitive_type(data_flow_graph);
    }
}

std::optional<types::PrimitiveType> MatMulNode::uniform_quantization(const data_flow::DataFlowGraph& data_flow_graph
) const {
    if (fixed_quantization_ != QUANTIZATION_MATCH_INPUTS) {
        auto inferred = this->primitive_type(data_flow_graph);
        if (inferred == fixed_quantization_) {
            return fixed_quantization_;
        } else {
            return std::nullopt;
        }
    } else {
        return this->primitive_type(data_flow_graph);
    }
}

std::string MatMulNode::toStr() const {
    std::stringstream ss;
    ss << "MatMul(";
    ss << types::primitive_type_to_string(fixed_quantization_) << ", ";
    ss << "A: " << layout_a_;
    ss << ", B: " << layout_b_;
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

bool MatMulNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(*this) != 3 || dataflow.out_degree(*this) != 0) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    // Get input and output edges
    auto iedges = dataflow.in_edges_by_connector(*this);
    if (iedges.size() != 3) {
        return false;
    }
    auto* iedge_y = iedges.at(Y_INPUT_IDX);
    auto* iedge_a = iedges.at(A_INPUT_IDX);
    auto* iedge_b = iedges.at(B_INPUT_IDX);

    // Check if legal - access nodes must not have other connections
    auto& input_node_a = static_cast<data_flow::AccessNode&>(iedge_a->src());
    auto& input_node_b = static_cast<data_flow::AccessNode&>(iedge_b->src());
    auto& output_ptr = static_cast<data_flow::AccessNode&>(iedge_y->src());

    if (dataflow.in_degree(input_node_a) != 0 || dataflow.in_degree(input_node_b) != 0 ||
        dataflow.in_degree(output_ptr) != 0) {
        return false;
    }

    // Determine BLAS precision from primitive type
    auto prim_type = this->uniform_quantization(dataflow);
    if (!prim_type) {
        return false;
    }
    blas::BLAS_Precision precision;
    switch (prim_type.value()) {
        case types::PrimitiveType::Half:
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
            return false;
    };

    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

    auto copy_name_a = input_node_a.data();
    auto copy_name_b = input_node_b.data();

    // Check if A and B have basic strides and whether they are transposed in the last dimension
    blas::BLAS_Transpose trans_a, trans_b;
    if (layout_a_.has_linear_accesses_no_padding()) {
        trans_a = blas::BLAS_Transpose::No;
    } else if (layout_a_.has_transposed_strides_no_padding()) {
        trans_a = blas::BLAS_Transpose::Trans;
    } else {
        trans_a = blas::BLAS_Transpose::No;
        throw InvalidSDFGException("A must be in c-order");
    }
    if (layout_b_.has_linear_accesses_no_padding()) {
        trans_b = blas::BLAS_Transpose::No;
    } else if (layout_b_.has_transposed_strides_no_padding()) {
        trans_b = blas::BLAS_Transpose::Trans;
    } else {
        trans_b = blas::BLAS_Transpose::No;
        throw InvalidSDFGException("B must be in c-order");
    }

    // Create maps for batch dimensions and M, N dimensions
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    symbolic::MultiExpression batch_vars;

    // Compute batch dimensions (all except last 2)
    size_t batch_dims_a = layout_a_.dims() - 2;
    size_t batch_dims_b = layout_b_.dims() - 2;
    size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

    // Create maps for batch dimensions (using broadcasting)
    for (size_t i = 0; i < max_batch_dims; ++i) {
        std::string indvar_str = builder.find_new_name("_b");
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = symbolic::zero();
        auto update = symbolic::add(indvar, symbolic::one());

        // Determine the bound for this batch dimension (max of A and B for broadcasting)
        symbolic::Expression bound;
        size_t a_idx = batch_dims_a >= (max_batch_dims - i) ? i - (max_batch_dims - batch_dims_a) : SIZE_MAX;
        size_t b_idx = batch_dims_b >= (max_batch_dims - i) ? i - (max_batch_dims - batch_dims_b) : SIZE_MAX;

        if (a_idx != SIZE_MAX && b_idx != SIZE_MAX) {
            // Both have this dimension - they should be equal or one should be 1 (broadcasting)
            bound = layout_a_.get_dim(a_idx); // Assume they match or broadcasting is handled
        } else if (a_idx != SIZE_MAX) {
            bound = layout_a_.get_dim(a_idx);
        } else {
            bound = layout_b_.get_dim(b_idx);
        }

        auto condition = symbolic::Lt(indvar, bound);
        last_map = &builder.add_map(
            *last_scope,
            indvar,
            condition,
            init,
            update,
            structured_control_flow::ScheduleType_Sequential::create(),
            {},
            block.debug_info()
        );
        last_scope = &last_map->root();
        batch_vars.push_back(indvar);
    }

    auto& ref_block = builder.add_block(*last_scope, {}, block.debug_info());

    auto scalar_type = types::Scalar(prim_type.value());

    // Compute offsets for this batch iteration
    // For A: base_offset_a = offset_a + sum_i(batch_idx_i * batch_stride_a_i)
    symbolic::Expression a_batch_offset = layout_a_.offset();
    for (size_t i = 0; i < batch_dims_a; ++i) {
        size_t batch_idx = max_batch_dims - batch_dims_a + i;
        a_batch_offset = symbolic::add(a_batch_offset, symbolic::mul(batch_vars[batch_idx], layout_a_.get_stride(i)));
    }

    // For B: base_offset_b = offset_b + sum_i(batch_idx_i * batch_stride_b_i)
    symbolic::Expression b_batch_offset = layout_b_.offset();
    for (size_t i = 0; i < batch_dims_b; ++i) {
        size_t batch_idx = max_batch_dims - batch_dims_b + i;
        b_batch_offset = symbolic::add(b_batch_offset, symbolic::mul(batch_vars[batch_idx], layout_b_.get_stride(i)));
    }

    // Compute output batch offset (same as batch_vars pattern for Y)
    symbolic::Expression c_batch_offset = symbolic::integer(0);
    for (size_t i = 0; i < batch_vars.size(); ++i) {
        // Output has shape [batch..., M, N] with row-major strides
        // Stride for batch dim i is: M * N * product of remaining batch dims
        symbolic::Expression c_stride = symbolic::mul(this->m(), this->n());
        for (size_t j = i + 1; j < batch_vars.size(); ++j) {
            // Multiply by subsequent batch dimensions
            if (j < batch_dims_a) {
                c_stride = symbolic::mul(c_stride, layout_a_.get_dim(j));
            } else if (j - batch_dims_a < batch_dims_b) {
                c_stride = symbolic::mul(c_stride, layout_b_.get_dim(j - batch_dims_a));
            }
        }
        c_batch_offset = symbolic::add(c_batch_offset, symbolic::mul(batch_vars[i], c_stride));
    }

    // Create access nodes
    auto& a_access = builder.add_access(ref_block, copy_name_a, debug_info());
    auto& b_access = builder.add_access(ref_block, copy_name_b, debug_info());
    auto& c_access_in = builder.add_access(ref_block, output_ptr.data(), debug_info());

    std::string ref_name_a = builder.find_new_name(copy_name_a + "_ref");
    builder.add_container(ref_name_a, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
    auto& a_access_ref = builder.add_access(ref_block, ref_name_a, debug_info());
    std::string ref_name_b = builder.find_new_name(copy_name_b + "_ref");
    builder.add_container(ref_name_b, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
    auto& b_access_ref = builder.add_access(ref_block, ref_name_b, debug_info());
    std::string ref_name_c = builder.find_new_name(output_ptr.data() + "_ref");
    builder.add_container(ref_name_c, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
    auto& c_access_ref_in = builder.add_access(ref_block, ref_name_c, debug_info());

    builder.add_reference_memlet(
        ref_block, a_access, a_access_ref, {a_batch_offset}, ::sdfg::types::Pointer(scalar_type), debug_info()
    );
    builder.add_reference_memlet(
        ref_block, b_access, b_access_ref, {b_batch_offset}, ::sdfg::types::Pointer(scalar_type), debug_info()
    );
    builder.add_reference_memlet(
        ref_block, c_access_in, c_access_ref_in, {c_batch_offset}, ::sdfg::types::Pointer(scalar_type), debug_info()
    );

    // Create block with GEMM library node
    auto& gemm_block = builder.add_block(*last_scope, {}, block.debug_info());

    // Leading dimensions: stride of the row dimension (second-to-last dim)
    symbolic::Expression lda, ldb;
    if (trans_a == blas::BLAS_Transpose::No) {
        // For row-major A [m * k] -> lda = k
        lda = layout_a_.get_stride_innermost(1);
    } else {
        // For row-major A [m * k] -> lda = m
        lda = layout_a_.get_stride_innermost(0);
    }
    if (trans_b == blas::BLAS_Transpose::No) {
        // For row-major B [k * n] -> ldb = n
        ldb = layout_b_.get_stride_innermost(1);
    } else {
        // For row-major B [k * n] -> ldb = k
        ldb = layout_b_.get_stride_innermost(0);
    }
    // For row-major C [m * n] -> ldc = n
    auto ldc = this->n();

    // Add GEMM node: C = alpha * A * B + beta * C
    // With alpha = 1.0, beta = 0.0: C = A * B
    auto& gemm_node = builder.add_library_node<blas::GEMMNode>(
        gemm_block,
        debug_info(),
        blas::ImplementationType_BLAS,
        precision,
        blas::BLAS_Layout::RowMajor,
        trans_a,
        trans_b,
        this->m(),
        this->n(),
        this->k(),
        lda,
        ldb,
        ldc
    );

    auto& a_access_ref_in_gemm = builder.add_access(gemm_block, ref_name_a, debug_info());
    auto& b_access_ref_in_gemm = builder.add_access(gemm_block, ref_name_b, debug_info());
    auto& c_access_ref_in_gemm = builder.add_access(gemm_block, ref_name_c, debug_info());

    // Create alpha and beta constants
    auto& alpha_const = builder.add_constant(gemm_block, "1.0", scalar_type, debug_info());
    auto& beta_const = builder.add_constant(gemm_block, "0.0", scalar_type, debug_info());

    // Connect memlets with batch offsets
    // Input A with offset
    builder.add_computational_memlet(
        gemm_block, a_access_ref_in_gemm, gemm_node, "__A", {}, ::sdfg::types::Pointer(scalar_type), debug_info()
    );
    // Input B with offset
    builder.add_computational_memlet(
        gemm_block, b_access_ref_in_gemm, gemm_node, "__B", {}, ::sdfg::types::Pointer(scalar_type), debug_info()
    );
    // Input C (for beta * C, but beta=0 so just needs to be connected)
    builder.add_computational_memlet(
        gemm_block, c_access_ref_in_gemm, gemm_node, "__C", {}, ::sdfg::types::Pointer(scalar_type), debug_info()
    );
    // Alpha constant
    builder.add_computational_memlet(gemm_block, alpha_const, gemm_node, "__alpha", {}, scalar_type, debug_info());
    // Beta constant
    builder.add_computational_memlet(gemm_block, beta_const, gemm_node, "__beta", {}, scalar_type, debug_info());

    // Free copies if we made them
    if (copy_name_a != input_node_a.data()) {
        free_after_copy(copy_name_a, builder, new_sequence);
    }
    if (copy_name_b != input_node_b.data()) {
        free_after_copy(copy_name_b, builder, new_sequence);
    }

    // Remove the original nodes
    builder.remove_memlet(block, *iedge_a);
    builder.remove_memlet(block, *iedge_b);
    builder.remove_memlet(block, *iedge_y);
    if (&input_node_a != &input_node_b) {
        builder.remove_node(block, input_node_a);
    }
    builder.remove_node(block, input_node_b);
    builder.remove_node(block, output_ptr);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

nlohmann::json MatMulNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MatMulNode& matmul_node = static_cast<const MatMulNode&>(library_node);
    nlohmann::json j;

    j["code"] = matmul_node.code().value();

    serializer::JSONSerializer serializer;

    matmul_node.layout_a().serialize_to_json(j["layout_a"]);
    matmul_node.layout_b().serialize_to_json(j["layout_b"]);

    j["result_quant"] = matmul_node.fixed_quantization();

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
    types::PrimitiveType quantization = QUANTIZATION_MATCH_INPUTS;

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

    auto result_quant = j.find("result_quant");
    if (result_quant != j.end()) {
        quantization = result_quant->get<types::PrimitiveType>();
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<MatMulNode>(parent, debug_info, layout_a.value(), layout_b.value(), quantization);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
