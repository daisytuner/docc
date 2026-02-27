#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include <string>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/element.h"
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

MatMulNode::MatMulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const symbolic::MultiExpression& shape_a,
    const symbolic::MultiExpression& shape_b,
    const symbolic::MultiExpression& strides_a,
    const symbolic::MultiExpression& strides_b,
    symbolic::Expression offset_a,
    symbolic::Expression offset_b
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_MatMul,
          {"Y"},
          {"A", "B"},
          data_flow::ImplementationType_NONE
      ),
      shape_a_(shape_a), shape_b_(shape_b), strides_a_(strides_a), strides_b_(strides_b), offset_a_(offset_a),
      offset_b_(offset_b) {
    if (shape_a_.size() < 2) {
        throw std::invalid_argument("MatMulNode: Input A must have at least 2 dimensions");
    }
    if (shape_b_.size() < 2) {
        throw std::invalid_argument("MatMulNode: Input B must have at least 2 dimensions");
    }
    // Compute default row-major strides if not provided
    if (strides_a_.empty()) {
        strides_a_.resize(shape_a_.size());
        strides_a_[shape_a_.size() - 1] = symbolic::integer(1);
        for (int i = static_cast<int>(shape_a_.size()) - 2; i >= 0; --i) {
            strides_a_[i] = symbolic::mul(strides_a_[i + 1], shape_a_[i + 1]);
        }
    }
    if (strides_b_.empty()) {
        strides_b_.resize(shape_b_.size());
        strides_b_[shape_b_.size() - 1] = symbolic::integer(1);
        for (int i = static_cast<int>(shape_b_.size()) - 2; i >= 0; --i) {
            strides_b_[i] = symbolic::mul(strides_b_[i + 1], shape_b_[i + 1]);
        }
    }
}

symbolic::Expression MatMulNode::m() const {
    // M is the second-to-last dimension of A
    return shape_a_[shape_a_.size() - 2];
}

symbolic::Expression MatMulNode::n() const {
    // N is the last dimension of B
    return shape_b_[shape_b_.size() - 1];
}

symbolic::Expression MatMulNode::k() const {
    // K is the last dimension of A (and second-to-last of B)
    return shape_a_[shape_a_.size() - 1];
}

void MatMulNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    // Check that we have exactly 2 inputs and 1 output
    if (graph.in_degree(*this) != 2) {
        throw InvalidSDFGException("MatMulNode: Expected exactly 2 inputs (A and B)");
    }
    if (graph.out_degree(*this) != 1) {
        throw InvalidSDFGException("MatMulNode: Expected exactly 1 output (Y)");
    }

    // Validate K dimension matches between A and B
    auto k_a = shape_a_[shape_a_.size() - 1];
    auto k_b = shape_b_[shape_b_.size() - 2];
    if (!symbolic::eq(k_a, k_b)) {
        throw InvalidSDFGException(
            "MatMulNode: K dimension mismatch. A has K=" + k_a->__str__() + ", B has K=" + k_b->__str__()
        );
    }
}

symbolic::SymbolSet MatMulNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : shape_a_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (const auto& dim : shape_b_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (const auto& stride : strides_a_) {
        for (auto& atom : symbolic::atoms(stride)) {
            syms.insert(atom);
        }
    }
    for (const auto& stride : strides_b_) {
        for (auto& atom : symbolic::atoms(stride)) {
            syms.insert(atom);
        }
    }
    for (auto& atom : symbolic::atoms(offset_a_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(offset_b_)) {
        syms.insert(atom);
    }
    return syms;
}

void MatMulNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : shape_a_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : shape_b_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& stride : strides_a_) {
        stride = symbolic::subs(stride, old_expression, new_expression);
    }
    for (auto& stride : strides_b_) {
        stride = symbolic::subs(stride, old_expression, new_expression);
    }
    offset_a_ = symbolic::subs(offset_a_, old_expression, new_expression);
    offset_b_ = symbolic::subs(offset_b_, old_expression, new_expression);
}

std::unique_ptr<data_flow::DataFlowNode> MatMulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new MatMulNode(
        element_id, debug_info(), vertex, parent, shape_a_, shape_b_, strides_a_, strides_b_, offset_a_, offset_b_
    ));
}

std::string MatMulNode::toStr() const {
    std::stringstream ss;
    ss << "MatMul(";
    ss << "A=[";
    for (size_t i = 0; i < shape_a_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_a_[i]->__str__();
    }
    ss << "], strides_a=[";
    for (size_t i = 0; i < strides_a_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << strides_a_[i]->__str__();
    }
    ss << "], offset_a=" << offset_a_->__str__();
    ss << ", B=[";
    for (size_t i = 0; i < shape_b_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_b_[i]->__str__();
    }
    ss << "], strides_b=[";
    for (size_t i = 0; i < strides_b_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << strides_b_[i]->__str__();
    }
    ss << "], offset_b=" << offset_b_->__str__();
    ss << ")";
    return ss.str();
}

std::string copy_if_view(
    const std::string& name,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    types::PrimitiveType type,
    const symbolic::MultiExpression& shape,
    const symbolic::MultiExpression& strides,
    symbolic::Expression offset
) {
    // If the tensor is already a view (has non-default strides or offset), we need to create a copy to ensure correct
    // semantics
    types::Tensor tensor_type(type, shape, strides, offset);

    auto C_style_strides = tensor_type.strides_from_shape(shape);

    bool is_view = false;
    for (size_t i = 0; i < strides.size(); ++i) {
        if (!symbolic::eq(strides[i], C_style_strides[i])) {
            is_view = true;
            break;
        }
    }

    if (is_view) {
        std::string copy_name = builder.find_new_name(name + "_copy");
        types::Pointer copy_type((types::Scalar(types::PrimitiveType::Void)));
        builder.add_container(copy_name, copy_type);
        symbolic::Expression num_elements = symbolic::one();
        for (const auto& dim : shape) {
            num_elements = symbolic::mul(num_elements, dim);
        }
        auto elem_size = types::get_type_size(types::Scalar(type));
        auto copy_size = symbolic::mul(num_elements, elem_size);

        // Allocate a C-order copy
        auto& alloc_block = builder.add_block(parent, {}, DebugInfo());
        auto& out_access = builder.add_access(alloc_block, copy_name);
        auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(alloc_block, DebugInfo(), copy_size);
        builder.add_computational_memlet(
            alloc_block, malloc_node, "_ret", out_access, {}, types::Pointer(types::Scalar(type))
        );

        // Build a loop nest over each dimension
        structured_control_flow::Sequence* inner_scope = &parent;
        std::vector<symbolic::Expression> loop_vars;
        std::vector<symbolic::Expression> orig_accesses;
        for (size_t i = 0; i < shape.size(); ++i) {
            std::string indvar_str = builder.find_new_name(name + "_ci");
            builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));
            auto indvar = symbolic::symbol(indvar_str);
            auto init = symbolic::zero();
            auto update = symbolic::add(indvar, symbolic::one());
            auto condition = symbolic::Lt(indvar, shape[i]);
            auto& copy_map =
                builder.add_map(*inner_scope, indvar, condition, init, update, ScheduleType_Sequential::create());
            inner_scope = &copy_map.root();
            loop_vars.push_back(indvar);
        }

        // Inside the innermost loop: copy one element
        auto& copy_block = builder.add_block(*inner_scope);
        auto& in_access_copy = builder.add_access(copy_block, name);
        auto& out_access_copy = builder.add_access(copy_block, copy_name);
        auto& tasklet = builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"});

        // Read with original strides/offset
        builder.add_computational_memlet(copy_block, in_access_copy, tasklet, "_in", loop_vars, tensor_type);
        // Write with C-order strides (default strides, zero offset)
        types::Tensor c_order_type(type, shape);
        builder.add_computational_memlet(copy_block, tasklet, "_out", out_access_copy, loop_vars, c_order_type);

        return copy_name;
    } else {
        return name;
    }
}

void free_after_copy(
    const std::string& copy_name, builder::StructuredSDFGBuilder& builder, structured_control_flow::Sequence& parent
) {
    auto& block = builder.add_block(parent, {}, DebugInfo());
    auto& access_in = builder.add_access(block, copy_name);
    auto& access_out = builder.add_access(block, copy_name);
    auto& free_node = builder.add_library_node<stdlib::FreeNode>(block, DebugInfo());
    builder.add_computational_memlet(
        block, access_in, free_node, "_ptr", {}, types::Pointer(types::Scalar(types::PrimitiveType::Void))
    );
    builder.add_computational_memlet(
        block, free_node, "_ptr", access_out, {}, types::Pointer(types::Scalar(types::PrimitiveType::Void))
    );
}

bool MatMulNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());

    if (dataflow.in_degree(*this) != 2 || dataflow.out_degree(*this) != 1) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    // Get input and output edges
    data_flow::Memlet* iedge_a = nullptr;
    data_flow::Memlet* iedge_b = nullptr;
    for (auto& iedge : dataflow.in_edges(*this)) {
        if (iedge.dst_conn() == "A") {
            iedge_a = &iedge;
        } else if (iedge.dst_conn() == "B") {
            iedge_b = &iedge;
        }
    }
    auto& oedge = *dataflow.out_edges(*this).begin();

    if (!iedge_a || !iedge_b) {
        return false;
    }

    // Check if legal - access nodes must not have other connections
    auto& input_node_a = static_cast<data_flow::AccessNode&>(iedge_a->src());
    auto& input_node_b = static_cast<data_flow::AccessNode&>(iedge_b->src());
    auto& output_node = static_cast<data_flow::AccessNode&>(oedge.dst());

    if (dataflow.in_degree(input_node_a) != 0 || dataflow.in_degree(input_node_b) != 0 ||
        dataflow.out_degree(output_node) != 0) {
        return false;
    }

    // Determine BLAS precision from primitive type
    auto prim_type = this->primitive_type(dataflow);
    blas::BLAS_Precision precision;
    switch (prim_type) {
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

    auto copy_name_a =
        copy_if_view(input_node_a.data(), builder, new_sequence, prim_type, shape_a_, strides_a_, offset_a_);
    strides_a_ = types::Tensor::strides_from_shape(shape_a_);
    auto copy_name_b =
        copy_if_view(input_node_b.data(), builder, new_sequence, prim_type, shape_b_, strides_b_, offset_b_);
    strides_b_ = types::Tensor::strides_from_shape(shape_b_);

    // Create maps for batch dimensions and M, N dimensions
    structured_control_flow::Sequence* last_scope = &new_sequence;
    structured_control_flow::Map* last_map = nullptr;
    symbolic::MultiExpression batch_vars;

    // Compute batch dimensions (all except last 2)
    size_t batch_dims_a = shape_a_.size() - 2;
    size_t batch_dims_b = shape_b_.size() - 2;
    size_t max_batch_dims = std::max(batch_dims_a, batch_dims_b);

    auto& ref_block = builder.add_block(*last_scope, {}, block.debug_info());

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
            bound = shape_a_[a_idx]; // Assume they match or broadcasting is handled
        } else if (a_idx != SIZE_MAX) {
            bound = shape_a_[a_idx];
        } else {
            bound = shape_b_[b_idx];
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

    auto scalar_type = types::Scalar(prim_type);

    // Compute offsets for this batch iteration
    // For A: base_offset_a = offset_a + sum_i(batch_idx_i * batch_stride_a_i)
    symbolic::Expression a_batch_offset = offset_a_;
    for (size_t i = 0; i < batch_dims_a; ++i) {
        size_t batch_idx = max_batch_dims - batch_dims_a + i;
        a_batch_offset = symbolic::add(a_batch_offset, symbolic::mul(batch_vars[batch_idx], strides_a_[i]));
    }

    // For B: base_offset_b = offset_b + sum_i(batch_idx_i * batch_stride_b_i)
    symbolic::Expression b_batch_offset = offset_b_;
    for (size_t i = 0; i < batch_dims_b; ++i) {
        size_t batch_idx = max_batch_dims - batch_dims_b + i;
        b_batch_offset = symbolic::add(b_batch_offset, symbolic::mul(batch_vars[batch_idx], strides_b_[i]));
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
                c_stride = symbolic::mul(c_stride, shape_a_[j]);
            } else if (j - batch_dims_a < batch_dims_b) {
                c_stride = symbolic::mul(c_stride, shape_b_[j - batch_dims_a]);
            }
        }
        c_batch_offset = symbolic::add(c_batch_offset, symbolic::mul(batch_vars[i], c_stride));
    }

    // Create access nodes
    auto& a_access = builder.add_access(ref_block, copy_name_a, debug_info());
    auto& b_access = builder.add_access(ref_block, copy_name_b, debug_info());
    auto& c_access_in = builder.add_access(ref_block, output_node.data(), debug_info());

    std::string ref_name_a = builder.find_new_name(copy_name_a + "_ref");
    builder.add_container(ref_name_a, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
    auto& a_access_ref = builder.add_access(ref_block, ref_name_a, debug_info());
    std::string ref_name_b = builder.find_new_name(copy_name_b + "_ref");
    builder.add_container(ref_name_b, types::Pointer(types::Scalar(types::PrimitiveType::Void)));
    auto& b_access_ref = builder.add_access(ref_block, ref_name_b, debug_info());
    std::string ref_name_c = builder.find_new_name(output_node.data() + "_ref");
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
    // For row-major A[M, K]: lda = stride for M dimension = strides_a_[-2]
    // For row-major B[K, N]: ldb = stride for K dimension = strides_b_[-2]
    auto lda = strides_a_[strides_a_.size() - 2];
    auto ldb = strides_b_[strides_b_.size() - 2];
    // For output C[M, N] in row-major: ldc = N
    auto ldc = this->n();

    // Add GEMM node: C = alpha * A * B + beta * C
    // With alpha = 1.0, beta = 0.0: C = A * B
    auto& gemm_node = builder.add_library_node<blas::GEMMNode>(
        gemm_block,
        debug_info(),
        blas::ImplementationType_BLAS,
        precision,
        blas::BLAS_Layout::RowMajor,
        blas::BLAS_Transpose::No, // trans_a
        blas::BLAS_Transpose::No, // trans_b
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

    auto& c_access_ref_out = builder.add_access(gemm_block, ref_name_c, debug_info());

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
    // Output C
    builder.add_computational_memlet(
        gemm_block, gemm_node, "__C", c_access_ref_out, {}, ::sdfg::types::Pointer(scalar_type), debug_info()
    );

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
    builder.remove_memlet(block, oedge);
    if (&input_node_a != &input_node_b) {
        builder.remove_node(block, input_node_a);
    }
    builder.remove_node(block, input_node_b);
    builder.remove_node(block, output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
}

nlohmann::json MatMulNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const MatMulNode& matmul_node = static_cast<const MatMulNode&>(library_node);
    nlohmann::json j;

    j["code"] = matmul_node.code().value();

    serializer::JSONSerializer serializer;

    j["shape_a"] = nlohmann::json::array();
    for (auto& dim : matmul_node.shape_a()) {
        j["shape_a"].push_back(serializer.expression(dim));
    }

    j["shape_b"] = nlohmann::json::array();
    for (auto& dim : matmul_node.shape_b()) {
        j["shape_b"].push_back(serializer.expression(dim));
    }

    j["strides_a"] = nlohmann::json::array();
    for (auto& stride : matmul_node.strides_a()) {
        j["strides_a"].push_back(serializer.expression(stride));
    }

    j["strides_b"] = nlohmann::json::array();
    for (auto& stride : matmul_node.strides_b()) {
        j["strides_b"].push_back(serializer.expression(stride));
    }

    j["offset_a"] = serializer.expression(matmul_node.offset_a());
    j["offset_b"] = serializer.expression(matmul_node.offset_b());

    return j;
}

data_flow::LibraryNode& MatMulNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
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

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder
        .add_library_node<MatMulNode>(parent, debug_info, shape_a, shape_b, strides_a, strides_b, offset_a, offset_b);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
