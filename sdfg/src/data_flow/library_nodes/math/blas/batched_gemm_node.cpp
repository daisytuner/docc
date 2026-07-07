#include "sdfg/data_flow/library_nodes/math/blas/batched_gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace blas {

BatchedGEMMNode::BatchedGEMMNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const BLAS_Precision& precision,
    const BLAS_Layout& layout,
    const BLAS_Transpose& trans_a,
    const BLAS_Transpose& trans_b,
    symbolic::Expression batch_count,
    symbolic::Expression m,
    symbolic::Expression n,
    symbolic::Expression k,
    symbolic::Expression lda,
    symbolic::Expression ldb,
    symbolic::Expression ldc,
    symbolic::Expression stride_a,
    symbolic::Expression stride_b,
    symbolic::Expression stride_c
)
    : BLASNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_BatchedGEMM,
          {},
          {"__A", "__B", "__C", "__alpha", "__beta"},
          implementation_type,
          precision
      ),
      layout_(layout), trans_a_(trans_a), trans_b_(trans_b), batch_count_(batch_count), m_(m), n_(n), k_(k), lda_(lda),
      ldb_(ldb), ldc_(ldc), stride_a_(stride_a), stride_b_(stride_b), stride_c_(stride_c) {}

BLAS_Layout BatchedGEMMNode::layout() const { return this->layout_; }

BLAS_Transpose BatchedGEMMNode::trans_a() const { return this->trans_a_; }

BLAS_Transpose BatchedGEMMNode::trans_b() const { return this->trans_b_; }

symbolic::Expression BatchedGEMMNode::batch_count() const { return this->batch_count_; }

symbolic::Expression BatchedGEMMNode::m() const { return this->m_; }

symbolic::Expression BatchedGEMMNode::n() const { return this->n_; }

symbolic::Expression BatchedGEMMNode::k() const { return this->k_; }

symbolic::Expression BatchedGEMMNode::lda() const { return this->lda_; }

symbolic::Expression BatchedGEMMNode::ldb() const { return this->ldb_; }

symbolic::Expression BatchedGEMMNode::ldc() const { return this->ldc_; }

symbolic::Expression BatchedGEMMNode::stride_a() const { return this->stride_a_; }

symbolic::Expression BatchedGEMMNode::stride_b() const { return this->stride_b_; }

symbolic::Expression BatchedGEMMNode::stride_c() const { return this->stride_c_; }

symbolic::SymbolSet BatchedGEMMNode::symbols() const {
    symbolic::SymbolSet syms;

    for (auto& atom : symbolic::atoms(this->batch_count_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->m_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->n_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->k_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->lda_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->ldb_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->ldc_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->stride_a_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->stride_b_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(this->stride_c_)) {
        syms.insert(atom);
    }

    return syms;
}

void BatchedGEMMNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->batch_count_ = symbolic::subs(this->batch_count_, old_expression, new_expression);
    this->m_ = symbolic::subs(this->m_, old_expression, new_expression);
    this->n_ = symbolic::subs(this->n_, old_expression, new_expression);
    this->k_ = symbolic::subs(this->k_, old_expression, new_expression);
    this->lda_ = symbolic::subs(this->lda_, old_expression, new_expression);
    this->ldb_ = symbolic::subs(this->ldb_, old_expression, new_expression);
    this->ldc_ = symbolic::subs(this->ldc_, old_expression, new_expression);
    this->stride_a_ = symbolic::subs(this->stride_a_, old_expression, new_expression);
    this->stride_b_ = symbolic::subs(this->stride_b_, old_expression, new_expression);
    this->stride_c_ = symbolic::subs(this->stride_c_, old_expression, new_expression);
}

void BatchedGEMMNode::replace(const symbolic::ExpressionMapping& replacements) {
    this->batch_count_ = symbolic::subs(this->batch_count_, replacements);
    this->m_ = symbolic::subs(this->m_, replacements);
    this->n_ = symbolic::subs(this->n_, replacements);
    this->k_ = symbolic::subs(this->k_, replacements);
    this->lda_ = symbolic::subs(this->lda_, replacements);
    this->ldb_ = symbolic::subs(this->ldb_, replacements);
    this->ldc_ = symbolic::subs(this->ldc_, replacements);
    this->stride_a_ = symbolic::subs(this->stride_a_, replacements);
    this->stride_b_ = symbolic::subs(this->stride_b_, replacements);
    this->stride_c_ = symbolic::subs(this->stride_c_, replacements);
}

void BatchedGEMMNode::validate(const Function& function) const { BLASNode::validate(function); }

symbolic::Expression BatchedGEMMNode::flop() const {
    // batch_count * (2*m*n*k) approximately
    auto res_elems = symbolic::mul(this->m_, this->n_);
    auto mm_mul_ops = symbolic::mul(res_elems, this->k_);
    auto mm_sum_ops = symbolic::mul(res_elems, symbolic::sub(this->k_, symbolic::one()));
    auto per_batch = symbolic::add(mm_mul_ops, mm_sum_ops);
    return symbolic::mul(this->batch_count_, per_batch);
}

std::unique_ptr<data_flow::DataFlowNode> BatchedGEMMNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node_clone = std::unique_ptr<BatchedGEMMNode>(new BatchedGEMMNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
        this->precision_,
        this->layout_,
        this->trans_a_,
        this->trans_b_,
        this->batch_count_,
        this->m_,
        this->n_,
        this->k_,
        this->lda_,
        this->ldb_,
        this->ldc_,
        this->stride_a_,
        this->stride_b_,
        this->stride_c_
    ));
    return std::move(node_clone);
}

std::string BatchedGEMMNode::toStr() const {
    return LibraryNode::toStr() + "(" + static_cast<char>(precision_) + ", " +
           std::string(BLAS_Layout_to_short_string(layout_)) + ", " + BLAS_Transpose_to_char(trans_a_) +
           BLAS_Transpose_to_char(trans_b_) + ", batch=" + batch_count_->__str__() + ", " + m_->__str__() + ", " +
           n_->__str__() + ", " + k_->__str__() + ", lda=" + lda_->__str__() + ", ldb=" + ldb_->__str__() +
           ", ldc=" + ldc_->__str__() + ", strA=" + stride_a_->__str__() + ", strB=" + stride_b_->__str__() +
           ", strC=" + stride_c_->__str__() + ")";
}

data_flow::PointerAccessType BatchedGEMMNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) { // A: batched m x k
        auto per_batch_range = GEMMNode::calc_matrix_access_range(m_, k_, lda_, trans_a_, layout_);
        return data_flow::PointerAccessMeta::create_read_only(symbolic::mul(batch_count_, per_batch_range), true);
    } else if (input_idx == 1) { // B: batched k x n
        auto per_batch_range = GEMMNode::calc_matrix_access_range(k_, n_, ldb_, trans_b_, layout_);
        return data_flow::PointerAccessMeta::create_read_only(symbolic::mul(batch_count_, per_batch_range), true);
    } else if (input_idx == 2) {
        auto per_batch_range = GEMMNode::calc_matrix_access_range(m_, n_, ldc_, BLAS_Transpose::No, layout_);
        auto range = symbolic::mul(batch_count_, per_batch_range);

        // Match GEMM handling: use full-write for dense layout and generic access otherwise.
        if (symbolic::eq(ldc_, n_)) {
            return data_flow::PointerAccessMeta::create_full_write_only(range, true);
        } else {
            auto pattern = data_flow::ConvexAccessPattern::create(range);
            return data_flow::PointerAccessMeta::create_generic(pattern->ref(), std::move(pattern), true);
        }
    } else {
        return LibraryNode::pointer_access_type(input_idx);
    }
}

passes::LibNodeExpander::ExpandOutcome BatchedGEMMNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    if (trans_a_ == BLAS_Transpose::ConjTrans || trans_b_ == BLAS_Transpose::ConjTrans) {
        return context.unable();
    }

    auto primitive_type = scalar_primitive();
    if (primitive_type == types::PrimitiveType::Void) {
        return context.unable();
    }

    types::Scalar scalar_type(primitive_type);

    auto in_edges = dataflow.in_edges(*this);

    data_flow::Memlet* iedge_a = dataflow.in_edges_by_connector(*this).at(0);
    data_flow::Memlet* iedge_b = dataflow.in_edges_by_connector(*this).at(1);
    data_flow::Memlet* iedge_c = dataflow.in_edges_by_connector(*this).at(2);
    data_flow::Memlet* alpha_edge = dataflow.in_edges_by_connector(*this).at(3);
    data_flow::Memlet* beta_edge = dataflow.in_edges_by_connector(*this).at(4);

    using Use = passes::LibNodeExpander::InputUse;
    auto standalone = context.replacement_requires_access_nodes(
        {Use::IndirectRead, Use::IndirectRead, Use::IndirectWrite, Use::Scalar, Use::Scalar}
    );

    if (!standalone) {
        return context.unable();
    }

    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    // Batch loop (outermost)
    std::string batch_indvar_str = builder.find_new_name("_batch");
    builder.add_container(batch_indvar_str, types::Scalar(types::PrimitiveType::UInt64));
    auto batch_indvar = symbolic::symbol(batch_indvar_str);
    auto& batch_loop = builder.add_map(
        new_sequence,
        batch_indvar,
        symbolic::Lt(batch_indvar, this->batch_count()),
        symbolic::zero(),
        symbolic::add(batch_indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        {},
        block.debug_info()
    );

    // Inner loops: i (m), j (n), k (k)
    std::vector<symbolic::Expression> indvar_ends{this->m(), this->n(), this->k()};
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &batch_loop.root();
    structured_control_flow::StructuredLoop* last_map = nullptr;
    structured_control_flow::StructuredLoop* output_loop = nullptr;
    std::vector<std::string> indvar_names{"_i", "_j", "_k"};

    std::string sum_var = builder.find_new_name("_sum");
    builder.add_container(sum_var, scalar_type);

    for (size_t i = 0; i < 3; i++) {
        auto dim_begin = symbolic::zero();
        auto& dim_end = indvar_ends[i];

        std::string indvar_str = builder.find_new_name(indvar_names[i]);
        builder.add_container(indvar_str, types::Scalar(types::PrimitiveType::UInt64));

        auto indvar = symbolic::symbol(indvar_str);
        auto init = dim_begin;
        auto update = symbolic::add(indvar, symbolic::one());
        auto condition = symbolic::Lt(indvar, dim_end);
        if (i < 2) {
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
        } else {
            last_map = &builder.add_for(*last_scope, indvar, condition, init, update, {}, block.debug_info());
        }
        last_scope = &last_map->root();

        if (i == 1) {
            output_loop = last_map;
        }

        new_subset.push_back(indvar);
    }

    // Batch offsets for A, B, C
    auto a_batch_offset = symbolic::mul(batch_indvar, stride_a_);
    auto b_batch_offset = symbolic::mul(batch_indvar, stride_b_);
    auto c_batch_offset = symbolic::mul(batch_indvar, stride_c_);

    // Init sum = 0
    auto& init_block = builder.add_block_before(output_loop->root(), *last_map, {}, block.debug_info());
    auto& sum_init = builder.add_access(init_block, sum_var, block.debug_info());

    auto& zero_node = builder.add_constant(init_block, "0.0", alpha_edge->base_type(), block.debug_info());
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
    builder.add_computational_memlet(init_block, zero_node, init_tasklet, "_in", {}, block.debug_info());
    builder.add_computational_memlet(init_block, init_tasklet, "_out", sum_init, {}, block.debug_info());

    // FMA: sum += A[batch_offset + ...] * B[batch_offset + ...]
    auto& code_block = builder.add_block(*last_scope, {}, block.debug_info());
    auto& input_node_a_new = standalone->add_indirect_read_access(code_block, A_INPUT_IDX);
    auto& input_node_b_new = standalone->add_indirect_read_access(code_block, B_INPUT_IDX);

    auto& core_fma =
        builder.add_tasklet(code_block, data_flow::fp_fma, "_out", {"_in1", "_in2", "_in3"}, block.debug_info());
    auto& sum_in = builder.add_access(code_block, sum_var, block.debug_info());
    auto& sum_out = builder.add_access(code_block, sum_var, block.debug_info());
    builder.add_computational_memlet(code_block, sum_in, core_fma, "_in3", {}, block.debug_info());

    // Row-major indexing with batch offset
    // No transpose: A is m×k, access A[batch*stride_a + lda*i + k]
    // Transpose:    A is k×m stored, access A[batch*stride_a + lda*k + i]
    symbolic::Expression a_idx =
        (trans_a_ == BLAS_Transpose::Trans)
            ? symbolic::add(a_batch_offset, symbolic::add(symbolic::mul(lda(), new_subset[2]), new_subset[0]))
            : symbolic::add(a_batch_offset, symbolic::add(symbolic::mul(lda(), new_subset[0]), new_subset[2]));
    builder.add_computational_memlet(
        code_block, input_node_a_new, core_fma, "_in1", {a_idx}, iedge_a->base_type(), iedge_a->debug_info()
    );
    // No transpose: B is k×n, access B[batch*stride_b + ldb*k + j]
    // Transpose:    B is n×k stored, access B[batch*stride_b + ldb*j + k]
    symbolic::Expression b_idx =
        (trans_b_ == BLAS_Transpose::Trans)
            ? symbolic::add(b_batch_offset, symbolic::add(symbolic::mul(ldb(), new_subset[1]), new_subset[2]))
            : symbolic::add(b_batch_offset, symbolic::add(symbolic::mul(ldb(), new_subset[2]), new_subset[1]));
    builder.add_computational_memlet(
        code_block, input_node_b_new, core_fma, "_in2", {b_idx}, iedge_b->base_type(), iedge_b->debug_info()
    );
    builder.add_computational_memlet(code_block, core_fma, "_out", sum_out, {}, iedge_c->debug_info());

    // Flush: C[batch*stride_c + ldc*i + j] = alpha * sum + beta * C[...]
    auto& flush_block = builder.add_block_after(output_loop->root(), *last_map, {}, block.debug_info());
    auto& sum_final = builder.add_access(flush_block, sum_var, block.debug_info());
    auto& input_node_c_new = standalone->add_indirect_read_access(flush_block, C_INPUT_IDX);
    symbolic::Expression c_idx =
        symbolic::add(c_batch_offset, symbolic::add(symbolic::mul(ldc(), new_subset[0]), new_subset[1]));

    // alpha * sum
    auto& scale_sum_tasklet =
        builder.add_tasklet(flush_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, block.debug_info());
    builder.add_computational_memlet(flush_block, sum_final, scale_sum_tasklet, "_in1", {}, block.debug_info());
    auto& alpha_node = standalone->add_scalar_input_access(flush_block, ALPHA_INPUT_IDX);
    builder.add_computational_memlet(flush_block, alpha_node, scale_sum_tasklet, "_in2", {}, block.debug_info());

    std::string scaled_sum_temp = builder.find_new_name("scaled_sum_temp");
    builder.add_container(scaled_sum_temp, scalar_type);
    auto& scaled_sum_final = builder.add_access(flush_block, scaled_sum_temp, block.debug_info());
    builder.add_computational_memlet(
        flush_block, scale_sum_tasklet, "_out", scaled_sum_final, {}, scalar_type, block.debug_info()
    );

    // beta * C[...]
    auto& scale_input_tasklet =
        builder.add_tasklet(flush_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, block.debug_info());
    builder.add_computational_memlet(
        flush_block, input_node_c_new, scale_input_tasklet, "_in1", {c_idx}, iedge_c->base_type(), iedge_c->debug_info()
    );
    auto& beta_node = standalone->add_scalar_input_access(flush_block, BETA_INPUT_IDX);
    builder.add_computational_memlet(flush_block, beta_node, scale_input_tasklet, "_in2", {}, block.debug_info());

    std::string scaled_input_temp = builder.find_new_name("scaled_input_temp");
    builder.add_container(scaled_input_temp, scalar_type);
    auto& scaled_input_c = builder.add_access(flush_block, scaled_input_temp, block.debug_info());
    builder.add_computational_memlet(
        flush_block, scale_input_tasklet, "_out", scaled_input_c, {}, scalar_type, block.debug_info()
    );

    // alpha*sum + beta*C
    auto& flush_add_tasklet =
        builder.add_tasklet(flush_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, block.debug_info());
    auto& output_node_new = standalone->add_indirect_write_access(flush_block, C_INPUT_IDX);
    builder.add_computational_memlet(
        flush_block, scaled_sum_final, flush_add_tasklet, "_in1", {}, scalar_type, block.debug_info()
    );
    builder.add_computational_memlet(
        flush_block, scaled_input_c, flush_add_tasklet, "_in2", {}, scalar_type, block.debug_info()
    );
    builder.add_computational_memlet(
        flush_block, flush_add_tasklet, "_out", output_node_new, {c_idx}, iedge_c->base_type(), iedge_c->debug_info()
    );

    return standalone->successfully_expanded();
}

nlohmann::json BatchedGEMMNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const BatchedGEMMNode& node = static_cast<const BatchedGEMMNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = node.code().value();
    j["precision"] = node.precision();
    j["layout"] = node.layout();
    j["trans_a"] = node.trans_a();
    j["trans_b"] = node.trans_b();
    j["batch_count"] = serializer.expression(node.batch_count());
    j["m"] = serializer.expression(node.m());
    j["n"] = serializer.expression(node.n());
    j["k"] = serializer.expression(node.k());
    j["lda"] = serializer.expression(node.lda());
    j["ldb"] = serializer.expression(node.ldb());
    j["ldc"] = serializer.expression(node.ldc());
    j["stride_a"] = serializer.expression(node.stride_a());
    j["stride_b"] = serializer.expression(node.stride_b());
    j["stride_c"] = serializer.expression(node.stride_c());

    return j;
}

data_flow::LibraryNode& BatchedGEMMNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_BatchedGEMM.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto precision = j.at("precision").get<BLAS_Precision>();
    auto layout = j.at("layout").get<BLAS_Layout>();
    auto trans_a = j.at("trans_a").get<BLAS_Transpose>();
    auto trans_b = j.at("trans_b").get<BLAS_Transpose>();
    auto batch_count = symbolic::parse(j.at("batch_count"));
    auto m = symbolic::parse(j.at("m"));
    auto n = symbolic::parse(j.at("n"));
    auto k = symbolic::parse(j.at("k"));
    auto lda = symbolic::parse(j.at("lda"));
    auto ldb = symbolic::parse(j.at("ldb"));
    auto ldc = symbolic::parse(j.at("ldc"));
    auto stride_a = symbolic::parse(j.at("stride_a"));
    auto stride_b = symbolic::parse(j.at("stride_b"));
    auto stride_c = symbolic::parse(j.at("stride_c"));

    auto implementation_type = j.at("implementation_type").get<std::string>();

    return builder.add_library_node<BatchedGEMMNode>(
        parent,
        debug_info,
        implementation_type,
        precision,
        layout,
        trans_a,
        trans_b,
        batch_count,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        stride_a,
        stride_b,
        stride_c
    );
}

} // namespace blas
} // namespace math
} // namespace sdfg
