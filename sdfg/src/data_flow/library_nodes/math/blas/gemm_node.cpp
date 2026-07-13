#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

namespace sdfg {
namespace math {
namespace blas {

GEMMNode::GEMMNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const data_flow::ImplementationType& implementation_type,
    const BLAS_Precision& precision,
    const BLAS_Layout& layout,
    const BLAS_Transpose& trans_a,
    const BLAS_Transpose& trans_b,
    symbolic::Expression m,
    symbolic::Expression n,
    symbolic::Expression k,
    symbolic::Expression lda,
    symbolic::Expression ldb,
    symbolic::Expression ldc
)
    : BLASNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_GEMM,
          {},
          {"__A", "__B", "__C", "__alpha", "__beta"},
          implementation_type,
          precision
      ),
      layout_(layout), trans_a_(trans_a), trans_b_(trans_b), m_(m), n_(n), k_(k), lda_(lda), ldb_(ldb), ldc_(ldc) {}

BLAS_Layout GEMMNode::layout() const { return this->layout_; };

BLAS_Transpose GEMMNode::trans_a() const { return this->trans_a_; };

BLAS_Transpose GEMMNode::trans_b() const { return this->trans_b_; };

symbolic::Expression GEMMNode::m() const { return this->m_; };

symbolic::Expression GEMMNode::n() const { return this->n_; };

symbolic::Expression GEMMNode::k() const { return this->k_; };

symbolic::Expression GEMMNode::lda() const { return this->lda_; };

symbolic::Expression GEMMNode::ldb() const { return this->ldb_; };

symbolic::Expression GEMMNode::ldc() const { return this->ldc_; };

symbolic::SymbolSet GEMMNode::symbols() const {
    symbolic::SymbolSet syms;

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

    return syms;
};

void GEMMNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    this->m_ = symbolic::subs(this->m_, old_expression, new_expression);
    this->n_ = symbolic::subs(this->n_, old_expression, new_expression);
    this->k_ = symbolic::subs(this->k_, old_expression, new_expression);
    this->lda_ = symbolic::subs(this->lda_, old_expression, new_expression);
    this->ldb_ = symbolic::subs(this->ldb_, old_expression, new_expression);
    this->ldc_ = symbolic::subs(this->ldc_, old_expression, new_expression);
};

void GEMMNode::replace(const symbolic::ExpressionMapping& replacements) {
    this->m_ = symbolic::subs(this->m_, replacements);
    this->n_ = symbolic::subs(this->n_, replacements);
    this->k_ = symbolic::subs(this->k_, replacements);
    this->lda_ = symbolic::subs(this->lda_, replacements);
    this->ldb_ = symbolic::subs(this->ldb_, replacements);
    this->ldc_ = symbolic::subs(this->ldc_, replacements);
};

void GEMMNode::validate(const Function& function) const { BLASNode::validate(function); }

passes::LibNodeExpander::ExpandOutcome GEMMNode::
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
    auto in_edges_it = in_edges.begin();

    data_flow::Memlet* iedge_a = nullptr;
    data_flow::Memlet* iedge_b = nullptr;
    data_flow::Memlet* iedge_c = nullptr;
    data_flow::Memlet* alpha_edge = nullptr;
    data_flow::Memlet* beta_edge = nullptr;
    while (in_edges_it != in_edges.end()) {
        auto& edge = *in_edges_it;
        auto dst_conn = edge.dst_conn();
        if (dst_conn == "__A") {
            iedge_a = &edge;
        } else if (dst_conn == "__B") {
            iedge_b = &edge;
        } else if (dst_conn == "__C") {
            iedge_c = &edge;
        } else if (dst_conn == "__alpha") {
            alpha_edge = &edge;
        } else if (dst_conn == "__beta") {
            beta_edge = &edge;
        } else {
            throw InvalidSDFGException("GEMMNode has unexpected input: " + dst_conn);
        }
        ++in_edges_it;
    }

    using Dir = passes::LibNodeExpander::InputUse;
    auto standalone = context.replacement_requires_access_nodes(
        {Dir::IndirectRead, Dir::IndirectRead, Dir::IndirectReadWrite, Dir::Scalar, Dir::Scalar}
    );

    if (!standalone) {
        return context.unable();
    }

    // Add new graph after the current block
    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    // Add maps
    std::vector<symbolic::Expression> indvar_ends{this->m(), this->n(), this->k()};
    data_flow::Subset new_subset;
    structured_control_flow::Sequence* last_scope = &new_sequence;
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
                block.debug_info()
            );
        } else {
            last_map = &builder.add_for(*last_scope, indvar, condition, init, update, block.debug_info());
        }
        last_scope = &last_map->root();

        if (i == 1) {
            output_loop = last_map;
        }

        new_subset.push_back(indvar);
    }


    // Add code
    auto& init_block = builder.add_block_before(output_loop->root(), *last_map, block.debug_info());
    auto& sum_init = builder.add_access(init_block, sum_var, block.debug_info());

    auto& zero_node = builder.add_constant(init_block, "0.0", alpha_edge->base_type(), block.debug_info());
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
    builder.add_computational_memlet(init_block, zero_node, init_tasklet, "_in", {}, block.debug_info());
    builder.add_computational_memlet(init_block, init_tasklet, "_out", sum_init, {}, block.debug_info());

    auto& code_block = builder.add_block(*last_scope, block.debug_info());
    auto& input_node_a_new = standalone->add_indirect_read_access(code_block, A_INPUT_IDX);
    auto& input_node_b_new = standalone->add_indirect_read_access(code_block, B_INPUT_IDX);

    auto& core_fma =
        builder.add_tasklet(code_block, data_flow::fp_fma, "_out", {"_in1", "_in2", "_in3"}, block.debug_info());
    auto& sum_in = builder.add_access(code_block, sum_var, block.debug_info());
    auto& sum_out = builder.add_access(code_block, sum_var, block.debug_info());
    builder.add_computational_memlet(code_block, sum_in, core_fma, "_in3", {}, block.debug_info());

    // Row-major indexing: address = ld * row + col
    // No transpose: A is m×k, access A[i, k] => lda*i + k
    // Transpose:    A is k×m stored, access A[k, i] => lda*k + i
    symbolic::Expression a_idx = (trans_a_ == BLAS_Transpose::Trans)
                                     ? symbolic::add(symbolic::mul(lda(), new_subset[2]), new_subset[0])
                                     : symbolic::add(symbolic::mul(lda(), new_subset[0]), new_subset[2]);
    builder.add_computational_memlet(
        code_block, input_node_a_new, core_fma, "_in1", {a_idx}, iedge_a->base_type(), iedge_a->debug_info()
    );
    // No transpose: B is k×n, access B[k, j] => ldb*k + j
    // Transpose:    B is n×k stored, access B[j, k] => ldb*j + k
    symbolic::Expression b_idx = (trans_b_ == BLAS_Transpose::Trans)
                                     ? symbolic::add(symbolic::mul(ldb(), new_subset[1]), new_subset[2])
                                     : symbolic::add(symbolic::mul(ldb(), new_subset[2]), new_subset[1]);
    builder.add_computational_memlet(
        code_block, input_node_b_new, core_fma, "_in2", {b_idx}, iedge_b->base_type(), iedge_b->debug_info()
    );
    builder.add_computational_memlet(code_block, core_fma, "_out", sum_out, {}, iedge_c->debug_info());

    auto& flush_block = builder.add_block_after(output_loop->root(), *last_map, block.debug_info());
    auto& sum_final = builder.add_access(flush_block, sum_var, block.debug_info());
    auto& input_node_c_new = standalone->add_indirect_read_access(flush_block, C_INPUT_IDX);
    symbolic::Expression c_idx = symbolic::add(symbolic::mul(ldc(), new_subset[0]), new_subset[1]);

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

symbolic::Expression GEMMNode::flop() const {
    return flops(symbolic::__true__(), symbolic::__true__(), symbolic::__true__(), symbolic::__true__());
}

symbolic::Expression GEMMNode::flops(
    symbolic::Condition alpha_non_zero,
    symbolic::Condition alpha_non_ident,
    symbolic::Condition beta_non_zero,
    symbolic::Condition beta_non_ident
) const {
    auto res_elems = symbolic::mul(this->m_, this->n_);

    // conditional on alpha != 0.0
    auto mm_mul_ops = symbolic::mul(symbolic::mul(res_elems, this->k_), alpha_non_zero);
    auto mm_sum_ops = symbolic::mul(symbolic::mul(res_elems, symbolic::sub(this->k_, symbolic::one())), alpha_non_zero);
    // conditional on alpha != 1.0 && alpha != 0.0
    auto mm_alpha_scale_ops = symbolic::mul(res_elems, symbolic::And(alpha_non_ident, alpha_non_zero));
    // conditional on beta != 1.0 && beta != 0.0
    auto mm_beta_scale_ops = symbolic::mul(res_elems, symbolic::And(beta_non_ident, beta_non_zero));
    auto mm_beta_scaled_sum_ops = symbolic::mul(res_elems, beta_non_zero);
    auto mul_ops = symbolic::add(mm_mul_ops, symbolic::add(mm_alpha_scale_ops, mm_beta_scale_ops));
    auto add_ops = symbolic::add(mm_sum_ops, mm_beta_scaled_sum_ops);
    return symbolic::add(mul_ops, add_ops);
}

std::unique_ptr<data_flow::DataFlowNode> GEMMNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    auto node_clone = std::unique_ptr<GEMMNode>(new GEMMNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        this->implementation_type_,
        this->precision_,
        this->layout_,
        this->trans_a_,
        this->trans_b_,
        this->m_,
        this->n_,
        this->k_,
        this->lda_,
        this->ldb_,
        this->ldc_
    ));
    return std::move(node_clone);
}

std::string GEMMNode::toStr() const {
    return LibraryNode::toStr() + "(" + static_cast<char>(precision_) + ", " +
           std::string(BLAS_Layout_to_short_string(layout_)) + ", " + BLAS_Transpose_to_char(trans_a_) +
           BLAS_Transpose_to_char(trans_b_) + ", " + m_->__str__() + ", " + n_->__str__() + ", " + k_->__str__() +
           ", " + lda_->__str__() + ", " + ldb_->__str__() + ", " + ldc_->__str__() + ")";
}

symbolic::Expression GEMMNode::calc_matrix_access_range(
    const symbolic::Expression& outer_dim,
    const symbolic::Expression& inner_dim,
    const symbolic::Expression& line_size,
    BLAS_Transpose trans,
    BLAS_Layout layout
) {
    if ((trans == BLAS_Transpose::No) ^ (layout == BLAS_Layout::ColMajor)) {
        return symbolic::mul(outer_dim, line_size);
    } else {
        return symbolic::mul(inner_dim, line_size);
    }
}


data_flow::PointerAccessType GEMMNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) { // A: m x k
        return data_flow::PointerAccessMeta::
            create_read_only(calc_matrix_access_range(m_, k_, lda_, trans_a_, layout_), true);
    } else if (input_idx == 1) { // B: k x n
        return data_flow::PointerAccessMeta::
            create_read_only(calc_matrix_access_range(k_, n_, ldb_, trans_b_, layout_), true);
    } else if (input_idx == 2) {
        // for beta == 0, there would no reads of C. But we currently have no mechanism to access const-prop knowledge
        // like tha
        if (symbolic::eq(ldc_, n_)) { // non-sparse access over the m x n range
            return data_flow::PointerAccessMeta::
                create_full_write_only(calc_matrix_access_range(m_, n_, ldc_, BLAS_Transpose::No, layout_), true);
        } else {
            // sparse access. But with only Convex Pattern for now, we cannot represent which values are
            auto pattern =
                data_flow::ConvexAccessPattern::create(calc_matrix_access_range(m_, n_, ldc_, BLAS_Transpose::No, layout_)
                );
            // full-overwritten and which are DC.
            return data_flow::PointerAccessMeta::create_generic(pattern->ref(), std::move(pattern), true);
        }
    } else {
        return LibraryNode::pointer_access_type(input_idx);
    }
}

nlohmann::json GEMMNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const GEMMNode& gemm_node = static_cast<const GEMMNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["code"] = gemm_node.code().value();
    j["precision"] = gemm_node.precision();
    j["layout"] = gemm_node.layout();
    j["trans_a"] = gemm_node.trans_a();
    j["trans_b"] = gemm_node.trans_b();
    j["m"] = serializer.expression(gemm_node.m());
    j["n"] = serializer.expression(gemm_node.n());
    j["k"] = serializer.expression(gemm_node.k());
    j["lda"] = serializer.expression(gemm_node.lda());
    j["ldb"] = serializer.expression(gemm_node.ldb());
    j["ldc"] = serializer.expression(gemm_node.ldc());

    return j;
}

data_flow::LibraryNode& GEMMNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    // Assertions for required fields
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto code = j["code"].get<std::string>();
    if (code != LibraryNodeType_GEMM.value()) {
        throw std::runtime_error("Invalid library node code");
    }

    // Extract debug info using JSONSerializer
    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    auto precision = j.at("precision").get<BLAS_Precision>();
    auto layout = j.at("layout").get<BLAS_Layout>();
    auto trans_a = j.at("trans_a").get<BLAS_Transpose>();
    auto trans_b = j.at("trans_b").get<BLAS_Transpose>();
    auto m = symbolic::parse(j.at("m"));
    auto n = symbolic::parse(j.at("n"));
    auto k = symbolic::parse(j.at("k"));
    auto lda = symbolic::parse(j.at("lda"));
    auto ldb = symbolic::parse(j.at("ldb"));
    auto ldc = symbolic::parse(j.at("ldc"));

    auto implementation_type = j.at("implementation_type").get<std::string>();

    return builder.add_library_node<
        GEMMNode>(parent, debug_info, implementation_type, precision, layout, trans_a, trans_b, m, n, k, lda, ldb, ldc);
}

GEMMNodeDispatcher_BLAS::GEMMNodeDispatcher_BLAS(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const GEMMNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void GEMMNodeDispatcher_BLAS::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& gemm_node = static_cast<const GEMMNode&>(this->node_);

    sdfg::types::Scalar base_type(types::PrimitiveType::Void);
    switch (gemm_node.precision()) {
        case BLAS_Precision::h:
            base_type = types::Scalar(types::PrimitiveType::Half);
            break;
        case BLAS_Precision::s:
            base_type = types::Scalar(types::PrimitiveType::Float);
            break;
        case BLAS_Precision::d:
            base_type = types::Scalar(types::PrimitiveType::Double);
            break;
        default:
            throw std::runtime_error("Invalid BLAS_Precision value");
    }

    out.library_snippet_factory.require_dependency(BLASLibDependency::instance());

    out.stream << "cblas_" << BLAS_Precision_to_string(gemm_node.precision()) << "gemm(";
    out.stream.changeIndent(+4);
    out.stream << BLAS_Layout_to_string(gemm_node.layout());
    out.stream << ", ";
    out.stream << BLAS_Transpose_to_string(gemm_node.trans_a());
    out.stream << ", ";
    out.stream << BLAS_Transpose_to_string(gemm_node.trans_b());
    out.stream << ", ";
    out.stream << this->language_extension_.expression(gemm_node.m());
    out.stream << ", ";
    out.stream << this->language_extension_.expression(gemm_node.n());
    out.stream << ", ";
    out.stream << this->language_extension_.expression(gemm_node.k());
    out.stream << ", ";
    out.stream << inputs.at(GEMMNode::ALPHA_INPUT_IDX).expr;
    out.stream << ", ";
    out.stream << inputs.at(GEMMNode::A_INPUT_IDX).expr;
    out.stream << ", ";
    out.stream << this->language_extension_.expression(gemm_node.lda());
    out.stream << ", ";
    out.stream << inputs.at(GEMMNode::B_INPUT_IDX).expr;
    out.stream << ", ";
    out.stream << this->language_extension_.expression(gemm_node.ldb());
    out.stream << ", ";
    out.stream << inputs.at(GEMMNode::BETA_INPUT_IDX).expr;
    out.stream << ", ";
    out.stream << inputs.at(GEMMNode::C_INPUT_IDX).expr;
    out.stream << ", ";
    out.stream << this->language_extension_.expression(gemm_node.ldc());

    out.stream.changeIndent(-4);
    out.stream << ");" << std::endl;
}

GEMMNode& add_gemm_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& ptr_a,
    const std::string& ptr_b,
    const std::string& ptr_c,
    data_flow::AccessNode& alpha_node,
    data_flow::AccessNode& beta_node,
    const BLAS_Precision& precision,
    const BLAS_Layout& layout,
    const BLAS_Transpose& trans_a,
    const BLAS_Transpose& trans_b,
    symbolic::Expression& m,
    symbolic::Expression& n,
    symbolic::Expression& k,
    symbolic::Expression& lda,
    symbolic::Expression& ldb,
    symbolic::Expression& ldc,
    const types::IType& a_type,
    const types::IType& b_type,
    const types::IType& c_type,
    const types::IType& factor_type,
    DebugInfo debug_info,
    DebugInfo a_access_deb_info,
    DebugInfo b_access_deb_info,
    DebugInfo c_access_deb_info,
    DebugInfo a_edge_deb_info,
    DebugInfo b_edge_deb_info,
    DebugInfo c_edge_deb_info,
    data_flow::ImplementationType impl_type
) {
    auto& gemm_node = builder.add_library_node<sdfg::math::blas::GEMMNode>(
        block, debug_info, std::move(impl_type), precision, layout, trans_a, trans_b, m, n, k, lda, ldb, ldc
    );

    // Add access nodes
    auto& a_node_in = builder.add_access(block, ptr_a, a_access_deb_info);
    auto& b_node_in = builder.add_access(block, ptr_b, b_access_deb_info);
    auto& c_node_in = builder.add_access(block, ptr_c, c_access_deb_info);

    // Add edges
    builder.add_computational_memlet(block, a_node_in, gemm_node, "__A", {}, a_type, a_edge_deb_info);
    builder.add_computational_memlet(block, b_node_in, gemm_node, "__B", {}, b_type, b_edge_deb_info);
    builder.add_computational_memlet(block, c_node_in, gemm_node, "__C", {}, c_type, c_edge_deb_info);
    builder.add_computational_memlet(block, alpha_node, gemm_node, "__alpha", {}, factor_type, debug_info);
    builder.add_computational_memlet(block, beta_node, gemm_node, "__beta", {}, factor_type, debug_info);

    return static_cast<GEMMNode&>(gemm_node);
}

GEMMNode& add_gemm_node(
    builder::StructuredSDFGBuilder& builder,
    Block& block,
    const std::string& ptr_a,
    const std::string& ptr_b,
    const std::string& ptr_c,
    data_flow::AccessNode& alpha_node,
    data_flow::AccessNode& beta_node,
    const BLAS_Precision& precision,
    const BLAS_Layout& layout,
    const BLAS_Transpose& trans_a,
    const BLAS_Transpose& trans_b,
    symbolic::Expression& m,
    symbolic::Expression& n,
    symbolic::Expression& k,
    symbolic::Expression& lda,
    symbolic::Expression& ldb,
    symbolic::Expression& ldc,
    const types::IType& ptr_type,
    const types::IType& factor_type,
    DebugInfo debug_info,
    data_flow::ImplementationType impl_type
) {
    return add_gemm_node(
        builder,
        block,
        ptr_a,
        ptr_b,
        ptr_c,
        alpha_node,
        beta_node,
        precision,
        layout,
        trans_a,
        trans_b,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        ptr_type,
        ptr_type,
        ptr_type,
        factor_type,
        debug_info,
        debug_info,
        debug_info,
        debug_info,
        debug_info,
        debug_info,
        debug_info,
        impl_type
    );
}

} // namespace blas
} // namespace math
} // namespace sdfg
