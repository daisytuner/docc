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

    // Emit C's initialization as its own (m, n) nest, then a separate accumulate
    // nest. Accumulating in place (C[i,j] = A*B + C[i,j]) instead of into a
    // register reduction keeps the k loop a plain, interchangeable loop the
    // vectorizer can transform (matching the numpy C[i,j] += form).
    std::vector<symbolic::Expression> indvar_ends{this->m(), this->n(), this->k()};
    std::vector<std::string> indvar_names{"_i", "_j", "_k"};

    bool alpha_is_one = alpha_edge->is_src_constant(1.0);
    bool beta_is_zero = beta_edge->is_src_constant(0.0);
    bool beta_is_one = beta_edge->is_src_constant(1.0);

    auto add_loop = [&](structured_control_flow::Sequence& scope, size_t dim, bool as_map
                    ) -> structured_control_flow::StructuredLoop& {
        std::string iv = builder.find_new_name(indvar_names[dim]);
        builder.add_container(iv, types::Scalar(types::PrimitiveType::UInt64));
        auto sym = symbolic::symbol(iv);
        auto cond = symbolic::Lt(sym, indvar_ends[dim]);
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

    // ---- Init nest: for i, j: C[i,j] = (beta == 0 ? 0 : beta * C[i,j]) ----
    // Skipped when beta == 1 (identity): C already holds its starting value, so
    // only the accumulate nest remains -- a single, unfused, vectorizable loop.
    if (!beta_is_one) {
        auto& i_loop = add_loop(new_sequence, 0, true);
        auto i_sym = i_loop.indvar();
        auto& j_loop = add_loop(i_loop.root(), 1, true);
        auto j_sym = j_loop.indvar();
        auto& init_blk = builder.add_block(j_loop.root(), block.debug_info());
        symbolic::Expression c_idx = symbolic::add(symbolic::mul(ldc(), i_sym), j_sym);
        auto& c_write = standalone->add_indirect_write_access(init_blk, C_INPUT_IDX);

        if (beta_is_zero) {
            auto& zero_node = builder.add_constant(init_blk, "0.0", scalar_type, block.debug_info());
            auto& t = builder.add_tasklet(init_blk, data_flow::assign, "_out", {"_in"}, block.debug_info());
            builder.add_computational_memlet(init_blk, zero_node, t, "_in", {}, block.debug_info());
            builder.add_computational_memlet(
                init_blk, t, "_out", c_write, {c_idx}, iedge_c->base_type(), iedge_c->debug_info()
            );
        } else {
            auto& c_read = standalone->add_indirect_read_access(init_blk, C_INPUT_IDX);
            auto& beta_node = standalone->add_scalar_input_access(init_blk, BETA_INPUT_IDX);
            auto& t =
                builder
                    .add_tasklet(init_blk, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, block.debug_info());
            builder.add_computational_memlet(
                init_blk, c_read, t, "_in1", {c_idx}, iedge_c->base_type(), iedge_c->debug_info()
            );
            builder.add_computational_memlet(init_blk, beta_node, t, "_in2", {}, block.debug_info());
            builder.add_computational_memlet(
                init_blk, t, "_out", c_write, {c_idx}, iedge_c->base_type(), iedge_c->debug_info()
            );
        }
    }

    // ---- Compute nest: for i, j, k: C[i,j] = alpha * A[i,k] * B[k,j] + C[i,j] ----
    {
        auto& i_loop = add_loop(new_sequence, 0, true);
        auto i_sym = i_loop.indvar();
        auto& j_loop = add_loop(i_loop.root(), 1, true);
        auto j_sym = j_loop.indvar();
        auto& k_loop = add_loop(j_loop.root(), 2, false);
        auto k_sym = k_loop.indvar();
        auto& code_block = builder.add_block(k_loop.root(), block.debug_info());

        // Row-major indexing: address = ld * row + col.
        symbolic::Expression a_idx = (trans_a_ == BLAS_Transpose::Trans)
                                         ? symbolic::add(symbolic::mul(lda(), k_sym), i_sym)
                                         : symbolic::add(symbolic::mul(lda(), i_sym), k_sym);
        symbolic::Expression b_idx = (trans_b_ == BLAS_Transpose::Trans)
                                         ? symbolic::add(symbolic::mul(ldb(), j_sym), k_sym)
                                         : symbolic::add(symbolic::mul(ldb(), k_sym), j_sym);
        symbolic::Expression c_idx = symbolic::add(symbolic::mul(ldc(), i_sym), j_sym);

        auto& a_node = standalone->add_indirect_read_access(code_block, A_INPUT_IDX);
        auto& b_node = standalone->add_indirect_read_access(code_block, B_INPUT_IDX);
        auto& c_in = standalone->add_indirect_read_access(code_block, C_INPUT_IDX);
        auto& c_out = standalone->add_indirect_write_access(code_block, C_INPUT_IDX);

        auto& fma =
            builder.add_tasklet(code_block, data_flow::fp_fma, "_out", {"_in1", "_in2", "_in3"}, block.debug_info());
        builder
            .add_computational_memlet(code_block, c_in, fma, "_in3", {c_idx}, iedge_c->base_type(), iedge_c->debug_info());
        builder.add_computational_memlet(
            code_block, fma, "_out", c_out, {c_idx}, iedge_c->base_type(), iedge_c->debug_info()
        );

        if (alpha_is_one) {
            // C[i,j] = A[i,k] * B[k,j] + C[i,j]
            builder.add_computational_memlet(
                code_block, a_node, fma, "_in1", {a_idx}, iedge_a->base_type(), iedge_a->debug_info()
            );
            builder.add_computational_memlet(
                code_block, b_node, fma, "_in2", {b_idx}, iedge_b->base_type(), iedge_b->debug_info()
            );
        } else {
            // p = A[i,k] * B[k,j]; C[i,j] = alpha * p + C[i,j]
            auto& mul_t =
                builder
                    .add_tasklet(code_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, block.debug_info());
            builder.add_computational_memlet(
                code_block, a_node, mul_t, "_in1", {a_idx}, iedge_a->base_type(), iedge_a->debug_info()
            );
            builder.add_computational_memlet(
                code_block, b_node, mul_t, "_in2", {b_idx}, iedge_b->base_type(), iedge_b->debug_info()
            );
            std::string prod = builder.find_new_name("_prod");
            builder.add_container(prod, scalar_type);
            auto& prod_node = builder.add_access(code_block, prod, block.debug_info());
            builder.add_computational_memlet(code_block, mul_t, "_out", prod_node, {}, scalar_type, block.debug_info());

            auto& alpha_node = standalone->add_scalar_input_access(code_block, ALPHA_INPUT_IDX);
            builder.add_computational_memlet(code_block, alpha_node, fma, "_in1", {}, block.debug_info());
            builder.add_computational_memlet(code_block, prod_node, fma, "_in2", {}, scalar_type, block.debug_info());
        }
    }

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

codegen::InstrumentationInfo GEMMNodeDispatcher_BLAS::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        codegen::TargetType_CPU_PARALLEL,
        codegen::InstrumentationEventType::CPU,
        analysis::LoopInfo{},
        {}
    };
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
