#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"

#include "sdfg/analysis/scope_analysis.h"

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
          {"__C"},
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

void GEMMNode::validate(const Function& function) const { BLASNode::validate(function); }

bool GEMMNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();

    auto& dataflow = this->get_parent();
    auto& block = static_cast<structured_control_flow::Block&>(*dataflow.get_parent());
    auto& parent = static_cast<structured_control_flow::Sequence&>(*scope_analysis.parent_scope(&block));
    int index = parent.index(block);
    auto& transition = parent.at(index).second;

    if (trans_a_ == BLAS_Transpose::ConjTrans || trans_b_ == BLAS_Transpose::ConjTrans) {
        return false;
    }

    auto primitive_type = scalar_primitive();
    if (primitive_type == types::PrimitiveType::Void) {
        return false;
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

    auto& oedge = *dataflow.out_edges(*this).begin();

    // Checks if legal
    auto* input_node_a = static_cast<data_flow::AccessNode*>(&iedge_a->src());
    auto* input_node_b = static_cast<data_flow::AccessNode*>(&iedge_b->src());
    auto* input_node_c = static_cast<data_flow::AccessNode*>(&iedge_c->src());
    auto* output_node = static_cast<data_flow::AccessNode*>(&oedge.dst());
    auto* alpha_node = static_cast<data_flow::AccessNode*>(&alpha_edge->src());
    auto* beta_node = static_cast<data_flow::AccessNode*>(&beta_edge->src());

    // we must be the only thing in this block, as we do not support splitting a block into pre, expanded lib-node, post
    if (!input_node_a || dataflow.in_degree(*input_node_a) != 0 || !input_node_b ||
        dataflow.in_degree(*input_node_b) != 0 || !input_node_c || dataflow.in_degree(*input_node_c) != 0 ||
        !output_node || dataflow.out_degree(*output_node) != 0) {
        return false; // data nodes are not standalone
    }
    if (dataflow.in_degree(*alpha_node) != 0 || dataflow.in_degree(*beta_node) != 0) {
        return false; // alpha and beta are not standalone
    }
    for (auto* nd : dataflow.data_nodes()) {
        if (nd != input_node_a && nd != input_node_b && nd != input_node_c && nd != output_node &&
            (!alpha_node || nd != alpha_node) && (!beta_node || nd != beta_node)) {
            return false; // there are other nodes in here that we could not preserve correctly
        }
    }

    auto& A_var = input_node_a->data();
    auto& B_var = input_node_b->data();
    auto& C_in_var = input_node_c->data();
    auto& C_out_var = output_node->data();


    // Add new graph after the current block
    auto& new_sequence = builder.add_sequence_before(parent, block, transition.assignments(), block.debug_info());

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


    // Add code
    auto& init_block = builder.add_block_before(output_loop->root(), *last_map, {}, block.debug_info());
    auto& sum_init = builder.add_access(init_block, sum_var, block.debug_info());

    auto& zero_node = builder.add_constant(init_block, "0.0", alpha_edge->base_type(), block.debug_info());
    auto& init_tasklet = builder.add_tasklet(init_block, data_flow::assign, "_out", {"_in"}, block.debug_info());
    builder.add_computational_memlet(init_block, zero_node, init_tasklet, "_in", {}, block.debug_info());
    builder.add_computational_memlet(init_block, init_tasklet, "_out", sum_init, {}, block.debug_info());

    auto& code_block = builder.add_block(*last_scope, {}, block.debug_info());
    auto& input_node_a_new = builder.add_access(code_block, A_var, input_node_a->debug_info());
    auto& input_node_b_new = builder.add_access(code_block, B_var, input_node_b->debug_info());

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
    builder.add_computational_memlet(code_block, core_fma, "_out", sum_out, {}, oedge.debug_info());

    auto& flush_block = builder.add_block_after(output_loop->root(), *last_map, {}, block.debug_info());
    auto& sum_final = builder.add_access(flush_block, sum_var, block.debug_info());
    auto& input_node_c_new = builder.add_access(flush_block, C_in_var, input_node_c->debug_info());
    symbolic::Expression c_idx = symbolic::add(symbolic::mul(ldc(), new_subset[0]), new_subset[1]);

    auto& scale_sum_tasklet =
        builder.add_tasklet(flush_block, data_flow::TaskletCode::fp_mul, "_out", {"_in1", "_in2"}, block.debug_info());
    builder.add_computational_memlet(flush_block, sum_final, scale_sum_tasklet, "_in1", {}, block.debug_info());
    if (auto const_node = dynamic_cast<data_flow::ConstantNode*>(alpha_node)) {
        auto& alpha_node_new =
            builder.add_constant(flush_block, const_node->data(), const_node->type(), block.debug_info());
        builder.add_computational_memlet(flush_block, alpha_node_new, scale_sum_tasklet, "_in2", {}, block.debug_info());
    } else {
        auto& alpha_node_new = builder.add_access(flush_block, alpha_node->data(), block.debug_info());
        builder.add_computational_memlet(flush_block, alpha_node_new, scale_sum_tasklet, "_in2", {}, block.debug_info());
    }

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
    if (auto const_node = dynamic_cast<data_flow::ConstantNode*>(beta_node)) {
        auto& beta_node_new =
            builder.add_constant(flush_block, const_node->data(), const_node->type(), block.debug_info());
        builder
            .add_computational_memlet(flush_block, beta_node_new, scale_input_tasklet, "_in2", {}, block.debug_info());
    } else {
        auto& beta_node_new = builder.add_access(flush_block, beta_node->data(), block.debug_info());
        builder
            .add_computational_memlet(flush_block, beta_node_new, scale_input_tasklet, "_in2", {}, block.debug_info());
    }

    std::string scaled_input_temp = builder.find_new_name("scaled_input_temp");
    builder.add_container(scaled_input_temp, scalar_type);
    auto& scaled_input_c = builder.add_access(flush_block, scaled_input_temp, block.debug_info());
    builder.add_computational_memlet(
        flush_block, scale_input_tasklet, "_out", scaled_input_c, {}, scalar_type, block.debug_info()
    );

    auto& flush_add_tasklet =
        builder.add_tasklet(flush_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, block.debug_info());
    auto& output_node_new = builder.add_access(flush_block, C_out_var, output_node->debug_info());
    builder.add_computational_memlet(
        flush_block, scaled_sum_final, flush_add_tasklet, "_in1", {}, scalar_type, block.debug_info()
    );
    builder.add_computational_memlet(
        flush_block, scaled_input_c, flush_add_tasklet, "_in2", {}, scalar_type, block.debug_info()
    );
    builder.add_computational_memlet(
        flush_block, flush_add_tasklet, "_out", output_node_new, {c_idx}, iedge_c->base_type(), iedge_c->debug_info()
    );


    // Clean up block
    builder.remove_memlet(block, *iedge_a);
    builder.remove_memlet(block, *iedge_b);
    builder.remove_memlet(block, *iedge_c);
    builder.remove_memlet(block, *alpha_edge);
    builder.remove_node(block, *alpha_node);
    builder.remove_memlet(block, *beta_edge);
    builder.remove_node(block, *beta_node);
    builder.remove_memlet(block, oedge);
    builder.remove_node(block, *input_node_a);
    builder.remove_node(block, *input_node_b);
    builder.remove_node(block, *input_node_c);
    builder.remove_node(block, *output_node);
    builder.remove_node(block, *this);
    builder.remove_child(parent, index + 1);

    return true;
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

void GEMMNodeDispatcher_BLAS::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

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

    stream << "cblas_" << BLAS_Precision_to_string(gemm_node.precision()) << "gemm(";
    stream.setIndent(stream.indent() + 4);
    stream << BLAS_Layout_to_string(gemm_node.layout());
    stream << ", ";
    stream << BLAS_Transpose_to_string(gemm_node.trans_a());
    stream << ", ";
    stream << BLAS_Transpose_to_string(gemm_node.trans_b());
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.m());
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.n());
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.k());
    stream << ", ";
    stream << "__alpha";
    stream << ", ";
    stream << "__A";
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.lda());
    stream << ", ";
    stream << "__B";
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.ldb());
    stream << ", ";
    stream << "__beta";
    stream << ", ";
    stream << "__C";
    stream << ", ";
    stream << this->language_extension_.expression(gemm_node.ldc());

    stream.setIndent(stream.indent() - 4);
    stream << ");" << std::endl;

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace math
} // namespace sdfg
