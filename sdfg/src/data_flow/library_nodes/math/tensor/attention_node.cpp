#include "sdfg/data_flow/library_nodes/math/tensor/attention_node.h"

#include <memory>
#include <sstream>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

AttentionNode::AttentionNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& o_layout,
    const TensorLayout& q_layout,
    const TensorLayout& k_layout,
    const TensorLayout& v_layout,
    double scale,
    bool is_causal,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_Attention, {}, {"O", "Q", "K", "V"}, impl_type),
      o_layout_(o_layout), q_layout_(q_layout), k_layout_(k_layout), v_layout_(v_layout), mask_layout_(std::nullopt),
      scale_(scale), is_causal_(is_causal), fixed_quantization_(quantization) {}

AttentionNode::AttentionNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& o_layout,
    const TensorLayout& q_layout,
    const TensorLayout& k_layout,
    const TensorLayout& v_layout,
    const TensorLayout& mask_layout,
    double scale,
    bool is_causal,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Attention, {}, {"O", "Q", "K", "V", "M"}, impl_type
      ),
      o_layout_(o_layout), q_layout_(q_layout), k_layout_(k_layout), v_layout_(v_layout), mask_layout_(mask_layout),
      scale_(scale), is_causal_(is_causal), fixed_quantization_(quantization) {}

void AttentionNode::validate(const Function& function) const { TensorNode::validate(function); }

passes::LibNodeExpander::ExpandOutcome AttentionNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    using data_flow::TaskletCode;
    namespace scf = structured_control_flow;

    auto& dataflow = this->get_parent();

    const bool has_mask = mask_layout_.has_value();
    const size_t n_in = has_mask ? 5 : 4;
    if (dataflow.in_degree(*this) != n_in) {
        return context.unable();
    }

    // Arbitrary batch/head leading dims; last two are [seq, head]. All operands share the rank.
    const size_t rank = o_layout_.dims();
    if (rank < 2 || q_layout_.dims() != rank || k_layout_.dims() != rank || v_layout_.dims() != rank) {
        return context.unable();
    }
    if (has_mask && mask_layout_->dims() != rank) {
        return context.unable();
    }
    const size_t num_lead = rank - 2;

    auto* o_edge = dataflow.in_edge_for_connector(*this, "O");
    auto* q_edge = dataflow.in_edge_for_connector(*this, "Q");
    auto* k_edge = dataflow.in_edge_for_connector(*this, "K");
    auto* v_edge = dataflow.in_edge_for_connector(*this, "V");
    auto* m_edge = has_mask ? dataflow.in_edge_for_connector(*this, "M") : nullptr;
    if (!o_edge || !q_edge || !k_edge || !v_edge || (has_mask && !m_edge)) {
        return context.unable();
    }

    const auto& seq_q = o_layout_.get_dim(num_lead);
    const auto& head_d = q_layout_.get_dim(rank - 1);
    const auto& seq_k = k_layout_.get_dim(num_lead);
    const auto& head_dv = v_layout_.get_dim(rank - 1);

    std::vector<passes::LibNodeExpander::InputUse> dirs = {
        passes::LibNodeExpander::InputUse::IndirectReadWrite,
        passes::LibNodeExpander::InputUse::IndirectRead,
        passes::LibNodeExpander::InputUse::IndirectRead,
        passes::LibNodeExpander::InputUse::IndirectRead
    };
    if (has_mask) {
        dirs.push_back(passes::LibNodeExpander::InputUse::IndirectRead);
    }
    auto access = context.replacement_requires_access_nodes(dirs);
    if (!access) {
        return context.unable();
    }

    auto& seq = access->replace_with_sequence();
    auto& builder = access->builder();

    const auto prim = this->primitive_type(dataflow);
    types::Scalar elem(prim);
    const auto& o_type = o_edge->base_type();
    const auto& q_type = q_edge->base_type();
    const auto& k_type = k_edge->base_type();
    const auto& v_type = v_edge->base_type();
    const auto& dbg = this->debug_info();

    auto scalar = [&](const std::string& base) {
        std::string name = builder.find_new_name(base);
        builder.add_container(name, elem);
        return name;
    };
    auto indvar = [&](const std::string& base, const symbolic::Expression& limit) {
        std::string name = builder.find_new_name(base);
        builder.add_container(name, types::Scalar(types::get_primitive_type_to_hold_upper_bound(limit)));
        return symbolic::symbol(name);
    };

    std::ostringstream scale_ss;
    scale_ss.precision(17);
    scale_ss << scale_;
    const std::string scale_str = scale_ss.str();

    // Outer parallel maps over the batch/head leading dims, then the query position.
    std::vector<symbolic::Expression> lead;
    lead.reserve(num_lead);
    structured_control_flow::Sequence* scope = &seq;
    for (size_t d = 0; d < num_lead; ++d) {
        const auto& limit = o_layout_.get_dim(d);
        auto sym = indvar("_b", limit);
        auto& mp = builder.add_map(
            *scope,
            sym,
            symbolic::Lt(sym, limit),
            symbolic::zero(),
            symbolic::add(sym, symbolic::one()),
            scf::ScheduleType_Sequential::create(),
            dbg
        );
        scope = &mp.root();
        lead.push_back(sym);
    }
    auto i_sym = indvar("_i", seq_q);
    auto& imap = builder.add_map(
        *scope,
        i_sym,
        symbolic::Lt(i_sym, seq_q),
        symbolic::zero(),
        symbolic::add(i_sym, symbolic::one()),
        scf::ScheduleType_Sequential::create(),
        dbg
    );
    auto* iscope = &imap.root();

    // Key/value leading index per dim: identity, broadcast (dim == 1), or grouped (GQA/MQA: q/k groups).
    std::vector<symbolic::Expression> kvlead;
    kvlead.reserve(num_lead);
    for (size_t d = 0; d < num_lead; ++d) {
        const auto& qd = q_layout_.get_dim(d);
        const auto& kd = k_layout_.get_dim(d);
        if (symbolic::eq(qd, kd)) {
            kvlead.push_back(lead[d]);
        } else if (symbolic::eq(kd, symbolic::one())) {
            kvlead.push_back(symbolic::zero());
        } else {
            kvlead.push_back(symbolic::div(lead[d], symbolic::div(qd, kd)));
        }
    }
    auto with = [](const std::vector<symbolic::Expression>& head,
                   const symbolic::Expression& row,
                   const symbolic::Expression& col) {
        std::vector<symbolic::Expression> s = head;
        s.push_back(row);
        s.push_back(col);
        return s;
    };
    auto q_sub = [&](const symbolic::Expression& col) { return with(lead, i_sym, col); };
    auto o_sub = [&](const symbolic::Expression& col) { return with(lead, i_sym, col); };
    auto k_sub = [&](const symbolic::Expression& row, const symbolic::Expression& col) {
        return with(kvlead, row, col);
    };
    // Mask subset with per-axis broadcasting (leading, query, key axes may be size 1).
    auto mask_sub = [&](const symbolic::Expression& row, const symbolic::Expression& col) {
        const symbolic::Expression zero_e = symbolic::zero();
        std::vector<symbolic::Expression> s;
        for (size_t d = 0; d < num_lead; ++d) {
            s.push_back(symbolic::eq(mask_layout_->get_dim(d), symbolic::one()) ? zero_e : lead[d]);
        }
        s.push_back(symbolic::eq(mask_layout_->get_dim(num_lead), symbolic::one()) ? zero_e : row);
        s.push_back(symbolic::eq(mask_layout_->get_dim(rank - 1), symbolic::one()) ? zero_e : col);
        return s;
    };

    const std::string m_name = scalar("_attn_max");
    const std::string l_name = scalar("_attn_denom");
    const std::string mold_name = scalar("_attn_maxold");
    const std::string corr_name = scalar("_attn_corr");
    const std::string p_name = scalar("_attn_p");
    const std::string s_name = scalar("_attn_score");
    const std::string dmm_name = scalar("_attn_dmax");
    const std::string smm_name = scalar("_attn_ssub");

    // m = -INFINITY
    {
        auto& blk = builder.add_block(*iscope, {}, dbg);
        auto& t = builder.add_tasklet(blk, TaskletCode::assign, {"_out"}, {"_in"}, dbg);
        auto& c = builder.add_constant(blk, "-INFINITY", elem, dbg);
        auto& w = builder.add_access(blk, m_name, dbg);
        builder.add_computational_memlet(blk, c, t, "_in", {}, elem, dbg);
        builder.add_computational_memlet(blk, t, "_out", w, {}, elem, dbg);
    }
    // l = 0
    {
        auto& blk = builder.add_block(*iscope, {}, dbg);
        auto& t = builder.add_tasklet(blk, TaskletCode::assign, {"_out"}, {"_in"}, dbg);
        auto& c = builder.add_constant(blk, "0.0", elem, dbg);
        auto& w = builder.add_access(blk, l_name, dbg);
        builder.add_computational_memlet(blk, c, t, "_in", {}, elem, dbg);
        builder.add_computational_memlet(blk, t, "_out", w, {}, elem, dbg);
    }
    // O[i, e] = 0  (the running output accumulator lives in O itself)
    {
        auto e0 = indvar("_e", head_dv);
        auto& eloop = builder.add_for(
            *iscope, e0, symbolic::Lt(e0, head_dv), symbolic::zero(), symbolic::add(e0, symbolic::one()), dbg
        );
        auto& blk = builder.add_block(eloop.root(), {}, dbg);
        auto& t = builder.add_tasklet(blk, TaskletCode::assign, {"_out"}, {"_in"}, dbg);
        auto& c = builder.add_constant(blk, "0.0", elem, dbg);
        auto& o_w = access->add_indirect_write_access(blk, O_INPUT_IDX);
        builder.add_computational_memlet(blk, c, t, "_in", {}, elem, dbg);
        builder.add_computational_memlet(blk, t, "_out", o_w, o_sub(e0), o_type, dbg);
    }

    // Streaming pass over keys/values (causal => keys up to the query index).
    auto j_limit = is_causal_ ? symbolic::add(i_sym, symbolic::one()) : seq_k;
    auto j_sym = indvar("_j", seq_k);
    auto& jloop = builder.add_for(
        *iscope, j_sym, symbolic::Lt(j_sym, j_limit), symbolic::zero(), symbolic::add(j_sym, symbolic::one()), dbg
    );
    auto* jscope = &jloop.root();

    // s = 0
    {
        auto& blk = builder.add_block(*jscope, {}, dbg);
        auto& t = builder.add_tasklet(blk, TaskletCode::assign, {"_out"}, {"_in"}, dbg);
        auto& c = builder.add_constant(blk, "0.0", elem, dbg);
        auto& w = builder.add_access(blk, s_name, dbg);
        builder.add_computational_memlet(blk, c, t, "_in", {}, elem, dbg);
        builder.add_computational_memlet(blk, t, "_out", w, {}, elem, dbg);
    }
    // s += Q[i, c] * K[j, c]
    {
        auto c_sym = indvar("_c", head_d);
        auto& cloop = builder.add_for(
            *jscope, c_sym, symbolic::Lt(c_sym, head_d), symbolic::zero(), symbolic::add(c_sym, symbolic::one()), dbg
        );
        auto& blk = builder.add_block(cloop.root(), {}, dbg);
        auto& fma = builder.add_tasklet(blk, TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"}, dbg);
        auto& q_a = access->add_indirect_read_access(blk, Q_INPUT_IDX);
        auto& k_a = access->add_indirect_read_access(blk, K_INPUT_IDX);
        auto& s_r = builder.add_access(blk, s_name, dbg);
        auto& s_w = builder.add_access(blk, s_name, dbg);
        builder.add_computational_memlet(blk, q_a, fma, "_in1", q_sub(c_sym), q_type, dbg);
        builder.add_computational_memlet(blk, k_a, fma, "_in2", k_sub(j_sym, c_sym), k_type, dbg);
        builder.add_computational_memlet(blk, s_r, fma, "_in3", {}, elem, dbg);
        builder.add_computational_memlet(blk, fma, "_out", s_w, {}, elem, dbg);
    }
    // s = s * scale
    {
        auto& blk = builder.add_block(*jscope, {}, dbg);
        auto& mul = builder.add_tasklet(blk, TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"}, dbg);
        auto& s_r = builder.add_access(blk, s_name, dbg);
        auto& c = builder.add_constant(blk, scale_str, elem, dbg);
        auto& s_w = builder.add_access(blk, s_name, dbg);
        builder.add_computational_memlet(blk, s_r, mul, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, c, mul, "_in2", {}, elem, dbg);
        builder.add_computational_memlet(blk, mul, "_out", s_w, {}, elem, dbg);
    }
    // s += mask[..., i, j]  (additive attention bias, with per-axis broadcasting)
    if (has_mask) {
        const auto& mask_type = m_edge->base_type();
        auto& blk = builder.add_block(*jscope, {}, dbg);
        auto& add = builder.add_tasklet(blk, TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"}, dbg);
        auto& s_r = builder.add_access(blk, s_name, dbg);
        auto& mask_a = access->add_indirect_read_access(blk, MASK_INPUT_IDX);
        auto& s_w = builder.add_access(blk, s_name, dbg);
        builder.add_computational_memlet(blk, s_r, add, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, mask_a, add, "_in2", mask_sub(i_sym, j_sym), mask_type, dbg);
        builder.add_computational_memlet(blk, add, "_out", s_w, {}, elem, dbg);
    }
    // Online-softmax update of the running max m and denominator l.
    {
        auto& blk = builder.add_block(*jscope, {}, dbg);
        // mold = m
        auto& m_r = builder.add_access(blk, m_name, dbg);
        auto& snap = builder.add_tasklet(blk, TaskletCode::assign, {"_out"}, {"_in"}, dbg);
        auto& mold_w = builder.add_access(blk, mold_name, dbg);
        builder.add_computational_memlet(blk, m_r, snap, "_in", {}, elem, dbg);
        builder.add_computational_memlet(blk, snap, "_out", mold_w, {}, elem, dbg);
        // m = fmax(mold, s)
        auto& fmax = builder.add_library_node<cmath::CMathNode>(blk, dbg, cmath::CMathFunction::fmax, prim);
        auto& s_r1 = builder.add_access(blk, s_name, dbg);
        auto& m_w = builder.add_access(blk, m_name, dbg);
        builder.add_computational_memlet(blk, mold_w, fmax, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, s_r1, fmax, "_in2", {}, elem, dbg);
        builder.add_computational_memlet(blk, fmax, "_out", m_w, {}, elem, dbg);
        // corr = exp(mold - m)
        auto& sub1 = builder.add_tasklet(blk, TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"}, dbg);
        auto& dmm = builder.add_access(blk, dmm_name, dbg);
        builder.add_computational_memlet(blk, mold_w, sub1, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, m_w, sub1, "_in2", {}, elem, dbg);
        builder.add_computational_memlet(blk, sub1, "_out", dmm, {}, elem, dbg);
        auto& exp1 = builder.add_library_node<cmath::CMathNode>(blk, dbg, cmath::CMathFunction::exp, prim);
        auto& corr_w = builder.add_access(blk, corr_name, dbg);
        builder.add_computational_memlet(blk, dmm, exp1, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, exp1, "_out", corr_w, {}, elem, dbg);
        // p = exp(s - m)
        auto& sub2 = builder.add_tasklet(blk, TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"}, dbg);
        auto& s_r2 = builder.add_access(blk, s_name, dbg);
        auto& smm = builder.add_access(blk, smm_name, dbg);
        builder.add_computational_memlet(blk, s_r2, sub2, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, m_w, sub2, "_in2", {}, elem, dbg);
        builder.add_computational_memlet(blk, sub2, "_out", smm, {}, elem, dbg);
        auto& exp2 = builder.add_library_node<cmath::CMathNode>(blk, dbg, cmath::CMathFunction::exp, prim);
        auto& p_w = builder.add_access(blk, p_name, dbg);
        builder.add_computational_memlet(blk, smm, exp2, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, exp2, "_out", p_w, {}, elem, dbg);
        // l = l * corr + p
        auto& fma_l = builder.add_tasklet(blk, TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"}, dbg);
        auto& l_r = builder.add_access(blk, l_name, dbg);
        auto& l_w = builder.add_access(blk, l_name, dbg);
        builder.add_computational_memlet(blk, l_r, fma_l, "_in1", {}, elem, dbg);
        builder.add_computational_memlet(blk, corr_w, fma_l, "_in2", {}, elem, dbg);
        builder.add_computational_memlet(blk, p_w, fma_l, "_in3", {}, elem, dbg);
        builder.add_computational_memlet(blk, fma_l, "_out", l_w, {}, elem, dbg);
    }
    // O[i, e] = O[i, e] * corr + p * V[j, e]  (rescale the accumulator, then add this key).
    // Two single-tasklet read-modify-writes on O (strictly ordered as separate blocks); avoids a
    // reused temporary that the intra-block scheduler would treat as loop-carried.
    {
        auto e_sym = indvar("_e", head_dv);
        auto& eloop = builder.add_for(
            *jscope, e_sym, symbolic::Lt(e_sym, head_dv), symbolic::zero(), symbolic::add(e_sym, symbolic::one()), dbg
        );
        // O[i, e] *= corr
        {
            auto& blk = builder.add_block(eloop.root(), {}, dbg);
            auto& mul = builder.add_tasklet(blk, TaskletCode::fp_mul, {"_out"}, {"_in1", "_in2"}, dbg);
            auto& o_r = access->add_indirect_read_access(blk, O_INPUT_IDX);
            auto& corr_r = builder.add_access(blk, corr_name, dbg);
            auto& o_w = access->add_indirect_write_access(blk, O_INPUT_IDX);
            builder.add_computational_memlet(blk, o_r, mul, "_in1", o_sub(e_sym), o_type, dbg);
            builder.add_computational_memlet(blk, corr_r, mul, "_in2", {}, elem, dbg);
            builder.add_computational_memlet(blk, mul, "_out", o_w, o_sub(e_sym), o_type, dbg);
        }
        // O[i, e] += p * V[j, e]
        {
            auto& blk = builder.add_block(eloop.root(), {}, dbg);
            auto& fma_o = builder.add_tasklet(blk, TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"}, dbg);
            auto& p_r = builder.add_access(blk, p_name, dbg);
            auto& v_a = access->add_indirect_read_access(blk, V_INPUT_IDX);
            auto& o_r = access->add_indirect_read_access(blk, O_INPUT_IDX);
            auto& o_w = access->add_indirect_write_access(blk, O_INPUT_IDX);
            builder.add_computational_memlet(blk, p_r, fma_o, "_in1", {}, elem, dbg);
            builder.add_computational_memlet(blk, v_a, fma_o, "_in2", k_sub(j_sym, e_sym), v_type, dbg);
            builder.add_computational_memlet(blk, o_r, fma_o, "_in3", o_sub(e_sym), o_type, dbg);
            builder.add_computational_memlet(blk, fma_o, "_out", o_w, o_sub(e_sym), o_type, dbg);
        }
    }
    // O[i, e] = O[i, e] / l  (final normalization)
    {
        auto en = indvar("_e", head_dv);
        auto& nloop = builder.add_for(
            *iscope, en, symbolic::Lt(en, head_dv), symbolic::zero(), symbolic::add(en, symbolic::one()), dbg
        );
        auto& blk = builder.add_block(nloop.root(), {}, dbg);
        auto& div = builder.add_tasklet(blk, TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"}, dbg);
        auto& o_r = access->add_indirect_read_access(blk, O_INPUT_IDX);
        auto& l_r = builder.add_access(blk, l_name, dbg);
        auto& o_w = access->add_indirect_write_access(blk, O_INPUT_IDX);
        builder.add_computational_memlet(blk, o_r, div, "_in1", o_sub(en), o_type, dbg);
        builder.add_computational_memlet(blk, l_r, div, "_in2", {}, elem, dbg);
        builder.add_computational_memlet(blk, div, "_out", o_w, o_sub(en), o_type, dbg);
    }

    return access->successfully_expanded();
}

symbolic::SymbolSet AttentionNode::symbols() const {
    symbolic::SymbolSet syms;
    o_layout_.collect_symbols(syms);
    q_layout_.collect_symbols(syms);
    k_layout_.collect_symbols(syms);
    v_layout_.collect_symbols(syms);
    if (mask_layout_.has_value()) {
        mask_layout_->collect_symbols(syms);
    }
    return syms;
}

void AttentionNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    o_layout_.replace_symbols(old_expression, new_expression);
    q_layout_.replace_symbols(old_expression, new_expression);
    k_layout_.replace_symbols(old_expression, new_expression);
    v_layout_.replace_symbols(old_expression, new_expression);
    if (mask_layout_.has_value()) {
        mask_layout_->replace_symbols(old_expression, new_expression);
    }
}

void AttentionNode::replace(const symbolic::ExpressionMapping& replacements) {
    o_layout_.replace_symbols(replacements);
    q_layout_.replace_symbols(replacements);
    k_layout_.replace_symbols(replacements);
    v_layout_.replace_symbols(replacements);
    if (mask_layout_.has_value()) {
        mask_layout_->replace_symbols(replacements);
    }
}

std::unique_ptr<data_flow::DataFlowNode> AttentionNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    if (mask_layout_.has_value()) {
        return std::make_unique<AttentionNode>(
            element_id,
            this->debug_info(),
            vertex,
            parent,
            o_layout_,
            q_layout_,
            k_layout_,
            v_layout_,
            *mask_layout_,
            scale_,
            is_causal_,
            fixed_quantization_,
            this->implementation_type()
        );
    }
    return std::make_unique<AttentionNode>(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        o_layout_,
        q_layout_,
        k_layout_,
        v_layout_,
        scale_,
        is_causal_,
        fixed_quantization_,
        this->implementation_type()
    );
}

std::string AttentionNode::toStr() const {
    std::stringstream ss;
    ss << "Attention(scale=" << scale_ << ", causal=" << (is_causal_ ? "true" : "false") << ", Q: " << q_layout_
       << ", K: " << k_layout_ << ", V: " << v_layout_ << ")";
    return ss.str();
}

symbolic::Expression AttentionNode::flop() const {
    // 2·Nq·Nk·D (Q Kᵀ) + 2·Nq·Nk·Dv (P V).
    auto qk = symbolic::mul(seq_q(), symbolic::mul(seq_k(), head_dim()));
    auto pv = symbolic::mul(seq_q(), symbolic::mul(seq_k(), head_dim_v()));
    return symbolic::mul(symbolic::integer(2), symbolic::add(qk, pv));
}

nlohmann::json AttentionNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const AttentionNode& node = static_cast<const AttentionNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    node.o_layout().serialize_to_json(j["layout_o"]);
    node.q_layout().serialize_to_json(j["layout_q"]);
    node.k_layout().serialize_to_json(j["layout_k"]);
    node.v_layout().serialize_to_json(j["layout_v"]);
    if (node.mask_layout().has_value()) {
        node.mask_layout()->serialize_to_json(j["layout_m"]);
    }
    j["scale"] = node.scale();
    j["is_causal"] = node.is_causal();
    j["result_quant"] = node.quantization();
    return j;
}

data_flow::LibraryNode& AttentionNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));

    auto o_layout = TensorLayout::deserialize_from_json(j.at("layout_o"));
    auto q_layout = TensorLayout::deserialize_from_json(j.at("layout_q"));
    auto k_layout = TensorLayout::deserialize_from_json(j.at("layout_k"));
    auto v_layout = TensorLayout::deserialize_from_json(j.at("layout_v"));
    double scale = j.at("scale").get<double>();
    bool is_causal = j.at("is_causal").get<bool>();
    auto quantization = deserialize_quantization(j, "result_quant", QUANTIZATION_MATCH_INPUTS);

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    if (j.contains("layout_m")) {
        auto mask_layout = TensorLayout::deserialize_from_json(j.at("layout_m"));
        return builder.add_library_node<AttentionNode>(
            parent, debug_info, o_layout, q_layout, k_layout, v_layout, mask_layout, scale, is_causal, quantization
        );
    }
    return builder.add_library_node<
        AttentionNode>(parent, debug_info, o_layout, q_layout, k_layout, v_layout, scale, is_causal, quantization);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
