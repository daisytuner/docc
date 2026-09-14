#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/softmax_node.h"

#include <algorithm>
#include <map>
#include <string>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"

namespace sdfg {
namespace math {
namespace tensor {

SoftmaxNode::SoftmaxNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<int64_t>& axes,
    bool keepdims
)
    : ReduceNode(element_id, debug_info, vertex, parent, LibraryNodeType_Softmax, shape, axes, keepdims) {
    if (keepdims) {
        throw InvalidSDFGException("Unsupported attribute on library node: softmax");
    }
}

void SoftmaxNode::validate(const Function& function) const {}

passes::LibNodeExpander::ExpandOutcome SoftmaxNode::expand(passes::LibNodeExpander::ExpandContext& context, Block& block) {
    auto& dataflow = this->get_parent();

    // Select the online (log-sum-exp monoid) variant; defaults to the 3-pass form.
    const bool softmax_use_online = context.options().get(passes::LibraryNodeExpansionPass<>::ONLINE_SOFTMAX, false);

    if (dataflow.in_degree(*this) != 2) {
        return context.unable();
    }

    auto* in_edge = dataflow.in_edge_for_connector(*this, "X");
    auto* out_edge = dataflow.in_edge_for_connector(*this, "Y");
    if (!in_edge || !out_edge) {
        return context.unable();
    }

    // Normalize and validate axes
    std::vector<int64_t> sorted_axes = axes_;
    for (auto& axis : sorted_axes) {
        if (axis < 0) {
            axis = static_cast<int64_t>(shape_.size()) + axis;
        }
        if (axis < 0 || axis >= static_cast<int64_t>(shape_.size())) {
            throw InvalidSDFGException(
                "Library Node: Axis value out of bounds. Axis: " + std::to_string(axis) +
                " Shape size: " + std::to_string(shape_.size())
            );
        }
    }
    std::sort(sorted_axes.begin(), sorted_axes.end());

    // Partition dimensions: non-reduced (outer, parallel) vs. reduced (inner, sequential)
    std::vector<size_t> outer_dims;
    std::vector<size_t> inner_dims;
    for (size_t i = 0; i < shape_.size(); ++i) {
        bool is_axis = std::find(sorted_axes.begin(), sorted_axes.end(), static_cast<int64_t>(i)) != sorted_axes.end();
        if (is_axis) {
            inner_dims.push_back(i);
        } else {
            outer_dims.push_back(i);
        }
    }

    auto expansion = context.replacement_requires_access_nodes(
        {passes::LibNodeExpander::InputUse::IndirectReadWrite, passes::LibNodeExpander::InputUse::IndirectRead}
    );
    if (!expansion) {
        return context.unable();
    }

    auto& seq = expansion->replace_with_sequence();
    auto& builder = expansion->builder();

    types::Scalar element_type(this->primitive_type(dataflow));
    const auto& in_type = in_edge->base_type();
    const auto& out_type = out_edge->base_type();

    // Declare a fresh per-outer-iteration scalar temporary.
    auto scalar = [&](const std::string& base) {
        std::string name = builder.find_new_name(base);
        builder.add_container(name, element_type);
        return name;
    };

    // Outer parallel loop nest over the non-reduced dimensions
    std::map<size_t, symbolic::Expression> loop_vars;
    structured_control_flow::Sequence* outer_scope = &seq;
    for (size_t dim_idx : outer_dims) {
        std::string indvar_str = builder.find_new_name("_i");
        auto& limit = shape_[dim_idx];
        builder.add_container(indvar_str, types::Scalar(types::get_primitive_type_to_hold_upper_bound(limit)));
        auto indvar = symbolic::symbol(indvar_str);
        auto& map = builder.add_map(
            *outer_scope,
            indvar,
            symbolic::Lt(indvar, limit),
            symbolic::zero(),
            symbolic::add(indvar, symbolic::one()),
            structured_control_flow::ScheduleType_Sequential::create(),
            this->debug_info()
        );
        outer_scope = &map.root();
        loop_vars[dim_idx] = indvar;
    }

    auto build_inner_nest = [&](structured_control_flow::Sequence& start) {
        std::map<size_t, symbolic::Expression> vars = loop_vars;
        structured_control_flow::Sequence* scope = &start;
        for (size_t j = 0; j < inner_dims.size(); ++j) {
            size_t dim_idx = inner_dims[j];
            std::string indvar_str = builder.find_new_name("_k");
            auto& limit = shape_[dim_idx];
            builder.add_container(indvar_str, types::Scalar(types::get_primitive_type_to_hold_upper_bound(limit)));
            auto indvar = symbolic::symbol(indvar_str);
            bool innermost = (j + 1 == inner_dims.size());
            structured_control_flow::StructuredLoop* loop;
            loop = &builder.add_for(
                *scope,
                indvar,
                symbolic::Lt(indvar, limit),
                symbolic::zero(),
                symbolic::add(indvar, symbolic::one()),
                this->debug_info()
            );
            scope = &loop->root();
            vars[dim_idx] = indvar;
        }
        data_flow::Subset index;
        for (size_t i = 0; i < shape_.size(); ++i) {
            index.push_back(vars.at(i));
        }
        return std::make_pair(scope, index);
    };

    if (softmax_use_online) {
        std::string m_name = scalar("_softmax_max");
        std::string d_name = scalar("_softmax_denom");
        std::string mold_name = scalar("_softmax_maxold");
        std::string corr_name = scalar("_softmax_corr");
        std::string e_name = scalar("_softmax_exp");
        std::string dm_name = scalar("_softmax_dmax");
        std::string xm_name = scalar("_softmax_xsub");
        std::string en_name = scalar("_softmax_enorm");
        std::string xn_name = scalar("_softmax_xnorm");

        // 1. Initialize running max m = -INFINITY and denominator d = 0
        {
            auto& blk = builder.add_block(*outer_scope, {}, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(blk, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, this->debug_info());
            auto& cst = builder.add_constant(blk, "-INFINITY", element_type, this->debug_info());
            auto& m_write = builder.add_access(blk, m_name, this->debug_info());
            builder.add_computational_memlet(blk, cst, tasklet, "_in", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, tasklet, "_out", m_write, {}, element_type, this->debug_info());
        }
        {
            auto& blk = builder.add_block(*outer_scope, {}, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(blk, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, this->debug_info());
            auto& cst = builder.add_constant(blk, "0.0", element_type, this->debug_info());
            auto& d_write = builder.add_access(blk, d_name, this->debug_info());
            builder.add_computational_memlet(blk, cst, tasklet, "_in", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, tasklet, "_out", d_write, {}, element_type, this->debug_info());
        }

        // 2. Online statistics: one streaming pass computing running max m and
        //    denominator d = sum(exp(X - m)) via the running-max correction
        //    d <- d * exp(m_old - m_new) + exp(x - m_new).
        {
            auto [scope, index] = build_inner_nest(*outer_scope);

            // m_old = m ; m = fmax(m_old, X[idx])   (in-place running-max update)
            {
                auto& blk = builder.add_block(*scope, {}, this->debug_info());
                auto& m_read = builder.add_access(blk, m_name, this->debug_info());
                auto& snap =
                    builder.add_tasklet(blk, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, this->debug_info());
                auto& mold_write = builder.add_access(blk, mold_name, this->debug_info());
                builder.add_computational_memlet(blk, m_read, snap, "_in", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, snap, "_out", mold_write, {}, element_type, this->debug_info());
                auto& fmax_node = builder.add_library_node<
                    cmath::CMathNode>(blk, this->debug_info(), cmath::CMathFunction::fmax, element_type.primitive_type());
                auto& x_access = expansion->add_indirect_read_access(blk, X_INPUT_IDX);
                auto& m_write = builder.add_access(blk, m_name, this->debug_info());
                builder
                    .add_computational_memlet(blk, mold_write, fmax_node, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, x_access, fmax_node, "_in2", index, in_type, this->debug_info());
                builder.add_computational_memlet(blk, fmax_node, "_out", m_write, {}, element_type, this->debug_info());

                // corr = exp(mold - m)
                auto& sub1 =
                    builder
                        .add_tasklet(blk, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"}, this->debug_info());
                auto& dm_node = builder.add_access(blk, dm_name, this->debug_info());
                builder.add_computational_memlet(blk, mold_write, sub1, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, m_write, sub1, "_in2", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, sub1, "_out", dm_node, {}, element_type, this->debug_info());
                auto& exp_node1 = builder.add_library_node<
                    cmath::CMathNode>(blk, this->debug_info(), cmath::CMathFunction::exp, element_type.primitive_type());
                auto& corr_write = builder.add_access(blk, corr_name, this->debug_info());
                builder.add_computational_memlet(blk, dm_node, exp_node1, "_in1", {}, element_type, this->debug_info());
                builder
                    .add_computational_memlet(blk, exp_node1, "_out", corr_write, {}, element_type, this->debug_info());

                // e = exp(X[idx] - m)
                auto& sub2 =
                    builder
                        .add_tasklet(blk, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"}, this->debug_info());
                auto& xm_node = builder.add_access(blk, xm_name, this->debug_info());
                builder.add_computational_memlet(blk, x_access, sub2, "_in1", index, in_type, this->debug_info());
                builder.add_computational_memlet(blk, m_write, sub2, "_in2", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, sub2, "_out", xm_node, {}, element_type, this->debug_info());
                auto& exp_node2 = builder.add_library_node<
                    cmath::CMathNode>(blk, this->debug_info(), cmath::CMathFunction::exp, element_type.primitive_type());
                auto& e_write = builder.add_access(blk, e_name, this->debug_info());
                builder.add_computational_memlet(blk, xm_node, exp_node2, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, exp_node2, "_out", e_write, {}, element_type, this->debug_info());

                // d = d * corr + e
                auto& fma = builder.add_tasklet(
                    blk, data_flow::TaskletCode::fp_fma, {"_out"}, {"_in1", "_in2", "_in3"}, this->debug_info()
                );
                auto& d_read = builder.add_access(blk, d_name, this->debug_info());
                auto& d_write = builder.add_access(blk, d_name, this->debug_info());
                builder.add_computational_memlet(blk, d_read, fma, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, corr_write, fma, "_in2", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, e_write, fma, "_in3", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, fma, "_out", d_write, {}, element_type, this->debug_info());
            }
        }

        // 3. Normalize: Y[idx] = exp(X[idx] - m) / d over the reduced dimensions
        {
            auto [scope, index] = build_inner_nest(*outer_scope);

            // e = exp(X[idx] - m)
            {
                auto& blk = builder.add_block(*scope, {}, this->debug_info());
                auto& sub =
                    builder
                        .add_tasklet(blk, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"}, this->debug_info());
                auto& x_access = expansion->add_indirect_read_access(blk, X_INPUT_IDX);
                auto& m_read = builder.add_access(blk, m_name, this->debug_info());
                auto& xn_node = builder.add_access(blk, xn_name, this->debug_info());
                builder.add_computational_memlet(blk, x_access, sub, "_in1", index, in_type, this->debug_info());
                builder.add_computational_memlet(blk, m_read, sub, "_in2", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, sub, "_out", xn_node, {}, element_type, this->debug_info());
                auto& exp_node = builder.add_library_node<
                    cmath::CMathNode>(blk, this->debug_info(), cmath::CMathFunction::exp, element_type.primitive_type());
                auto& e_write = builder.add_access(blk, en_name, this->debug_info());
                builder.add_computational_memlet(blk, xn_node, exp_node, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, exp_node, "_out", e_write, {}, element_type, this->debug_info());
            }

            // Y[idx] = e / d
            {
                auto& blk = builder.add_block(*scope, {}, this->debug_info());
                auto& div =
                    builder
                        .add_tasklet(blk, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"}, this->debug_info());
                auto& e_read = builder.add_access(blk, en_name, this->debug_info());
                auto& d_read = builder.add_access(blk, d_name, this->debug_info());
                auto& y_write = expansion->add_indirect_write_access(blk, RESULT_PTR_IDX);
                builder.add_computational_memlet(blk, e_read, div, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, d_read, div, "_in2", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, div, "_out", y_write, index, out_type, this->debug_info());
            }
        }
    } else {
        // ===== Three-pass: max reduce; exp+sum reduce; normalize map =====
        // With the row cached on-chip this avoids the online correction (one fewer exp
        // per element and a plain Max/Add cooperative reduction instead of the coupled
        // monoid).
        std::string m_name = scalar("_softmax_max");
        std::string d_name = scalar("_softmax_denom");
        std::string ts_name = scalar("_softmax_tsum");
        std::string es_name = scalar("_softmax_esum");
        std::string tn_name = scalar("_softmax_tnorm");
        std::string en_name = scalar("_softmax_enorm");

        // Pass 1: m = max over X
        {
            auto& blk = builder.add_block(*outer_scope, {}, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(blk, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, this->debug_info());
            auto& cst = builder.add_constant(blk, "-INFINITY", element_type, this->debug_info());
            auto& m_write = builder.add_access(blk, m_name, this->debug_info());
            builder.add_computational_memlet(blk, cst, tasklet, "_in", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, tasklet, "_out", m_write, {}, element_type, this->debug_info());
        }
        {
            auto [scope, index] = build_inner_nest(*outer_scope);
            auto& blk = builder.add_block(*scope, {}, this->debug_info());
            auto& fmax_node = builder.add_library_node<
                cmath::CMathNode>(blk, this->debug_info(), cmath::CMathFunction::fmax, element_type.primitive_type());
            auto& m_read = builder.add_access(blk, m_name, this->debug_info());
            auto& x_access = expansion->add_indirect_read_access(blk, X_INPUT_IDX);
            auto& m_write = builder.add_access(blk, m_name, this->debug_info());
            builder.add_computational_memlet(blk, m_read, fmax_node, "_in1", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, x_access, fmax_node, "_in2", index, in_type, this->debug_info());
            builder.add_computational_memlet(blk, fmax_node, "_out", m_write, {}, element_type, this->debug_info());
        }

        // Pass 2: d = sum(exp(X - m))
        {
            auto& blk = builder.add_block(*outer_scope, {}, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(blk, data_flow::TaskletCode::assign, {"_out"}, {"_in"}, this->debug_info());
            auto& cst = builder.add_constant(blk, "0.0", element_type, this->debug_info());
            auto& d_write = builder.add_access(blk, d_name, this->debug_info());
            builder.add_computational_memlet(blk, cst, tasklet, "_in", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, tasklet, "_out", d_write, {}, element_type, this->debug_info());
        }
        {
            auto [scope, index] = build_inner_nest(*outer_scope);
            auto& blk = builder.add_block(*scope, {}, this->debug_info());
            auto& sub =
                builder.add_tasklet(blk, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"}, this->debug_info());
            auto& x_access = expansion->add_indirect_read_access(blk, X_INPUT_IDX);
            auto& m_read = builder.add_access(blk, m_name, this->debug_info());
            auto& t_node = builder.add_access(blk, ts_name, this->debug_info());
            builder.add_computational_memlet(blk, x_access, sub, "_in1", index, in_type, this->debug_info());
            builder.add_computational_memlet(blk, m_read, sub, "_in2", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, sub, "_out", t_node, {}, element_type, this->debug_info());
            auto& exp_node = builder.add_library_node<
                cmath::CMathNode>(blk, this->debug_info(), cmath::CMathFunction::exp, element_type.primitive_type());
            auto& e_node = builder.add_access(blk, es_name, this->debug_info());
            builder.add_computational_memlet(blk, t_node, exp_node, "_in1", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, exp_node, "_out", e_node, {}, element_type, this->debug_info());
            auto& add =
                builder.add_tasklet(blk, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"}, this->debug_info());
            auto& d_read = builder.add_access(blk, d_name, this->debug_info());
            auto& d_write = builder.add_access(blk, d_name, this->debug_info());
            builder.add_computational_memlet(blk, d_read, add, "_in1", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, e_node, add, "_in2", {}, element_type, this->debug_info());
            builder.add_computational_memlet(blk, add, "_out", d_write, {}, element_type, this->debug_info());
        }

        // Pass 3: Y[idx] = exp(X[idx] - m) / d
        {
            auto [scope, index] = build_inner_nest(*outer_scope);
            {
                auto& blk = builder.add_block(*scope, {}, this->debug_info());
                auto& sub =
                    builder
                        .add_tasklet(blk, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"}, this->debug_info());
                auto& x_access = expansion->add_indirect_read_access(blk, X_INPUT_IDX);
                auto& m_read = builder.add_access(blk, m_name, this->debug_info());
                auto& t_node = builder.add_access(blk, tn_name, this->debug_info());
                builder.add_computational_memlet(blk, x_access, sub, "_in1", index, in_type, this->debug_info());
                builder.add_computational_memlet(blk, m_read, sub, "_in2", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, sub, "_out", t_node, {}, element_type, this->debug_info());
                auto& exp_node = builder.add_library_node<
                    cmath::CMathNode>(blk, this->debug_info(), cmath::CMathFunction::exp, element_type.primitive_type());
                auto& e_node = builder.add_access(blk, en_name, this->debug_info());
                builder.add_computational_memlet(blk, t_node, exp_node, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, exp_node, "_out", e_node, {}, element_type, this->debug_info());
            }
            {
                auto& blk = builder.add_block(*scope, {}, this->debug_info());
                auto& div =
                    builder
                        .add_tasklet(blk, data_flow::TaskletCode::fp_div, {"_out"}, {"_in1", "_in2"}, this->debug_info());
                auto& e_read = builder.add_access(blk, en_name, this->debug_info());
                auto& d_read = builder.add_access(blk, d_name, this->debug_info());
                auto& y_write = expansion->add_indirect_write_access(blk, RESULT_PTR_IDX);
                builder.add_computational_memlet(blk, e_read, div, "_in1", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, d_read, div, "_in2", {}, element_type, this->debug_info());
                builder.add_computational_memlet(blk, div, "_out", y_write, index, out_type, this->debug_info());
            }
        }
    }

    return expansion->successfully_expanded();
}

bool SoftmaxNode::expand_reduction(
    passes::LibNodeExpander::AccessNodeExpand& expansion,
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& body,
    const types::Tensor& input_type,
    const types::Tensor& output_type,
    const data_flow::Subset& input_subset,
    const data_flow::Subset& output_subset
) {
    throw std::runtime_error("StdNode::expand_reduction should not be called");
}

std::unique_ptr<data_flow::DataFlowNode> SoftmaxNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<
        data_flow::DataFlowNode>(new SoftmaxNode(element_id, this->debug_info(), vertex, parent, this->shape_, this->axes_)
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
