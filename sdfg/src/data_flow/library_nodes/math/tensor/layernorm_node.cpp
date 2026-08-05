#include "sdfg/data_flow/library_nodes/math/tensor/layernorm_node.h"

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/cmath/cmath_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_ops/mul_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_expansion_utils.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/utils.h"

namespace sdfg::math::tensor {

static std::vector<std::string> layernorm_inputs(bool affine, bool has_bias) {
    std::vector<std::string> inputs;
    inputs.push_back("X");
    if (affine) {
        inputs.push_back("Gamma");
    }
    if (has_bias) {
        inputs.push_back("Beta");
    }
    inputs.push_back("epsilon");
    inputs.push_back("Y_out");
    return inputs;
}

LayerNormNode::LayerNormNode(
    size_t element_id,
    const DebugInfo& debug_info,
    graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    TensorLayout layout,
    QuantizationType quantization,
    size_t num_normalized_dims,
    bool affine,
    bool has_bias,
    data_flow::ImplementationType impl_type
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_LayerNorm,
          {},
          layernorm_inputs(affine, has_bias),
          std::move(impl_type)
      ),
      layout_(std::move(layout)), quantization_(quantization), num_normalized_dims_(num_normalized_dims),
      affine_(affine), has_bias_(has_bias) {}

symbolic::SymbolSet LayerNormNode::symbols() const {
    symbolic::SymbolSet syms;
    layout_.collect_symbols(syms);
    return syms;
}

types::PrimitiveType LayerNormNode::quantization() const { return quantization_; }

void LayerNormNode::set_quantization(const types::PrimitiveType quant) { quantization_ = quant; }

void LayerNormNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_.replace_symbols(old_expression, new_expression);
}

void LayerNormNode::replace(const symbolic::ExpressionMapping& replacements) { layout_.replace_symbols(replacements); }

std::unique_ptr<data_flow::DataFlowNode> LayerNormNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new LayerNormNode(
        element_id,
        debug_info(),
        vertex,
        parent,
        this->layout_,
        this->quantization_,
        this->num_normalized_dims_,
        this->affine_,
        this->has_bias_,
        this->implementation_type_
    ));
}

std::string LayerNormNode::toStr() const { return "LayerNorm(" + layout_.toStr() + ")"; }

passes::LibNodeExpander::ExpandOutcome LayerNormNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    auto* x_edge = dataflow.in_edge_for_connector(*this, "X");
    if (!x_edge) {
        return context.unable();
    }
    auto& in_type = x_edge->base_type();
    types::PrimitiveType prim = in_type.primitive_type();
    types::Scalar element_type(prim);
    types::Pointer pointer_type(element_type);

    const auto& shape = layout_.shape();
    const int n = static_cast<int>(shape.size());
    const int n_norm = static_cast<int>(num_normalized_dims_);
    const int n_lead = n - n_norm;

    if (n_norm <= 0 || n_lead < 0) {
        return context.unable();
    }

    std::vector<symbolic::Expression> full_shape(shape.begin(), shape.end());
    std::vector<symbolic::Expression> leading_shape(shape.begin(), shape.begin() + n_lead);
    std::vector<symbolic::Expression> trailing_shape(shape.begin() + n_lead, shape.end());

    types::Tensor full_tensor(prim, full_shape);
    types::Tensor leading_tensor(prim, leading_shape);

    // Reduce axes = the trailing (normalized) dimensions.
    std::vector<int64_t> axes;
    for (int i = n_lead; i < n; ++i) {
        axes.push_back(i);
    }

    // Connector indices (must match the order in layernorm_inputs()).
    int idx = 0;
    const int X_IDX = idx++;
    const int GAMMA_IDX = affine_ ? idx++ : -1;
    const int BETA_IDX = has_bias_ ? idx++ : -1;
    const int EPS_IDX = idx++;
    const int YOUT_IDX = idx++;

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> dirs;
    dirs.push_back(Use::IndirectRead); // X
    if (affine_) {
        dirs.push_back(Use::IndirectRead); // Gamma
    }
    if (has_bias_) {
        dirs.push_back(Use::IndirectRead); // Beta
    }
    dirs.push_back(Use::Scalar); // epsilon
    dirs.push_back(Use::IndirectWrite); // Y_out

    auto standalone = context.replacement_requires_access_nodes(dirs);
    if (!standalone) {
        return context.unable();
    }

    auto& seq = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    // Allocate a temporary buffer of the given shape and return its container name.
    auto make_buffer = [&](const std::vector<symbolic::Expression>& bshape, const std::string& prefix) -> std::string {
        std::string name = builder.find_new_name(prefix);
        if (bshape.empty()) {
            builder.add_container(name, element_type);
        } else {
            builder.add_container(name, pointer_type);
            symbolic::Expression bytes = types::get_type_size(element_type, false);
            for (auto& d : bshape) {
                bytes = symbolic::mul(d, bytes);
            }
            auto& alloc_block = builder.add_block(seq, debug_info());
            auto& acc = builder.add_access(alloc_block, name, debug_info());
            auto& malloc_node = builder.add_library_node<stdlib::MallocNode>(alloc_block, debug_info(), bytes);
            builder.add_computational_memlet(alloc_block, malloc_node, "_ret", acc, {}, pointer_type, debug_info());
        }
        return name;
    };

    std::string x2_name = make_buffer(full_shape, "_ln_x2");
    std::string sumx_name = make_buffer(leading_shape, "_ln_sumx");
    std::string sumx2_name = make_buffer(leading_shape, "_ln_sumx2");
    std::string mean_name = make_buffer(leading_shape, "_ln_mean");
    std::string rstd_name = make_buffer(leading_shape, "_ln_rstd");

    // count = product of the normalized (trailing) dimension sizes.
    // Computed symbolically and stored in a scalar so it also works for dynamic shapes.
    auto count_name = builder.find_new_name("_ln_count");
    builder.add_container(count_name, types::Scalar(types::PrimitiveType::Int64));
    {
        symbolic::Expression count_expr = symbolic::one();
        for (int i = n_lead; i < n; ++i) {
            count_expr = symbolic::mul(count_expr, shape.at(i));
        }
        builder.add_assignments(seq, {{symbolic::symbol(count_name), count_expr}}, debug_info());
    }

    // x2 = x * x  (elementwise, full shape)
    {
        auto& b = builder.add_block(seq, debug_info());
        auto& x_in = standalone->add_scalar_input_access(b, X_IDX);
        auto& x2_acc = builder.add_access(b, x2_name, debug_info());
        auto& mul = builder.add_library_node<MulNode>(b, debug_info(), full_shape);
        builder.add_computational_memlet(b, x_in, mul, "A", {}, full_tensor, debug_info());
        builder.add_computational_memlet(b, x_in, mul, "B", {}, full_tensor, debug_info());
        builder.add_computational_memlet(b, x2_acc, mul, "C", {}, full_tensor, debug_info());
    }

    // sum_x = Sum(x) over normalized axes.
    // NOTE: a plain Sum (not Mean) is used deliberately. MeanNode finalizes with an internal
    // divide-by-count that can get duplicated across kernels during GPU map fusion (dividing
    // E[x^2] twice), which corrupts the variance. We do the /count division ourselves below,
    // exactly once.
    {
        auto& b = builder.add_block(seq, debug_info());
        auto& x_in = standalone->add_scalar_input_access(b, X_IDX);
        auto& sumx_acc = builder.add_access(b, sumx_name, debug_info());
        auto& sum_node = builder.add_library_node<SumNode>(b, debug_info(), full_shape, axes, false);
        builder.add_computational_memlet(b, x_in, sum_node, "X", {}, full_tensor, debug_info());
        builder.add_computational_memlet(b, sumx_acc, sum_node, "Y", {}, leading_tensor, debug_info());
    }

    // sum_x2 = Sum(x^2) over normalized axes
    {
        auto& b = builder.add_block(seq, debug_info());
        auto& x2_in = builder.add_access(b, x2_name, debug_info());
        auto& sumx2_acc = builder.add_access(b, sumx2_name, debug_info());
        auto& sum_node = builder.add_library_node<SumNode>(b, debug_info(), full_shape, axes, false);
        builder.add_computational_memlet(b, x2_in, sum_node, "X", {}, full_tensor, debug_info());
        builder.add_computational_memlet(b, sumx2_acc, sum_node, "Y", {}, leading_tensor, debug_info());
    }

    // mean = sum_x / count ; rstd = 1 / sqrt(sum_x2/count - mean*mean + eps)  (per leading row)
    {
        auto lead_maps = create_maps(builder, leading_shape, seq);
        structured_control_flow::Sequence& lead_scope = leading_shape.empty() ? seq : lead_maps.back().seq;
        std::vector<symbolic::Expression> lead_subset;
        for (auto& m : lead_maps) {
            lead_subset.push_back(m.indvar);
        }

        auto& b = builder.add_block(lead_scope, debug_info());
        std::string prefix = "_ln_rstd_tmp";
        int t = 0;

        auto& sumx_in = builder.add_access(b, sumx_name, debug_info());
        auto& sumx2_in = builder.add_access(b, sumx2_name, debug_info());
        auto& eps_in = standalone->add_scalar_input_access(b, EPS_IDX);
        auto& count_a = builder.add_access(b, count_name, debug_info());
        auto& count_b = builder.add_access(b, count_name, debug_info());

        // mean = sum_x / count  (kept as a scalar temp and also written to the mean buffer)
        auto& mean_op = builder.add_tasklet(b, data_flow::fp_div, "_out", {"s", "c"}, debug_info());
        builder.add_computational_memlet(b, sumx_in, mean_op, "s", lead_subset, leading_tensor, debug_info());
        builder.add_computational_memlet(b, count_a, mean_op, "c", {}, element_type, debug_info());
        auto mean_s_name = create_temp_var(builder, prefix, t++, element_type);
        auto& mean_s = builder.add_access(b, mean_s_name, debug_info());
        builder.add_computational_memlet(b, mean_op, "_out", mean_s, {}, element_type, debug_info());

        auto& mean_store_op = builder.add_tasklet(b, data_flow::assign, "_out", {"_in"}, debug_info());
        builder.add_computational_memlet(b, mean_s, mean_store_op, "_in", {}, element_type, debug_info());
        auto& mean_acc = builder.add_access(b, mean_name, debug_info());
        builder.add_computational_memlet(b, mean_store_op, "_out", mean_acc, lead_subset, leading_tensor, debug_info());

        // sq = mean * mean
        auto& sq_op = builder.add_tasklet(b, data_flow::fp_mul, "_out", {"a", "b"}, debug_info());
        builder.add_computational_memlet(b, mean_s, sq_op, "a", {}, element_type, debug_info());
        builder.add_computational_memlet(b, mean_s, sq_op, "b", {}, element_type, debug_info());
        auto sq_name = create_temp_var(builder, prefix, t++, element_type);
        auto& sq_acc = builder.add_access(b, sq_name, debug_info());
        builder.add_computational_memlet(b, sq_op, "_out", sq_acc, {}, element_type, debug_info());

        // ex2 = sum_x2 / count
        auto& ex2_op = builder.add_tasklet(b, data_flow::fp_div, "_out", {"s", "c"}, debug_info());
        builder.add_computational_memlet(b, sumx2_in, ex2_op, "s", lead_subset, leading_tensor, debug_info());
        builder.add_computational_memlet(b, count_b, ex2_op, "c", {}, element_type, debug_info());
        auto ex2_name = create_temp_var(builder, prefix, t++, element_type);
        auto& ex2_acc = builder.add_access(b, ex2_name, debug_info());
        builder.add_computational_memlet(b, ex2_op, "_out", ex2_acc, {}, element_type, debug_info());

        // var = ex2 - sq
        auto& var_op = builder.add_tasklet(b, data_flow::fp_sub, "_out", {"x", "y"}, debug_info());
        builder.add_computational_memlet(b, ex2_acc, var_op, "x", {}, element_type, debug_info());
        builder.add_computational_memlet(b, sq_acc, var_op, "y", {}, element_type, debug_info());
        auto var_name = create_temp_var(builder, prefix, t++, element_type);
        auto& var_acc = builder.add_access(b, var_name, debug_info());
        builder.add_computational_memlet(b, var_op, "_out", var_acc, {}, element_type, debug_info());

        // ve = var + eps
        auto& ve_op = builder.add_tasklet(b, data_flow::fp_add, "_out", {"v", "e"}, debug_info());
        builder.add_computational_memlet(b, var_acc, ve_op, "v", {}, element_type, debug_info());
        builder.add_computational_memlet(b, eps_in, ve_op, "e", {}, element_type, debug_info());
        auto ve_name = create_temp_var(builder, prefix, t++, element_type);
        auto& ve_acc = builder.add_access(b, ve_name, debug_info());
        builder.add_computational_memlet(b, ve_op, "_out", ve_acc, {}, element_type, debug_info());

        // s = sqrt(ve)
        auto& sqrt_op = builder.add_library_node<cmath::CMathNode>(b, debug_info(), cmath::CMathFunction::sqrt, prim);
        builder.add_computational_memlet(b, ve_acc, sqrt_op, "_in1", {}, element_type, debug_info());
        auto s_name = create_temp_var(builder, prefix, t++, element_type);
        auto& s_acc = builder.add_access(b, s_name, debug_info());
        builder.add_computational_memlet(b, sqrt_op, "_out", s_acc, {}, element_type, debug_info());

        // rstd = 1 / s
        auto& one_c = builder.add_constant(b, "1.0", element_type, debug_info());
        auto& inv_op = builder.add_tasklet(b, data_flow::fp_div, "_out", {"o", "s"}, debug_info());
        builder.add_computational_memlet(b, one_c, inv_op, "o", {}, element_type, debug_info());
        builder.add_computational_memlet(b, s_acc, inv_op, "s", {}, element_type, debug_info());
        auto& rstd_acc = builder.add_access(b, rstd_name, debug_info());
        builder.add_computational_memlet(b, inv_op, "_out", rstd_acc, lead_subset, leading_tensor, debug_info());
    }

    // Y = (x - mean) * rstd [* Gamma] [+ Beta]  (per element)
    // Iterate over a flattened [lead_flat, trail_flat] index space and address every
    // operand with explicit linear indices built from the actual map indvars. This avoids
    // a multi-dimensional trailing (suffix) subset under a collapsed map: the index
    // simplifier mishandles that case (it drops the modulo on the leading dimension),
    // which would index Gamma/Beta out of bounds for every row past the first.
    {
        symbolic::Expression lead_flat = symbolic::one();
        for (auto& d : leading_shape) {
            lead_flat = symbolic::mul(lead_flat, d);
        }
        symbolic::Expression trail_flat = symbolic::one();
        for (auto& d : trailing_shape) {
            trail_flat = symbolic::mul(trail_flat, d);
        }
        symbolic::Expression total = symbolic::mul(lead_flat, trail_flat);

        const bool has_leading = !leading_shape.empty();

        std::vector<symbolic::Expression> map_shape;
        if (has_leading) {
            map_shape.push_back(lead_flat);
        }
        map_shape.push_back(trail_flat);

        auto maps = create_maps(builder, map_shape, seq);
        structured_control_flow::Sequence& scope = maps.back().seq;

        symbolic::Expression l_idx = symbolic::zero();
        if (has_leading) {
            l_idx = maps.at(0).indvar;
        }
        symbolic::Expression t_idx = maps.back().indvar;
        symbolic::Expression lin = symbolic::add(symbolic::mul(l_idx, trail_flat), t_idx);

        types::Tensor total_tensor(prim, {total});
        types::Tensor lead_flat_tensor(prim, {lead_flat});
        types::Tensor trail_flat_tensor(prim, {trail_flat});

        std::vector<symbolic::Expression> lin_subset{lin};
        std::vector<symbolic::Expression> l_subset;
        if (has_leading) {
            l_subset.push_back(l_idx);
        }
        std::vector<symbolic::Expression> t_subset{t_idx};

        auto& scope_block = builder.add_block(scope, debug_info());
        std::string prefix = "_ln_norm_tmp";
        int t = 0;

        // Reads mean/rstd either per leading row (contiguous [lead_flat] view of the
        // buffer written above) or as a scalar when normalizing over all dimensions.
        auto read_reduced = [&](data_flow::AccessNode& src, data_flow::Tasklet& op, const std::string& conn) {
            if (has_leading) {
                builder.add_computational_memlet(scope_block, src, op, conn, l_subset, lead_flat_tensor, debug_info());
            } else {
                builder.add_computational_memlet(scope_block, src, op, conn, {}, element_type, debug_info());
            }
        };

        auto& x_in = standalone->add_indirect_read_access(scope_block, X_IDX);
        auto& mean_in = builder.add_access(scope_block, mean_name, debug_info());
        auto& rstd_in = builder.add_access(scope_block, rstd_name, debug_info());

        // c = x - mean
        auto& sub_op = builder.add_tasklet(scope_block, data_flow::fp_sub, "_out", {"x", "m"}, debug_info());
        builder.add_computational_memlet(scope_block, x_in, sub_op, "x", lin_subset, total_tensor, debug_info());
        read_reduced(mean_in, sub_op, "m");
        auto c_name = create_temp_var(builder, prefix, t++, element_type);
        auto& c_acc = builder.add_access(scope_block, c_name, debug_info());
        builder.add_computational_memlet(scope_block, sub_op, "_out", c_acc, {}, element_type, debug_info());

        const bool has_scale_or_bias = affine_ || has_bias_;
        data_flow::AccessNode* cur = nullptr;

        // n = c * rstd
        {
            auto& mul_op = builder.add_tasklet(scope_block, data_flow::fp_mul, "_out", {"c", "r"}, debug_info());
            builder.add_computational_memlet(scope_block, c_acc, mul_op, "c", {}, element_type, debug_info());
            read_reduced(rstd_in, mul_op, "r");
            if (has_scale_or_bias) {
                auto n_name = create_temp_var(builder, prefix, t++, element_type);
                auto& n_acc = builder.add_access(scope_block, n_name, debug_info());
                builder.add_computational_memlet(scope_block, mul_op, "_out", n_acc, {}, element_type, debug_info());
                cur = &n_acc;
            } else {
                auto& y_out = standalone->add_indirect_write_access(scope_block, YOUT_IDX);
                builder
                    .add_computational_memlet(scope_block, mul_op, "_out", y_out, lin_subset, total_tensor, debug_info());
            }
        }

        // * Gamma
        if (affine_) {
            const bool last = !has_bias_;
            auto& gamma_in = standalone->add_indirect_read_access(scope_block, GAMMA_IDX);
            auto& g_op = builder.add_tasklet(scope_block, data_flow::fp_mul, "_out", {"n", "g"}, debug_info());
            builder.add_computational_memlet(scope_block, *cur, g_op, "n", {}, element_type, debug_info());
            builder
                .add_computational_memlet(scope_block, gamma_in, g_op, "g", t_subset, trail_flat_tensor, debug_info());
            if (last) {
                auto& y_out = standalone->add_indirect_write_access(scope_block, YOUT_IDX);
                builder
                    .add_computational_memlet(scope_block, g_op, "_out", y_out, lin_subset, total_tensor, debug_info());
            } else {
                auto g_name = create_temp_var(builder, prefix, t++, element_type);
                auto& g_acc = builder.add_access(scope_block, g_name, debug_info());
                builder.add_computational_memlet(scope_block, g_op, "_out", g_acc, {}, element_type, debug_info());
                cur = &g_acc;
            }
        }

        // + Beta
        if (has_bias_) {
            auto& beta_in = standalone->add_indirect_read_access(scope_block, BETA_IDX);
            auto& add_op = builder.add_tasklet(scope_block, data_flow::fp_add, "_out", {"n", "b"}, debug_info());
            builder.add_computational_memlet(scope_block, *cur, add_op, "n", {}, element_type, debug_info());
            builder
                .add_computational_memlet(scope_block, beta_in, add_op, "b", t_subset, trail_flat_tensor, debug_info());
            auto& y_out = standalone->add_indirect_write_access(scope_block, YOUT_IDX);
            builder.add_computational_memlet(scope_block, add_op, "_out", y_out, lin_subset, total_tensor, debug_info());
        }
    }

    return standalone->successfully_expanded();
}

symbolic::Expression LayerNormNode::flop() const {
    // Rough estimate: a few flops per element for the two reduction passes plus the
    // normalization and optional affine transform.
    auto total = layout_.total_elements();
    return symbolic::mul(total, symbolic::integer(8));
}

data_flow::PointerAccessType LayerNormNode::pointer_access_type(int input_idx) const {
    int idx = 0;
    const int x_idx = idx++;
    const int gamma_idx = affine_ ? idx++ : -1;
    const int beta_idx = has_bias_ ? idx++ : -1;
    idx++; // epsilon (scalar)
    const int yout_idx = idx++;

    if (input_idx == x_idx || (affine_ && input_idx == gamma_idx) || (has_bias_ && input_idx == beta_idx)) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else if (input_idx == yout_idx) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

nlohmann::json LayerNormNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    auto& node = static_cast<const LayerNormNode&>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();

    node.layernorm_layout().serialize_to_json(j["layout"]);

    j["quant"] = node.quantization();
    j["num_normalized_dims"] = node.num_normalized_dims();
    j["affine"] = node.affine();
    j["has_bias"] = node.has_bias();

    return j;
}

data_flow::LibraryNode& LayerNormNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    auto layout = TensorLayout::deserialize_from_json(j.at("layout"));
    auto quant = j.at("quant").get<types::PrimitiveType>();
    auto num_normalized_dims = j.at("num_normalized_dims").get<size_t>();
    auto affine = j.at("affine").get<bool>();
    auto has_bias = j.at("has_bias").get<bool>();

    serializer::JSONSerializer serializer;
    auto deb_info = serializer.json_to_debug_info(j.at("debug_info"));

    return builder
        .add_library_node<LayerNormNode>(parent, deb_info, layout, quant, num_normalized_dims, affine, has_bias);
}

} // namespace sdfg::math::tensor
