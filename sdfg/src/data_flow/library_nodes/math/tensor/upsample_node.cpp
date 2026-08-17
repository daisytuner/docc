#include "sdfg/data_flow/library_nodes/math/tensor/upsample_node.h"

#include <iomanip>
#include <sstream>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace math {
namespace tensor {

UpsampleBilinear2DNode::UpsampleBilinear2DNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& input_shape,
    const std::vector<symbolic::Expression>& output_shape,
    bool align_corners,
    const std::vector<double>& scale_factors,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(element_id, debug_info, vertex, parent, LibraryNodeType_UpsampleBilinear2D, {}, {"Y", "X"}, impl_type),
      input_shape_(input_shape), output_shape_(output_shape), align_corners_(align_corners),
      scale_factors_(scale_factors) {}

void UpsampleBilinear2DNode::validate(const Function& function) const {
    TensorNode::validate(function);

    if (input_shape_.size() != 4) {
        throw InvalidSDFGException("UpsampleBilinear2DNode: input_shape must have rank 4 [N, C, H, W]");
    }
    if (output_shape_.size() != 4) {
        throw InvalidSDFGException("UpsampleBilinear2DNode: output_shape must have rank 4 [N, C, H, W]");
    }
    if (!scale_factors_.empty() && scale_factors_.size() != 2) {
        throw InvalidSDFGException("UpsampleBilinear2DNode: scale_factors must be empty or have 2 entries");
    }
}

symbolic::SymbolSet UpsampleBilinear2DNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : input_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    for (const auto& dim : output_shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

void UpsampleBilinear2DNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
    for (auto& dim : output_shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void UpsampleBilinear2DNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : input_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
    for (auto& dim : output_shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

passes::LibNodeExpander::ExpandOutcome UpsampleBilinear2DNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dataflow = this->get_parent();

    auto x_edge = dataflow.in_edge_for_connector(*this, "X");
    if (!x_edge) {
        return context.unable();
    }
    auto y_edge = dataflow.in_edge_for_connector(*this, "Y");
    if (!y_edge) {
        return context.unable();
    }

    types::Scalar double_type(types::PrimitiveType::Double);
    types::Scalar int_type(types::PrimitiveType::Int64);
    types::Scalar loop_type(types::PrimitiveType::UInt64);

    symbolic::Expression N = output_shape_[0];
    symbolic::Expression C = output_shape_[1];
    symbolic::Expression Hin = input_shape_[2];
    symbolic::Expression Win = input_shape_[3];
    symbolic::Expression Hout = output_shape_[2];
    symbolic::Expression Wout = output_shape_[3];

    using Use = passes::LibNodeExpander::InputUse;
    auto standalone = context.replacement_requires_access_nodes({Use::IndirectWrite, Use::IndirectRead});
    if (!standalone) {
        return context.unable();
    }

    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();
    auto dbg = block.debug_info();

    // Casts an existing scalar operand (loop variable, shape symbol or integer constant)
    // into a fresh Double container using a real tasklet. Symbols already are containers,
    // so they are read directly; integer constants feed in as a single literal.
    auto emit_double_operand = [&](structured_control_flow::Sequence& scope,
                                   const symbolic::Expression& expr,
                                   const std::string& hint) -> std::string {
        std::string name = builder.find_new_name(hint);
        builder.add_container(name, double_type);
        auto& blk = builder.add_block(scope, {}, dbg);
        auto& dst_acc = builder.add_access(blk, name, dbg);
        auto& tk = builder.add_tasklet(blk, data_flow::assign, "_out", {"_in"}, dbg);
        if (SymEngine::is_a<SymEngine::Symbol>(*expr)) {
            std::string src_name = expr->__str__();
            auto& src_acc = builder.add_access(blk, src_name, dbg);
            builder.add_computational_memlet(blk, src_acc, tk, "_in", {}, builder.subject().type(src_name), dbg);
        } else if (SymEngine::is_a<SymEngine::Integer>(*expr)) {
            auto& c_cst = builder.add_constant(blk, expr->__str__(), double_type, dbg);
            builder.add_computational_memlet(blk, c_cst, tk, "_in", {}, double_type, dbg);
        } else {
            throw InvalidSDFGException(
                "UpsampleBilinear2DNode: unsupported expression type for operand: " + expr->__str__()
            );
        }
        builder.add_computational_memlet(blk, tk, "_out", dst_acc, {}, double_type, dbg);
        return name;
    };

    // Copies (and casts) a scalar container into a fresh container of dst_type.
    auto emit_cast = [&](structured_control_flow::Sequence& scope,
                         const std::string& src_name,
                         const types::Scalar& src_type,
                         const types::Scalar& dst_type,
                         const std::string& hint) -> std::string {
        std::string name = builder.find_new_name(hint);
        builder.add_container(name, dst_type);
        auto& blk = builder.add_block(scope, {}, dbg);
        auto& src_acc = builder.add_access(blk, src_name, dbg);
        auto& dst_acc = builder.add_access(blk, name, dbg);
        auto& tk = builder.add_tasklet(blk, data_flow::assign, "_out", {"_in"}, dbg);
        builder.add_computational_memlet(blk, src_acc, tk, "_in", {}, src_type, dbg);
        builder.add_computational_memlet(blk, tk, "_out", dst_acc, {}, dst_type, dbg);
        return name;
    };

    // Applies a binary floating-point tasklet to two Double containers.
    auto emit_binop = [&](structured_control_flow::Sequence& scope,
                          data_flow::TaskletCode code,
                          const std::string& a,
                          const std::string& b,
                          const std::string& hint) -> std::string {
        std::string name = builder.find_new_name(hint);
        builder.add_container(name, double_type);
        auto& blk = builder.add_block(scope, {}, dbg);
        auto& a_acc = builder.add_access(blk, a, dbg);
        auto& b_acc = builder.add_access(blk, b, dbg);
        auto& r_acc = builder.add_access(blk, name, dbg);
        auto& tk = builder.add_tasklet(blk, code, "_out", {"_in1", "_in2"}, dbg);
        builder.add_computational_memlet(blk, a_acc, tk, "_in1", {}, double_type, dbg);
        builder.add_computational_memlet(blk, b_acc, tk, "_in2", {}, double_type, dbg);
        builder.add_computational_memlet(blk, tk, "_out", r_acc, {}, double_type, dbg);
        return name;
    };

    // Applies a binary floating-point tasklet between a Double container and a single literal.
    auto emit_binop_lit = [&](structured_control_flow::Sequence& scope,
                              data_flow::TaskletCode code,
                              const std::string& a,
                              const std::string& literal,
                              const std::string& hint) -> std::string {
        std::string name = builder.find_new_name(hint);
        builder.add_container(name, double_type);
        auto& blk = builder.add_block(scope, {}, dbg);
        auto& a_acc = builder.add_access(blk, a, dbg);
        auto& c_cst = builder.add_constant(blk, literal, double_type, dbg);
        auto& r_acc = builder.add_access(blk, name, dbg);
        auto& tk = builder.add_tasklet(blk, code, "_out", {"_in1", "_in2"}, dbg);
        builder.add_computational_memlet(blk, a_acc, tk, "_in1", {}, double_type, dbg);
        builder.add_computational_memlet(blk, c_cst, tk, "_in2", {}, double_type, dbg);
        builder.add_computational_memlet(blk, tk, "_out", r_acc, {}, double_type, dbg);
        return name;
    };
    auto emit_one_minus = [&](structured_control_flow::Sequence& scope, const std::string& lam, const std::string& hint
                          ) -> std::string {
        std::string name = builder.find_new_name(hint);
        builder.add_container(name, double_type);
        auto& blk = builder.add_block(scope, {}, dbg);
        auto& one_c = builder.add_constant(blk, "1.0", double_type, dbg);
        auto& lam_acc = builder.add_access(blk, lam, dbg);
        auto& r_acc = builder.add_access(blk, name, dbg);
        auto& tk = builder.add_tasklet(blk, data_flow::fp_sub, "_out", {"_in1", "_in2"}, dbg);
        builder.add_computational_memlet(blk, one_c, tk, "_in1", {}, double_type, dbg);
        builder.add_computational_memlet(blk, lam_acc, tk, "_in2", {}, double_type, dbg);
        builder.add_computational_memlet(blk, tk, "_out", r_acc, {}, double_type, dbg);
        return name;
    };

    // Reads X[subset] into a fresh Double container.
    auto emit_pixel = [&](structured_control_flow::Sequence& scope,
                          const data_flow::Subset& subset,
                          const std::string& hint) -> std::string {
        std::string name = builder.find_new_name(hint);
        builder.add_container(name, double_type);
        auto& blk = builder.add_block(scope, {}, dbg);
        auto& x_acc = standalone->add_indirect_read_access(blk, X_INPUT_IDX);
        auto& p_acc = builder.add_access(blk, name, dbg);
        auto& tk = builder.add_tasklet(blk, data_flow::assign, "_out", {"_in"}, dbg);
        builder.add_computational_memlet(blk, x_acc, tk, "_in", subset, x_edge->base_type(), dbg);
        builder.add_computational_memlet(blk, tk, "_out", p_acc, {}, double_type, dbg);
        return name;
    };

    struct Coord {
        symbolic::Expression i0;
        symbolic::Expression i1;
        std::string lam;
        std::string lam0;
    };

    // Computes the fractional source coordinate, its two integer neighbours and the
    // interpolation weights for a single spatial dimension (align_corners aware).
    // All arithmetic is performed with real tasklets; constants are single literals only.
    auto compute_coord = [&](structured_control_flow::Sequence& scope,
                             const symbolic::Expression& o,
                             const symbolic::Expression& In,
                             const symbolic::Expression& Out,
                             size_t d) -> Coord {
        // o as Double (loop variable is already a container).
        std::string o_d = emit_double_operand(scope, o, "_o_d");

        std::string src;
        if (align_corners_) {
            if (symbolic::eq(Out, symbolic::one())) {
                // scale would divide by zero; PyTorch maps every output pixel to source 0.
                src = emit_binop_lit(scope, data_flow::fp_mul, o_d, "0.0", "_src");
            } else {
                // scale = (In - 1) / (Out - 1); src = scale * o.
                std::string in_d = emit_double_operand(scope, In, "_in_d");
                std::string out_d = emit_double_operand(scope, Out, "_out_d");
                std::string in_m1 = emit_binop_lit(scope, data_flow::fp_sub, in_d, "1.0", "_inm1");
                std::string out_m1 = emit_binop_lit(scope, data_flow::fp_sub, out_d, "1.0", "_outm1");
                std::string scale = emit_binop(scope, data_flow::fp_div, in_m1, out_m1, "_scale");
                src = emit_binop(scope, data_flow::fp_mul, scale, o_d, "_src");
            }
        } else {
            // src = max(0, rscale * (o + 0.5) - 0.5).
            std::string o_plus = emit_binop_lit(scope, data_flow::fp_add, o_d, "0.5", "_oplus");
            std::string scaled;
            if (!scale_factors_.empty()) {
                // rscale is a compile-time reciprocal of the requested scale factor.
                std::ostringstream oss;
                oss << std::setprecision(17) << (1.0 / scale_factors_[d]);
                scaled = emit_binop_lit(scope, data_flow::fp_mul, o_plus, oss.str(), "_scaled");
            } else {
                std::string in_d = emit_double_operand(scope, In, "_in_d");
                std::string out_d = emit_double_operand(scope, Out, "_out_d");
                std::string rscale = emit_binop(scope, data_flow::fp_div, in_d, out_d, "_rscale");
                scaled = emit_binop(scope, data_flow::fp_mul, rscale, o_plus, "_scaled");
            }
            std::string src_raw = emit_binop_lit(scope, data_flow::fp_sub, scaled, "0.5", "_srcraw");
            // Clamp negatives to zero: mask = (src_raw > 0) yields 1.0/0.0; src = src_raw * mask.
            std::string mask = emit_binop_lit(scope, data_flow::fp_ogt, src_raw, "0.0", "_mask");
            src = emit_binop(scope, data_flow::fp_mul, src_raw, mask, "_src");
        }

        std::string i0n = emit_cast(scope, src, double_type, int_type, "_i0");
        std::string i0d = emit_cast(scope, i0n, int_type, double_type, "_i0d");
        std::string lam = emit_binop(scope, data_flow::fp_sub, src, i0d, "_lam");
        std::string lam0 = emit_one_minus(scope, lam, "_lam0");

        auto i0_sym = symbolic::symbol(i0n);
        auto i1_sym = symbolic::min(symbolic::add(i0_sym, symbolic::one()), symbolic::sub(In, symbolic::one()));
        return Coord{i0_sym, i1_sym, lam, lam0};
    };

    structured_control_flow::Sequence* scope = &new_sequence;

    // Map over batch dimension N.
    std::string n_str = builder.find_new_name("n");
    builder.add_container(n_str, loop_type);
    auto n_var = symbolic::symbol(n_str);
    auto& map_n = builder.add_map(
        *scope,
        n_var,
        symbolic::Lt(n_var, N),
        symbolic::zero(),
        symbolic::add(n_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        dbg
    );
    scope = &map_n.root();

    // Map over channel dimension C.
    std::string c_str = builder.find_new_name("c");
    builder.add_container(c_str, loop_type);
    auto c_var = symbolic::symbol(c_str);
    auto& map_c = builder.add_map(
        *scope,
        c_var,
        symbolic::Lt(c_var, C),
        symbolic::zero(),
        symbolic::add(c_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        dbg
    );
    scope = &map_c.root();

    // Map over output height.
    std::string oh_str = builder.find_new_name("oh");
    builder.add_container(oh_str, loop_type);
    auto oh_var = symbolic::symbol(oh_str);
    auto& map_oh = builder.add_map(
        *scope,
        oh_var,
        symbolic::Lt(oh_var, Hout),
        symbolic::zero(),
        symbolic::add(oh_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        dbg
    );
    auto& oh_scope = map_oh.root();

    // Height source coordinate (depends only on oh).
    Coord hc = compute_coord(oh_scope, oh_var, Hin, Hout, 0);

    // Map over output width.
    std::string ow_str = builder.find_new_name("ow");
    builder.add_container(ow_str, loop_type);
    auto ow_var = symbolic::symbol(ow_str);
    auto& map_ow = builder.add_map(
        oh_scope,
        ow_var,
        symbolic::Lt(ow_var, Wout),
        symbolic::zero(),
        symbolic::add(ow_var, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        dbg
    );
    auto& ow_scope = map_ow.root();

    // Width source coordinate.
    Coord wc = compute_coord(ow_scope, ow_var, Win, Wout, 1);

    // Gather the four contributing input pixels.
    data_flow::Subset s00 = {n_var, c_var, hc.i0, wc.i0};
    data_flow::Subset s01 = {n_var, c_var, hc.i0, wc.i1};
    data_flow::Subset s10 = {n_var, c_var, hc.i1, wc.i0};
    data_flow::Subset s11 = {n_var, c_var, hc.i1, wc.i1};

    std::string p00 = emit_pixel(ow_scope, s00, "_p00");
    std::string p01 = emit_pixel(ow_scope, s01, "_p01");
    std::string p10 = emit_pixel(ow_scope, s10, "_p10");
    std::string p11 = emit_pixel(ow_scope, s11, "_p11");

    // Interpolate along width: top = p00 * (1 - lam_w) + p01 * lam_w.
    std::string t0 = emit_binop(ow_scope, data_flow::fp_mul, p00, wc.lam0, "_t0");
    std::string t1 = emit_binop(ow_scope, data_flow::fp_mul, p01, wc.lam, "_t1");
    std::string top = emit_binop(ow_scope, data_flow::fp_add, t0, t1, "_top");

    // bot = p10 * (1 - lam_w) + p11 * lam_w.
    std::string b0 = emit_binop(ow_scope, data_flow::fp_mul, p10, wc.lam0, "_b0");
    std::string b1 = emit_binop(ow_scope, data_flow::fp_mul, p11, wc.lam, "_b1");
    std::string bot = emit_binop(ow_scope, data_flow::fp_add, b0, b1, "_bot");

    // Interpolate along height: out = top * (1 - lam_h) + bot * lam_h.
    std::string o0 = emit_binop(ow_scope, data_flow::fp_mul, top, hc.lam0, "_o0");
    std::string o1 = emit_binop(ow_scope, data_flow::fp_mul, bot, hc.lam, "_o1");
    std::string out_c = emit_binop(ow_scope, data_flow::fp_add, o0, o1, "_out");

    // Write result into Y[n, c, oh, ow].
    data_flow::Subset y_subset = {n_var, c_var, oh_var, ow_var};
    auto& wblk = builder.add_block(ow_scope, {}, dbg);
    auto& out_acc = builder.add_access(wblk, out_c, dbg);
    auto& y_acc = standalone->add_indirect_write_access(wblk, Y_OUTPUT_IDX);
    auto& wtk = builder.add_tasklet(wblk, data_flow::assign, "_out", {"_in"}, dbg);
    builder.add_computational_memlet(wblk, out_acc, wtk, "_in", {}, double_type, dbg);
    builder.add_computational_memlet(wblk, wtk, "_out", y_acc, y_subset, y_edge->base_type(), dbg);

    return standalone->successfully_expanded();
}

std::unique_ptr<data_flow::DataFlowNode> UpsampleBilinear2DNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new UpsampleBilinear2DNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        input_shape_,
        output_shape_,
        align_corners_,
        scale_factors_,
        implementation_type_
    ));
}

symbolic::Expression UpsampleBilinear2DNode::flop() const {
    // N * C * Hout * Wout output elements, each requiring a fixed number of
    // multiply/add operations for the separable bilinear interpolation.
    auto output_elems = symbolic::
        mul(symbolic::mul(output_shape_[0], output_shape_[1]), symbolic::mul(output_shape_[2], output_shape_[3]));
    return symbolic::mul(output_elems, symbolic::integer(16));
}

data_flow::PointerAccessType UpsampleBilinear2DNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_OUTPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == X_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

std::string UpsampleBilinear2DNode::toStr() const {
    std::stringstream ss;
    ss << "UpsampleBilinear2D(align_corners=" << (align_corners_ ? "true" : "false");
    if (!scale_factors_.empty()) {
        ss << ", scale_factors=[" << scale_factors_[0] << ", " << scale_factors_[1] << "]";
    }
    ss << ")";
    return ss.str();
}

nlohmann::json UpsampleBilinear2DNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const UpsampleBilinear2DNode& node = static_cast<const UpsampleBilinear2DNode&>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();

    j["input_shape"] = nlohmann::json::array();
    for (auto& dim : node.input_shape()) {
        j["input_shape"].push_back(serializer::JSONSerializer::expression(dim));
    }
    j["output_shape"] = nlohmann::json::array();
    for (auto& dim : node.output_shape()) {
        j["output_shape"].push_back(serializer::JSONSerializer::expression(dim));
    }

    j["align_corners"] = node.align_corners();

    j["scale_factors"] = nlohmann::json::array();
    for (auto factor : node.scale_factors()) {
        j["scale_factors"].push_back(factor);
    }

    return j;
}

data_flow::LibraryNode& UpsampleBilinear2DNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("input_shape"));
    assert(j.contains("output_shape"));
    assert(j.contains("align_corners"));
    assert(j.contains("scale_factors"));

    std::vector<symbolic::Expression> input_shape;
    for (const auto& dim : j["input_shape"]) {
        input_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }
    std::vector<symbolic::Expression> output_shape;
    for (const auto& dim : j["output_shape"]) {
        output_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    bool align_corners = j["align_corners"].get<bool>();

    std::vector<double> scale_factors;
    for (const auto& factor : j["scale_factors"]) {
        scale_factors.push_back(factor.get<double>());
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<
        UpsampleBilinear2DNode>(parent, debug_info, input_shape, output_shape, align_corners, scale_factors);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
