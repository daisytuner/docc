#include "sdfg/data_flow/library_nodes/math/tensor/clamp_node.h"

#include <sstream>

#include "sdfg/data_flow/library_nodes/math/tensor/elementwise_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/tensor_layout.h"
#include "sdfg/data_flow/pointer_metadata.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace math {
namespace tensor {

static std::vector<std::string> clamp_inputs(bool has_min, bool has_max) {
    std::vector<std::string> inputs = {"Y", "X"};
    if (has_min) {
        inputs.push_back("min");
    }
    if (has_max) {
        inputs.push_back("max");
    }
    return inputs;
}

ClampNode::ClampNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    bool has_min,
    bool has_max,
    const data_flow::ImplementationType& impl_type
)
    : TensorNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_Clamp, {}, clamp_inputs(has_min, has_max), impl_type
      ),
      shape_(shape), has_min_(has_min), has_max_(has_max) {
}

const std::vector<symbolic::Expression>& ClampNode::shape() const {
    return this->shape_;
}

bool ClampNode::has_min() const {
    return this->has_min_;
}

bool ClampNode::has_max() const {
    return this->has_max_;
}

int ClampNode::min_input_idx() const {
    return this->has_min_ ? 2 : -1;
}

int ClampNode::max_input_idx() const {
    return this->has_max_ ? (this->has_min_ ? 3 : 2) : -1;
}

void ClampNode::validate(const Function& function) const {
    TensorNode::validate(function);

    if (!this->has_min_ && !this->has_max_) {
        throw InvalidSDFGException("ClampNode: At least one of min and max is required");
    }

    auto& graph = this->get_parent();
    if (graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("ClampNode: Expected no outputs but got: " + std::to_string(graph.out_degree(*this)));
    }

    for (int idx : {Y_INPUT_IDX, X_INPUT_IDX}) {
        const auto& conn = this->inputs_.at(idx);
        const auto* iedge = graph.in_edge_for_connector(*this, conn);
        if (!iedge) {
            throw InvalidSDFGException("ClampNode: No memlet connected at connector: " + conn);
        }
        if (iedge->base_type().type_id() != types::TypeID::Tensor) {
            throw InvalidSDFGException(
                "ClampNode: Expected tensor type at connector '" + conn + "' but got: " + iedge->base_type().print()
            );
        }
        this->validate_shape_matches(this->shape_, static_cast<const types::Tensor&>(iedge->base_type()).layout(), conn);
    }

    for (size_t idx = X_INPUT_IDX + 1; idx < this->inputs_.size(); ++idx) {
        const auto& conn = this->inputs_.at(idx);
        const auto* iedge = graph.in_edge_for_connector(*this, conn);
        if (!iedge) {
            throw InvalidSDFGException("ClampNode: No memlet connected at connector: " + conn);
        }
        if (iedge->base_type().type_id() != types::TypeID::Scalar) {
            throw InvalidSDFGException(
                "ClampNode: Expected scalar type at connector '" + conn + "' but got: " + iedge->base_type().print()
            );
        }
    }
}

bool ClampNode::supports_integer_types() const {
    return true;
}

using Use = passes::LibNodeExpander::InputUse;

passes::LibNodeExpander::ExpandOutcome ClampNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dfg = this->get_parent();
    std::vector<const data_flow::Memlet*> iedges;
    for (const auto& conn : this->inputs_) {
        const auto* iedge = dfg.in_edge_for_connector(*this, conn);
        if (!iedge) {
            return context.unable();
        }
        iedges.push_back(iedge);
    }

    std::vector<Use> uses = {Use::IndirectWrite, Use::IndirectRead};
    uses.resize(this->inputs_.size(), Use::Scalar);
    auto standalone = context.replacement_requires_access_nodes(uses);
    if (!standalone) {
        return context.unable();
    }

    auto& builder = standalone->builder();
    auto& new_sequence = standalone->replace_with_sequence();
    auto [body, subset] =
        ElementWiseDataflowTensorNode::add_eltwise_scope(builder, this->debug_info_, new_sequence, this->shape_);

    const auto* iedge_x = iedges.at(X_INPUT_IDX);
    const auto* iedge_y = iedges.at(Y_INPUT_IDX);
    auto prim = iedge_x->base_type().primitive_type();
    bool is_float = types::is_floating_point(prim);
    types::Scalar element_type(prim);
    types::Scalar bool_type(types::PrimitiveType::Bool);
    auto new_tmp = [&](const std::string& prefix, const types::Scalar& type) {
        auto name = builder.find_new_name(prefix);
        builder.add_container(name, type);
        return name;
    };

    auto x_container = new_tmp("tmp_clamp_x_", element_type);
    auto& load_block = builder.add_block(*body, this->debug_info_);
    {
        auto& x_in = standalone->add_indirect_read_access(load_block, X_INPUT_IDX);
        auto& x_tmp = builder.add_access(load_block, x_container, this->debug_info_);
        auto& assign =
            builder.add_tasklet(load_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(
            load_block, x_in, assign, "_in", subset, iedge_x->base_type(), iedge_x->debug_info()
        );
        builder.add_computational_memlet(load_block, assign, "_out", x_tmp, {}, this->debug_info_);
    }

    // Picks the bound if it is exceeded or NaN; a NaN in src never exceeds the bound and is kept.
    auto apply_bound = [&](const std::string& src, int bound_idx, bool upper) {
        data_flow::TaskletCode cmp_code;
        if (is_float) {
            cmp_code = upper ? data_flow::TaskletCode::fp_ogt : data_flow::TaskletCode::fp_olt;
        } else if (types::is_unsigned(prim)) {
            cmp_code = upper ? data_flow::TaskletCode::int_ugt : data_flow::TaskletCode::int_ult;
        } else {
            cmp_code = upper ? data_flow::TaskletCode::int_sgt : data_flow::TaskletCode::int_slt;
        }
        const auto* iedge_bound = iedges.at(bound_idx);

        auto exceeds_container = new_tmp("tmp_clamp_exceeds_", bool_type);
        std::string nan_container = is_float ? new_tmp("tmp_clamp_nan_", bool_type) : "";
        auto& cmp_block = builder.add_block(*body, this->debug_info_);
        {
            auto& src_tmp = builder.add_access(cmp_block, src, this->debug_info_);
            auto& bound = standalone->add_scalar_input_access(cmp_block, bound_idx);
            auto& exceeds = builder.add_access(cmp_block, exceeds_container, this->debug_info_);
            auto& cmp = builder.add_tasklet(cmp_block, cmp_code, "_out", {"_in1", "_in2"}, this->debug_info_);
            builder.add_computational_memlet(cmp_block, src_tmp, cmp, "_in1", {}, this->debug_info_);
            builder.add_computational_memlet(
                cmp_block, bound, cmp, "_in2", {}, iedge_bound->base_type(), iedge_bound->debug_info()
            );
            builder.add_computational_memlet(cmp_block, cmp, "_out", exceeds, {}, this->debug_info_);

            if (is_float) {
                auto& nan = builder.add_access(cmp_block, nan_container, this->debug_info_);
                auto& uno = builder.add_tasklet(
                    cmp_block, data_flow::TaskletCode::fp_uno, "_out", {"_in1", "_in2"}, this->debug_info_
                );
                builder.add_computational_memlet(
                    cmp_block, bound, uno, "_in1", {}, iedge_bound->base_type(), iedge_bound->debug_info()
                );
                builder.add_computational_memlet(
                    cmp_block, bound, uno, "_in2", {}, iedge_bound->base_type(), iedge_bound->debug_info()
                );
                builder.add_computational_memlet(cmp_block, uno, "_out", nan, {}, this->debug_info_);
            }
        }

        auto exceeds = symbolic::symbol(exceeds_container);
        symbolic::Condition pick_bound = symbolic::Eq(exceeds, symbolic::__true__());
        symbolic::Condition keep_src = symbolic::Eq(exceeds, symbolic::__false__());
        if (is_float) {
            auto nan = symbolic::symbol(nan_container);
            pick_bound = symbolic::Or(pick_bound, symbolic::Eq(nan, symbolic::__true__()));
            keep_src = symbolic::And(keep_src, symbolic::Eq(nan, symbolic::__false__()));
        }

        auto dst_container = new_tmp(upper ? "tmp_clamp_hi_" : "tmp_clamp_lo_", element_type);
        auto& if_else = builder.add_if_else(*body, this->debug_info_);
        auto& case_bound = builder.add_case(if_else, pick_bound, this->debug_info_);
        auto& case_src = builder.add_case(if_else, keep_src, this->debug_info_);

        auto& bound_block = builder.add_block(case_bound, this->debug_info_);
        {
            auto& bound = standalone->add_scalar_input_access(bound_block, bound_idx);
            auto& dst = builder.add_access(bound_block, dst_container, this->debug_info_);
            auto& assign =
                builder.add_tasklet(bound_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
            builder.add_computational_memlet(
                bound_block, bound, assign, "_in", {}, iedge_bound->base_type(), iedge_bound->debug_info()
            );
            builder.add_computational_memlet(bound_block, assign, "_out", dst, {}, this->debug_info_);
        }

        auto& src_block = builder.add_block(case_src, this->debug_info_);
        {
            auto& src_tmp = builder.add_access(src_block, src, this->debug_info_);
            auto& dst = builder.add_access(src_block, dst_container, this->debug_info_);
            auto& assign =
                builder.add_tasklet(src_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
            builder.add_computational_memlet(src_block, src_tmp, assign, "_in", {}, this->debug_info_);
            builder.add_computational_memlet(src_block, assign, "_out", dst, {}, this->debug_info_);
        }

        return dst_container;
    };

    // Same order as PyTorch: max with the lower bound first, so min > max yields max.
    std::string result = x_container;
    if (this->has_min_) {
        result = apply_bound(result, this->min_input_idx(), false);
    }
    if (this->has_max_) {
        result = apply_bound(result, this->max_input_idx(), true);
    }

    auto& store_block = builder.add_block(*body, this->debug_info_);
    {
        auto& res = builder.add_access(store_block, result, this->debug_info_);
        auto& y_out = standalone->add_indirect_write_access(store_block, Y_INPUT_IDX);
        auto& assign =
            builder.add_tasklet(store_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(store_block, res, assign, "_in", {}, this->debug_info_);
        builder.add_computational_memlet(
            store_block, assign, "_out", y_out, subset, iedge_y->base_type(), iedge_y->debug_info()
        );
    }

    return standalone->successfully_expanded();
}

data_flow::PointerAccessType ClampNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx == X_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    }
    return TensorNode::pointer_access_type(input_idx);
}

std::string ClampNode::toStr() const {
    std::stringstream stream;
    stream << "ClampNode(shape: ";
    TensorLayout::emit_symbolic_list(stream, this->shape_);
    stream << ", min: " << (this->has_min_ ? "yes" : "no") << ", max: " << (this->has_max_ ? "yes" : "no") << ")";
    return stream.str();
}

symbolic::SymbolSet ClampNode::symbols() const {
    symbolic::SymbolSet syms;
    for (const auto& dim : this->shape_) {
        for (auto& atom : symbolic::atoms(dim)) {
            syms.insert(atom);
        }
    }
    return syms;
}

std::unique_ptr<data_flow::DataFlowNode> ClampNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<ClampNode>(
        element_id,
        this->debug_info_,
        vertex,
        parent,
        this->shape_,
        this->has_min_,
        this->has_max_,
        this->implementation_type_
    );
}

void ClampNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& dim : this->shape_) {
        dim = symbolic::subs(dim, old_expression, new_expression);
    }
}

void ClampNode::replace(const symbolic::ExpressionMapping& replacements) {
    for (auto& dim : this->shape_) {
        dim = symbolic::subs(dim, replacements);
    }
}

nlohmann::json ClampNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const ClampNode&>(library_node);
    nlohmann::json j;

    j["code"] = node.code().value();
    j["has_min"] = node.has_min();
    j["has_max"] = node.has_max();

    serializer::JSONSerializer serializer;
    j["shape"] = nlohmann::json::array();
    for (const auto& dim : node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    return j;
}

data_flow::LibraryNode& ClampNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("shape"));
    assert(j.contains("has_min"));
    assert(j.contains("has_max"));
    assert(j.contains("debug_info"));

    std::vector<symbolic::Expression> shape;
    for (const auto& dim : j.at("shape")) {
        shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j.at("debug_info"));

    return builder.add_library_node<
        ClampNode>(parent, debug_info, shape, j.at("has_min").get<bool>(), j.at("has_max").get<bool>());
}

} // namespace tensor
} // namespace math
} // namespace sdfg
