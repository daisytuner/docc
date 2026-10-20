#include "sdfg/fp_math/library_nodes/ozaki_matmul_node.h"

#include <sstream>

#include <nlohmann/json.hpp>

#include "../contractions/contraction_kernels.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/visitor/for_each.h"

namespace sdfg {
namespace fp_math {

using math::tensor::TensorLayout;

namespace {

std::vector<std::string> inputs_of(int slices) {
    std::vector<std::string> inputs{"Y", "A", "SB"};
    for (int t = 0; t < slices; t++) {
        inputs.push_back("B" + std::to_string(t));
    }
    return inputs;
}

/// Reads A and writes Y through the node's layouts; B is read through its prepared slices only.
class PreparedOperands : public detail::ContractionOperands {
    passes::LibNodeExpander::AccessNodeExpand& standalone_;
    TensorLayout layout_a_, layout_y_;
    types::Pointer pointer_;

public:
    PreparedOperands(passes::LibNodeExpander::AccessNodeExpand& standalone, const OzakiMatMulNode& node)
        : standalone_(standalone), layout_a_(node.layout_a()), layout_y_(node.layout_y()),
          pointer_(types::Scalar(node.element())) {
        m = node.m();
        n = node.n();
        k = node.k();
        element = node.element();
        a_transposed = TensorLayout::is_last_dims_col_or_row_major(layout_a_.strides()) ==
                       TensorLayout::LAYOUT_COL_MAJOR;
    }

    KernelValue
    a(BlockBuilder& block,
      const symbolic::MultiExpression&,
      const symbolic::Expression& row,
      const symbolic::Expression& column) override {
        auto& access = standalone_.add_indirect_read_access(block.block(), OzakiMatMulNode::A_INPUT_IDX);
        return block.read(access, {layout_a_.resolve_element({row, column})}, pointer_);
    }

    KernelValue b(
        BlockBuilder&, const symbolic::MultiExpression&, const symbolic::Expression&, const symbolic::Expression&
    ) override {
        throw InvalidSDFGException("OzakiMatMul: B is read through its prepared slices only");
    }

    void store(
        BlockBuilder& block,
        const KernelValue& value,
        const symbolic::MultiExpression&,
        const symbolic::Expression& row,
        const symbolic::Expression& column
    ) override {
        auto& access = standalone_.add_indirect_write_access(block.block(), OzakiMatMulNode::Y_INPUT_IDX);
        block.store(block.convert(value, element), access, {layout_y_.resolve_element({row, column})}, pointer_);
    }
};

} // namespace

OzakiMatMulNode::OzakiMatMulNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const TensorLayout& layout_a,
    const TensorLayout& layout_y,
    types::PrimitiveType element,
    int slices,
    const data_flow::ImplementationType& impl_type
)
    : LibraryNode(
          element_id, debug_info, vertex, parent, LibraryNodeType_OzakiMatMul, {}, inputs_of(slices), false, impl_type
      ),
      layout_a_(layout_a), layout_y_(layout_y), element_(element), slices_(slices) {
    if (slices < 1) {
        throw std::invalid_argument("OzakiMatMulNode: at least one slice is needed");
    }
}

symbolic::Expression OzakiMatMulNode::m() const {
    return layout_a_.get_dim(0);
}

symbolic::Expression OzakiMatMulNode::n() const {
    return layout_y_.get_dim(1);
}

symbolic::Expression OzakiMatMulNode::k() const {
    return layout_a_.get_dim(1);
}

void OzakiMatMulNode::validate(const Function& function) const {
    LibraryNode::validate(function);
    if (layout_a_.dims() != 2 || layout_y_.dims() != 2) {
        throw InvalidSDFGException("OzakiMatMul: A and Y must be matrices");
    }
    if (element_ != types::PrimitiveType::Float && element_ != types::PrimitiveType::Double) {
        throw InvalidSDFGException("OzakiMatMul: A and Y must be Float or Double");
    }
    auto& graph = get_parent();
    if (graph.in_degree(*this) != inputs_.size() || graph.out_degree(*this) != 0) {
        throw InvalidSDFGException("OzakiMatMul: expected the inputs Y, A, SB, B0 ... and no outputs");
    }
}

symbolic::SymbolSet OzakiMatMulNode::symbols() const {
    symbolic::SymbolSet syms;
    layout_a_.collect_symbols(syms);
    layout_y_.collect_symbols(syms);
    return syms;
}

void OzakiMatMulNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    layout_a_.replace_symbols(old_expression, new_expression);
    layout_y_.replace_symbols(old_expression, new_expression);
}

void OzakiMatMulNode::replace(const symbolic::ExpressionMapping& replacements) {
    layout_a_.replace_symbols(replacements);
    layout_y_.replace_symbols(replacements);
}

std::unique_ptr<data_flow::DataFlowNode> OzakiMatMulNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::make_unique<OzakiMatMulNode>(
        element_id, debug_info(), vertex, parent, layout_a_, layout_y_, element_, slices_, implementation_type_
    );
}

std::string OzakiMatMulNode::toStr() const {
    std::stringstream ss;
    ss << "OzakiMatMul(" << m()->__str__() << " x " << k()->__str__() << " @ " << k()->__str__() << " x "
       << n()->__str__() << ", " << slices_ << " slices)";
    return ss.str();
}

data_flow::PointerAccessType OzakiMatMulNode::pointer_access_type(int input_idx) const {
    if (input_idx == Y_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_full_write_only(layout_y_.total_elements(), true, layout_y_);
    }
    if (input_idx == A_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(layout_a_.total_elements(), true, layout_a_);
    }
    if (input_idx == SCALES_INPUT_IDX) {
        return data_flow::PointerAccessMeta::create_read_only(n(), true);
    }
    if (input_idx >= static_cast<int>(FIRST_SLICE_INPUT_IDX) &&
        input_idx < static_cast<int>(FIRST_SLICE_INPUT_IDX) + slices_) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::mul(k(), n()), true);
    }
    throw std::invalid_argument("OzakiMatMulNode: invalid input index for pointer access type");
}

nlohmann::json OzakiMatMulNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const auto& node = static_cast<const OzakiMatMulNode&>(library_node);
    nlohmann::json j;
    j["code"] = node.code().value();
    node.layout_a().serialize_to_json(j["layout_a"]);
    node.layout_y().serialize_to_json(j["layout_y"]);
    j["element"] = node.element();
    j["slices"] = node.slices();
    return j;
}

data_flow::LibraryNode& OzakiMatMulNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);
    return builder.add_library_node<OzakiMatMulNode>(
        parent,
        debug_info,
        TensorLayout::deserialize_from_json(j.at("layout_a")),
        TensorLayout::deserialize_from_json(j.at("layout_y")),
        j.at("element").get<types::PrimitiveType>(),
        j.at("slices").get<int>()
    );
}

passes::LibNodeExpander::ExpandOutcome OzakiMatMulExpander::handle_expand(
    passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block&, OzakiMatMulNode& node
) const {
    // The slices and scales are read by name, so they are taken from the node's inputs before it is replaced.
    std::vector<std::string> slices(node.slices());
    std::string scales;
    for (auto& memlet : node.get_parent().in_edges(node)) {
        auto& source = static_cast<const data_flow::AccessNode&>(memlet.src());
        if (memlet.dst_conn() == "SB") {
            scales = source.data();
        } else if (memlet.dst_conn().starts_with("B")) {
            slices.at(std::stoi(memlet.dst_conn().substr(1))) = source.data();
        }
    }

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> uses{Use::IndirectWrite, Use::IndirectRead};
    uses.resize(node.inputs().size(), Use::Skip);
    auto standalone = context.replacement_requires_access_nodes(uses);
    if (!standalone) {
        return context.unable();
    }
    PreparedOperands operands(*standalone, node);
    auto& sequence = standalone->replace_with_sequence();
    KernelBuilder kb(standalone->builder(), sequence, node.debug_info());
    detail::build_slicing_direct_prepared(kb, operands, slices, scales);
    kb.finish();
    return standalone->successfully_expanded();
}

bool expand_ozaki_matmuls(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    std::vector<std::pair<structured_control_flow::Block*, OzakiMatMulNode*>> found;
    visitor::for_each_block(builder.subject().root(), [&](structured_control_flow::Block& block) {
        for (auto* node : block.dataflow().library_nodes()) {
            if (auto* ozaki = dynamic_cast<OzakiMatMulNode*>(node)) {
                found.emplace_back(&block, ozaki);
            }
        }
    });
    OzakiMatMulExpander expander;
    for (auto& [block, node] : found) {
        if (!passes::expansion::expand_single_node(builder, *block, *node, expander).expanded) {
            throw InvalidSDFGException("OzakiMatMul: the node's operands could not be isolated from its block");
        }
    }
    if (!found.empty()) {
        analysis_manager.invalidate_all();
    }
    return !found.empty();
}

} // namespace fp_math
} // namespace sdfg
