#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

#include <map>
#include <sstream>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"

namespace sdfg {
namespace math {
namespace tensor {

ConvNode::ConvNode(
    size_t element_id,
    const DebugInfo& debug_info,
    const graph::Vertex vertex,
    data_flow::DataFlowGraph& parent,
    const std::vector<symbolic::Expression>& shape,
    const std::vector<symbolic::Expression>& kernel_shape,
    const std::vector<symbolic::Expression>& strides,
    const std::vector<symbolic::Expression>& pads,
    const std::vector<symbolic::Expression>& dilations,
    symbolic::Expression output_channels,
    symbolic::Expression group
)
    : TensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Conv,
          {"Y"},
          {"X", "W", "B"}, // X and W are required, B (bias) is optional
          data_flow::ImplementationType_NONE
      ),
      shape_(shape), kernel_shape_(kernel_shape), strides_(strides), pads_(pads), dilations_(dilations),
      output_channels_(output_channels), group_(group) {}

void ConvNode::validate(const Function& function) const {
    TensorNode::validate(function);

    auto& graph = this->get_parent();

    // Custom validation for ConvNode that handles optional bias input
    // We expect X, W as required inputs and optionally B (bias)

    // Collect all input edges by connector name
    std::map<std::string, const data_flow::Memlet*> input_edges;
    for (auto& iedge : graph.in_edges(*this)) {
        input_edges[iedge.dst_conn()] = &iedge;
    }

    // Check that required inputs X and W are present
    if (input_edges.find("X") == input_edges.end()) {
        throw InvalidSDFGException("ConvNode: Required input 'X' is not connected");
    }
    if (input_edges.find("W") == input_edges.end()) {
        throw InvalidSDFGException("ConvNode: Required input 'W' is not connected");
    }

    // Validate that parameters are not empty
    if (shape_.empty()) {
        throw InvalidSDFGException("ConvNode shape cannot be empty");
    }
    if (kernel_shape_.empty()) {
        throw InvalidSDFGException("ConvNode kernel_shape cannot be empty");
    }
    if (strides_.empty()) {
        throw InvalidSDFGException("ConvNode strides cannot be empty");
    }
    if (pads_.empty()) {
        throw InvalidSDFGException("ConvNode pads cannot be empty");
    }
    if (dilations_.empty()) {
        throw InvalidSDFGException("ConvNode dilations cannot be empty");
    }

    // Validate consistent dimensions
    size_t spatial_dims = kernel_shape_.size();

    if (shape_.size() != spatial_dims + 2) {
        throw InvalidSDFGException("ConvNode shape must match kernel spatial dimensions + 2");
    }

    if (strides_.size() != spatial_dims) {
        throw InvalidSDFGException("ConvNode strides must match kernel spatial dimensions");
    }

    if (pads_.size() != 2 * spatial_dims) {
        throw InvalidSDFGException("ConvNode pads must have 2 * spatial dimensions (start and end for each axis)");
    }

    if (dilations_.size() != spatial_dims) {
        throw InvalidSDFGException("ConvNode dilations must match kernel spatial dimensions");
    }
}

bool ConvNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Validate nodes are standalone in the data flow graph
    auto& dfg = this->get_parent();
    if ((dfg.nodes().size() != 4 || dfg.edges().size() != 3) && (dfg.nodes().size() != 5 || dfg.edges().size() != 4)) {
        return false;
    }

    // Get edges
    auto iedges = dfg.in_edges_by_connector(*this);
    auto oedges = dfg.out_edges_by_connector(*this);
    if (iedges.size() != 3 || oedges.size() != 1) {
        return false;
    }
    auto* iedge_X = iedges.at(0);
    auto* iedge_W = iedges.at(1);
    auto* iedge_B = iedges.at(2);
    auto* oedge_Y = oedges.at(0);
    if (!iedge_X || !iedge_W || !oedge_Y) {
        return false;
    }
    bool has_bias = iedge_B != nullptr;

    // Get access nodes
    auto* access_X = dynamic_cast<data_flow::AccessNode*>(&iedge_X->src());
    auto* access_W = dynamic_cast<data_flow::AccessNode*>(&iedge_W->src());
    auto* access_B = (has_bias ? dynamic_cast<data_flow::AccessNode*>(&iedge_B->src()) : nullptr);
    auto* access_Y = dynamic_cast<data_flow::AccessNode*>(&oedge_Y->dst());
    if (!access_X || !access_W || (has_bias && !access_B) || !access_Y) {
        return false;
    }

    // Get block & its parent
    auto* block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (!block) {
        return false;
    }
    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    auto* block_parent = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(block));
    if (!block_parent) {
        return false;
    }
    size_t block_index = block_parent->index(*block);
    if (block_index >= block_parent->size()) {
        return false;
    }

    // Determine BLAS precision
    blas::BLAS_Precision precision;
    types::Scalar base_type(this->primitive_type(dfg));
    switch (base_type.primitive_type()) {
        case types::PrimitiveType::Half:
            precision = blas::BLAS_Precision::h;
            break;
        case types::PrimitiveType::Float:
            precision = blas::BLAS_Precision::s;
            break;
        case types::PrimitiveType::Double:
            precision = blas::BLAS_Precision::d;
            break;
        default:
            return false;
    }

    // Create new sequence for expansion
    auto& new_sequence = builder.add_sequence_before(
        *block_parent, *block, block_parent->at(block_index).second.assignments(), block->debug_info()
    );

    // Dimensions, i.e., 1D, 2D, 3D, ...
    size_t dims = this->kernel_shape_.size();
    symbolic::MultiExpression out_shape;
    out_shape.reserve(dims);
    // out_shape[i] = (shape[i + 2] + pads[i] + pads[dims + i] - dilations[i] * (kernel_shape[i] - 1) - 1)
    //                 / strides[i] + 1
    for (size_t i = 0; i < dims; i++) {
        out_shape.push_back(symbolic::add(
            symbolic::div(
                symbolic::sub(
                    symbolic::
                        sub(symbolic::add(this->shape_[i + 2], symbolic::add(this->pads_[i], this->pads_[dims + i])),
                            symbolic::mul(this->dilations_[i], symbolic::sub(this->kernel_shape_[i], symbolic::one()))),
                    symbolic::one()
                ),
                this->strides_[i]
            ),
            symbolic::one()
        ));
    }
    types::Scalar indvar_type(types::PrimitiveType::Int64);

    // Add patches container with malloc
    symbolic::Expression patches_size = this->shape_[1];
    for (size_t i = 0; i < dims; i++) {
        patches_size = symbolic::mul(patches_size, symbolic::mul(this->kernel_shape_[i], out_shape[i]));
    }
    types::Pointer patches_type(base_type);
    auto patches_container = builder.find_new_name("_patches");
    builder.add_container(patches_container, patches_type);
    auto& patches_malloc_block = builder.add_block(new_sequence, {}, block->debug_info());
    {
        auto& patches_access = builder.add_access(patches_malloc_block, patches_container, this->debug_info());
        auto& libnode = builder.add_library_node<stdlib::MallocNode>(
            patches_malloc_block, this->debug_info(), symbolic::mul(patches_size, symbolic::size_of_type(base_type))
        );
        builder.add_computational_memlet(
            patches_malloc_block, libnode, "_ret", patches_access, {}, patches_type, this->debug_info()
        );
    }

    // Add loop over batch size
    auto n_container = builder.find_new_name("_n");
    builder.add_container(n_container, indvar_type);
    auto n = symbolic::symbol(n_container);
    auto& for_loop_n = builder.add_for(
        new_sequence,
        n,
        symbolic::Lt(n, this->shape_[0]),
        symbolic::zero(),
        symbolic::add(n, symbolic::one()),
        {},
        block->debug_info()
    );
    structured_control_flow::Sequence* current_seq = &for_loop_n.root();

    // Add loops over output dimensions
    symbolic::MultiExpression os;
    os.reserve(dims);
    for (size_t i = 0; i < dims; i++) {
        auto o_container = builder.find_new_name("_o");
        builder.add_container(o_container, indvar_type);
        auto o = symbolic::symbol(o_container);
        os.push_back(o);
        auto& for_loop_o = builder.add_for(
            *current_seq,
            o,
            symbolic::Lt(o, out_shape[i]),
            symbolic::zero(),
            symbolic::add(o, symbolic::one()),
            {},
            block->debug_info()
        );
        current_seq = &for_loop_o.root();
    }

    // Add loop over channels
    auto c_container = builder.find_new_name("_c");
    builder.add_container(c_container, indvar_type);
    auto c = symbolic::symbol(c_container);
    auto& for_loop_c = builder.add_for(
        *current_seq,
        c,
        symbolic::Lt(c, this->shape_[1]),
        symbolic::zero(),
        symbolic::add(c, symbolic::one()),
        {},
        block->debug_info()
    );
    current_seq = &for_loop_c.root();

    // Add loops over kernel shape
    symbolic::MultiExpression ks;
    ks.reserve(dims);
    for (size_t i = 0; i < dims; i++) {
        auto k_container = builder.find_new_name("_k");
        builder.add_container(k_container, indvar_type);
        auto k = symbolic::symbol(k_container);
        ks.push_back(k);
        auto& for_loop_k = builder.add_for(
            *current_seq,
            k,
            symbolic::Lt(k, this->kernel_shape_[i]),
            symbolic::zero(),
            symbolic::add(k, symbolic::one()),
            {},
            block->debug_info()
        );
        current_seq = &for_loop_k.root();
    }

    // Add if/else
    auto& if_else = builder.add_if_else(*current_seq, {}, block->debug_info());
    symbolic::MultiExpression is;
    is.reserve(dims);
    symbolic::Condition true_condition = symbolic::__true__();
    symbolic::Condition false_condition = symbolic::__false__();
    for (size_t i = 0; i < dims; i++) {
        auto expr = symbolic::
            add(symbolic::sub(symbolic::mul(os[i], this->strides_[i]), this->pads_[i]),
                symbolic::mul(ks[i], this->dilations_[i]));
        is.push_back(expr);
        true_condition = symbolic::And(true_condition, symbolic::Lt(expr, this->shape_[i + 2]));
        false_condition = symbolic::Or(false_condition, symbolic::Ge(expr, this->shape_[i + 2]));
    }
    auto& true_case = builder.add_case(if_else, true_condition, block->debug_info());
    auto& false_case = builder.add_case(if_else, false_condition, block->debug_info());

    // Determine patches subset & tensor type
    data_flow::Subset patches_subset;
    patches_subset.push_back(c);
    patches_subset.insert(patches_subset.end(), ks.begin(), ks.end());
    patches_subset.insert(patches_subset.end(), os.begin(), os.end());
    symbolic::MultiExpression patches_shape;
    patches_shape.push_back(this->shape_[1]);
    patches_shape.insert(patches_shape.end(), this->kernel_shape_.begin(), this->kernel_shape_.end());
    patches_shape.insert(patches_shape.end(), out_shape.begin(), out_shape.end());
    types::Tensor patches_tensor_type(base_type, patches_shape);

    // Determine subset for X
    data_flow::Subset subset_X;
    subset_X.push_back(n);
    subset_X.push_back(c);
    subset_X.insert(subset_X.end(), is.begin(), is.end());

    // Add copy from X to patches to true case
    auto& true_block = builder.add_block(true_case, {}, block->debug_info());
    {
        auto& X_access = builder.add_access(true_block, access_X->data(), access_X->debug_info());
        auto& patches_access = builder.add_access(true_block, patches_container, this->debug_info());
        auto& tasklet = builder.add_tasklet(true_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(
            true_block, X_access, tasklet, "_in", subset_X, iedge_X->base_type(), iedge_X->debug_info()
        );
        builder.add_computational_memlet(
            true_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
        );
    }

    // Add zero assignment to false case
    auto& false_block = builder.add_block(false_case, {}, block->debug_info());
    {
        auto& const_zero = builder.add_constant(false_block, "0.0", base_type, this->debug_info());
        auto& patches_access = builder.add_access(false_block, patches_container, this->debug_info());
        auto& tasklet = builder.add_tasklet(false_block, data_flow::TaskletCode::assign, "_out", {"_in"});
        builder.add_computational_memlet(false_block, const_zero, tasklet, "_in", {}, this->debug_info());
        builder.add_computational_memlet(
            false_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
        );
    }

    // Add reference to output at current batch
    auto Y_ref = builder.find_new_name(access_Y->data());
    auto& Y_ref_type = builder.subject().type(access_Y->data());
    builder.add_container(Y_ref, Y_ref_type);
    auto& ref_block = builder.add_block(for_loop_n.root(), {}, block->debug_info());
    {
        auto& Y_access = builder.add_access(ref_block, access_Y->data(), access_Y->debug_info());
        auto& Y_ref_access = builder.add_access(ref_block, Y_ref, access_Y->debug_info());
        symbolic::Expression Y_ref_n = symbolic::mul(n, this->output_channels_);
        for (size_t i = 0; i < dims; i++) {
            Y_ref_n = symbolic::mul(Y_ref_n, out_shape[i]);
        }
        builder.add_reference_memlet(ref_block, Y_access, Y_ref_access, {Y_ref_n}, Y_ref_type, oedge_Y->debug_info());
    }

    // Add GEMM node
    auto& gemm_block = builder.add_block(for_loop_n.root(), {}, block->debug_info());
    {
        auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, this->debug_info());
        auto& beta = builder.add_constant(gemm_block, "0.0", base_type, this->debug_info());
        auto& W_access = builder.add_access(gemm_block, access_W->data(), access_W->debug_info());
        auto& patches_access = builder.add_access(gemm_block, patches_container, this->debug_info());
        auto& Y_access_in = builder.add_access(gemm_block, Y_ref, access_Y->debug_info());
        auto& Y_access_out = builder.add_access(gemm_block, Y_ref, access_Y->debug_info());
        symbolic::Expression gemm_n = symbolic::one();
        symbolic::Expression gemm_k = this->shape_[1];
        for (size_t i = 0; i < dims; i++) {
            gemm_n = symbolic::mul(gemm_n, out_shape[i]);
            gemm_k = symbolic::mul(gemm_k, this->kernel_shape_[i]);
        }
        auto& libnode = builder.add_library_node<blas::GEMMNode>(
            gemm_block,
            this->debug_info(),
            blas::ImplementationType_BLAS,
            precision, // precision
            blas::BLAS_Layout::RowMajor, // layout
            blas::BLAS_Transpose::No, // transA
            blas::BLAS_Transpose::No, // transB
            this->output_channels_, // m
            gemm_n, // n
            gemm_k, // k
            gemm_k, // lda
            gemm_n, // ldb
            gemm_n // ldc
        );
        builder.add_computational_memlet(gemm_block, alpha, libnode, "__alpha", {}, base_type, this->debug_info());
        builder.add_computational_memlet(gemm_block, beta, libnode, "__beta", {}, base_type, this->debug_info());
        builder.add_computational_memlet(
            gemm_block,
            W_access,
            libnode,
            "__A",
            {},
            types::Pointer(types::Scalar(iedge_W->base_type().primitive_type())),
            iedge_W->debug_info()
        );
        builder
            .add_computational_memlet(gemm_block, patches_access, libnode, "__B", {}, patches_type, this->debug_info());
        types::Pointer Y_edge_type(types::Scalar(oedge_Y->base_type().primitive_type()));
        builder
            .add_computational_memlet(gemm_block, Y_access_in, libnode, "__C", {}, Y_edge_type, oedge_Y->debug_info());
        builder
            .add_computational_memlet(gemm_block, libnode, "__C", Y_access_out, {}, Y_edge_type, oedge_Y->debug_info());
    }

    // Add free for patches container
    auto& patches_free_block = builder.add_block(new_sequence, {}, block->debug_info());
    {
        auto& patches_access_in = builder.add_access(patches_free_block, patches_container, this->debug_info());
        auto& patches_access_out = builder.add_access(patches_free_block, patches_container, this->debug_info());
        auto& libnode = builder.add_library_node<stdlib::FreeNode>(patches_free_block, this->debug_info());
        builder.add_computational_memlet(
            patches_free_block, patches_access_in, libnode, "_ptr", {}, patches_type, this->debug_info()
        );
        builder.add_computational_memlet(
            patches_free_block, libnode, "_ptr", patches_access_out, {}, patches_type, this->debug_info()
        );
    }

    // Clean up the original block
    builder.remove_memlet(*block, *iedge_X);
    builder.remove_memlet(*block, *iedge_W);
    if (has_bias) {
        builder.remove_memlet(*block, *iedge_B);
    }
    builder.remove_memlet(*block, *oedge_Y);
    builder.remove_node(*block, *access_X);
    builder.remove_node(*block, *access_W);
    if (has_bias) {
        builder.remove_node(*block, *access_B);
    }
    builder.remove_node(*block, *access_Y);
    builder.remove_node(*block, *this);
    builder.remove_child(*block_parent, block_index + 1);

    return true;
}

symbolic::SymbolSet ConvNode::symbols() const {
    symbolic::SymbolSet syms;

    for (auto& expr : shape_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : kernel_shape_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : strides_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : pads_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& expr : dilations_) {
        for (auto& atom : symbolic::atoms(expr)) {
            syms.insert(atom);
        }
    }
    for (auto& atom : symbolic::atoms(output_channels_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(group_)) {
        syms.insert(atom);
    }

    return syms;
}

void ConvNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    for (auto& expr : shape_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : kernel_shape_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : strides_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : pads_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    for (auto& expr : dilations_) {
        expr = symbolic::subs(expr, old_expression, new_expression);
    }
    output_channels_ = symbolic::subs(output_channels_, old_expression, new_expression);
    group_ = symbolic::subs(group_, old_expression, new_expression);
}

std::unique_ptr<data_flow::DataFlowNode> ConvNode::
    clone(size_t element_id, const graph::Vertex vertex, data_flow::DataFlowGraph& parent) const {
    return std::unique_ptr<data_flow::DataFlowNode>(new ConvNode(
        element_id,
        this->debug_info(),
        vertex,
        parent,
        shape_,
        kernel_shape_,
        strides_,
        pads_,
        dilations_,
        output_channels_,
        group_
    ));
}

std::string ConvNode::toStr() const {
    std::stringstream result;
    result << "Conv(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) {
            result << ", ";
        }
        result << shape_[i]->__str__();
    }
    result << "], kernel_shape=[";
    for (size_t i = 0; i < kernel_shape_.size(); ++i) {
        if (i > 0) {
            result << ", ";
        }
        result << kernel_shape_[i]->__str__();
    }
    result << "], strides=[";
    for (size_t i = 0; i < strides_.size(); ++i) {
        if (i > 0) {
            result << ", ";
        }
        result << strides_[i]->__str__();
    }
    result << "], pads=[";
    for (size_t i = 0; i < pads_.size(); ++i) {
        if (i > 0) {
            result << ", ";
        }
        result << pads_[i]->__str__();
    }
    result << "], dilations=[";
    for (size_t i = 0; i < dilations_.size(); ++i) {
        if (i > 0) {
            result << ", ";
        }
        result << dilations_[i]->__str__();
    }
    result << "], output_channels=" + output_channels_->__str__();
    result << ", group=" + group_->__str__() + ")";
    return result.str();
}

nlohmann::json ConvNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ConvNode& conv_node = static_cast<const ConvNode&>(library_node);
    nlohmann::json j;

    j["code"] = conv_node.code().value();

    serializer::JSONSerializer serializer;

    j["shape"] = nlohmann::json::array();
    for (auto& dim : conv_node.shape()) {
        j["shape"].push_back(serializer.expression(dim));
    }

    j["kernel_shape"] = nlohmann::json::array();
    for (auto& dim : conv_node.kernel_shape()) {
        j["kernel_shape"].push_back(serializer.expression(dim));
    }

    j["strides"] = nlohmann::json::array();
    for (auto& stride : conv_node.strides()) {
        j["strides"].push_back(serializer.expression(stride));
    }

    j["pads"] = nlohmann::json::array();
    for (auto& pad : conv_node.pads()) {
        j["pads"].push_back(serializer.expression(pad));
    }

    j["dilations"] = nlohmann::json::array();
    for (auto& dilation : conv_node.dilations()) {
        j["dilations"].push_back(serializer.expression(dilation));
    }

    j["output_channels"] = serializer.expression(conv_node.output_channels());
    j["group"] = serializer.expression(conv_node.group());

    return j;
}

data_flow::LibraryNode& ConvNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("element_id"));
    assert(j.contains("code"));
    assert(j.contains("debug_info"));
    assert(j.contains("kernel_shape"));

    std::vector<symbolic::Expression> shape;
    if (j.contains("shape")) {
        for (const auto& dim : j["shape"]) {
            shape.push_back(symbolic::parse(dim.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> kernel_shape;
    for (const auto& dim : j["kernel_shape"]) {
        kernel_shape.push_back(symbolic::parse(dim.get<std::string>()));
    }

    std::vector<symbolic::Expression> strides;
    if (j.contains("strides")) {
        for (const auto& stride : j["strides"]) {
            strides.push_back(symbolic::parse(stride.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> pads;
    if (j.contains("pads")) {
        for (const auto& pad : j["pads"]) {
            pads.push_back(symbolic::parse(pad.get<std::string>()));
        }
    }

    std::vector<symbolic::Expression> dilations;
    if (j.contains("dilations")) {
        for (const auto& dilation : j["dilations"]) {
            dilations.push_back(symbolic::parse(dilation.get<std::string>()));
        }
    }

    symbolic::Expression output_channels = symbolic::one();
    if (j.contains("output_channels")) {
        output_channels = symbolic::parse(j["output_channels"].get<std::string>());
    }

    symbolic::Expression group = symbolic::one();
    if (j.contains("group")) {
        group = symbolic::parse(j["group"].get<std::string>());
    }

    sdfg::serializer::JSONSerializer serializer;
    DebugInfo debug_info = serializer.json_to_debug_info(j["debug_info"]);

    return builder.add_library_node<
        ConvNode>(parent, debug_info, shape, kernel_shape, strides, pads, dilations, output_channels, group);
}

} // namespace tensor
} // namespace math
} // namespace sdfg
