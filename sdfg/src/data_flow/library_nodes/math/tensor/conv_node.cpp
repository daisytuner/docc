#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

#include <map>
#include <sstream>
#include <utility>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"
#include "symengine/integer.h"
#include "symengine/symengine_rcp.h"

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
    symbolic::Expression group,
    bool with_bias,
    QuantizationType quantization,
    const data_flow::ImplementationType& impl_type
)
    : SpatialTensorNode(
          element_id,
          debug_info,
          vertex,
          parent,
          LibraryNodeType_Conv,
          {},
          {"Y", "X", "W"}, // X and W are required, B (bias) is optional
          impl_type,
          quantization,
          shape,
          kernel_shape,
          strides,
          pads,
          dilations
      ),
      output_channels_(std::move(output_channels)), group_(std::move(group)), with_bias_(with_bias) {
    if (with_bias) {
        inputs_.push_back("B");
    }
}

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

    // Validate groups
    if (SymEngine::is_a<SymEngine::Integer>(*this->group_)) {
        auto group_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(this->group_)->as_int();
        if (SymEngine::is_a<SymEngine::Integer>(*this->shape_[1])) {
            auto input_channels_int = SymEngine::rcp_static_cast<const SymEngine::Integer>(this->shape_[1])->as_int();
            if (input_channels_int % group_int != 0) {
                throw InvalidSDFGException("ConvNode input channels must be divisible by groups");
            }
        }
        if (SymEngine::is_a<SymEngine::Integer>(*this->output_channels_)) {
            auto output_channels_int =
                SymEngine::rcp_static_cast<const SymEngine::Integer>(this->output_channels_)->as_int();
            if (output_channels_int % group_int != 0) {
                throw InvalidSDFGException("ConvNode output channels must be divisible by groups");
            }
        }
    }
}

blas::BLAS_Precision ConvNode::get_blas_precision(types::Scalar base_type) {
    switch (base_type.primitive_type()) {
        case types::PrimitiveType::Half:
            return blas::BLAS_Precision::h;
        case types::PrimitiveType::Float:
            return blas::BLAS_Precision::s;
        case types::PrimitiveType::Double:
            return blas::BLAS_Precision::d;
        default:
            return blas::BLAS_Precision::invalid;
    }
}

symbolic::MultiExpression ConvNode::get_out_shape() {
    size_t dims = kernel_shape_.size();
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
    return out_shape;
}

bool ConvNode::has_bias() const { return with_bias_; }

bool ConvNode::check_expandable(data_flow::DataFlowGraph& dfg, ConvExpandPrerequisits& boundary) const {
    if ((dfg.nodes().size() != 4 || dfg.edges().size() != 3) && (dfg.nodes().size() != 5 || dfg.edges().size() != 4)) {
        return false;
    }

    // Get edges
    boundary.iedge_X = dfg.in_edge_for_connector(*this, "X");
    boundary.iedge_W = dfg.in_edge_for_connector(*this, "W");
    boundary.iedge_B = with_bias_ ? dfg.in_edge_for_connector(*this, "B") : nullptr;
    boundary.iedge_Y = dfg.in_edge_for_connector(*this, "Y");
    if (!boundary.iedge_X || !boundary.iedge_W || !boundary.iedge_Y) {
        return false;
    }
    boundary.has_bias = boundary.iedge_B != nullptr;

    // Get access nodes
    boundary.access_X = dynamic_cast<const data_flow::AccessNode*>(&boundary.iedge_X->src());
    boundary.access_W = dynamic_cast<const data_flow::AccessNode*>(&boundary.iedge_W->src());
    boundary.access_B =
        (boundary.has_bias ? dynamic_cast<const data_flow::AccessNode*>(&boundary.iedge_B->src()) : nullptr);
    boundary.access_Y = dynamic_cast<const data_flow::AccessNode*>(&boundary.iedge_Y->src());
    if (!boundary.access_X || !boundary.access_W || (boundary.has_bias && !boundary.access_B) || !boundary.access_Y) {
        return false;
    }

    // Get block & its parent
    boundary.block = dyn_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (!boundary.block) {
        return false;
    }

    boundary.block_parent = dyn_cast<structured_control_flow::Sequence*>(boundary.block->get_parent());
    if (!boundary.block_parent) {
        return false;
    }

    boundary.block_index = boundary.block_parent->index(*boundary.block);
    if (boundary.block_index >= boundary.block_parent->size()) {
        return false;
    }

    return true;
}


passes::LibNodeExpander::ExpandOutcome ConvNode::
    expand(passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block) {
    auto& dfg = this->get_parent();
    ConvExpandPrerequisits b;
    if (!this->check_expandable(dfg, b)) {
        return context.unable();
    }

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> req_inputs = {Use::IndirectReadWrite, Use::IndirectRead, Use::IndirectRead};
    if (this->with_bias_) {
        req_inputs.push_back(Use::IndirectRead);
    }
    auto standalone = context.replacement_requires_access_nodes(req_inputs);

    if (!standalone) {
        return context.unable();
    }

    // Create new sequence for expansion
    types::Scalar base_type(this->primitive_type(dfg));
    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    // Dimensions, i.e., 1D, 2D, 3D, ...
    size_t dims = this->kernel_shape_.size();
    symbolic::MultiExpression out_shape = this->get_out_shape();
    types::Scalar indvar_type(types::PrimitiveType::Int64);

    // Create nested map structure for convolution
    structured_control_flow::Sequence* current_seq = &new_sequence;

    // Add loop over batch size
    auto n_container = builder.find_new_name("_n");
    builder.add_container(n_container, indvar_type);
    auto n = symbolic::symbol(n_container);
    auto& loop_n = builder.add_map(
        *current_seq,
        n,
        symbolic::Lt(n, this->shape_[0]),
        symbolic::zero(),
        symbolic::add(n, symbolic::one()),
        ScheduleType_Sequential::create(),
        b.block->debug_info()
    );
    current_seq = &loop_n.root();

    // Add loop over output channels
    auto l_container = builder.find_new_name("_l");
    builder.add_container(l_container, indvar_type);
    auto l = symbolic::symbol(l_container);
    auto& loop_l = builder.add_map(
        *current_seq,
        l,
        symbolic::Lt(l, this->output_channels_),
        symbolic::zero(),
        symbolic::add(l, symbolic::one()),
        ScheduleType_Sequential::create(),
        b.block->debug_info()
    );
    current_seq = &loop_l.root();

    // Add loops over output dimensions
    symbolic::SymbolVec os;
    os.reserve(dims);
    for (size_t i = 0; i < dims; i++) {
        auto o_container = builder.find_new_name("_o");
        builder.add_container(o_container, indvar_type);
        auto o = symbolic::symbol(o_container);
        os.push_back(o);
        auto& loop_o = builder.add_map(
            *current_seq,
            o,
            symbolic::Lt(o, out_shape[i]),
            symbolic::zero(),
            symbolic::add(o, symbolic::one()),
            ScheduleType_Sequential::create(),
            b.block->debug_info()
        );
        current_seq = &loop_o.root();
    }

    // Create accumulator variable for the sum
    std::string accum_container = builder.find_new_name("_conv_accum");
    builder.add_container(accum_container, base_type);

    // Initialize accumulator with zero
    structured_control_flow::Sequence* accum_seq = current_seq;
    auto& init_block = builder.add_block(*accum_seq, {}, b.block->debug_info());
    {
        auto& constant_zero = builder.add_constant(init_block, "0.0", base_type, this->debug_info_);
        auto& accum_access = builder.add_access(init_block, accum_container, this->debug_info_);
        auto& tasklet =
            builder.add_tasklet(init_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(init_block, constant_zero, tasklet, "_in", {}, this->debug_info_);
        builder.add_computational_memlet(init_block, tasklet, "_out", accum_access, {}, this->debug_info_);
    }

    // Add reduction over channels (per group)
    auto channels_per_group = symbolic::div(this->shape_[1], this->group_);
    auto c_container = builder.find_new_name("_c");
    builder.add_container(c_container, indvar_type);
    auto c = symbolic::symbol(c_container);
    auto& loop_c = builder.add_reduce(
        *current_seq,
        c,
        symbolic::Lt(c, channels_per_group),
        symbolic::zero(),
        symbolic::add(c, symbolic::one()),
        {{ReductionOperation::Add, accum_container}},
        structured_control_flow::ScheduleType_Sequential::create(),
        b.block->debug_info()
    );
    current_seq = &loop_c.root();

    // Add reductions over kernel shape
    symbolic::SymbolVec ks;
    ks.reserve(dims);
    for (size_t i = 0; i < dims; i++) {
        auto k_container = builder.find_new_name("_k");
        builder.add_container(k_container, indvar_type);
        auto k = symbolic::symbol(k_container);
        ks.push_back(k);
        auto& loop_k = builder.add_reduce(
            *current_seq,
            k,
            symbolic::Lt(k, this->kernel_shape_[i]),
            symbolic::zero(),
            symbolic::add(k, symbolic::one()),
            {{ReductionOperation::Add, accum_container}},
            structured_control_flow::ScheduleType_Sequential::create(),
            b.block->debug_info()
        );
        current_seq = &loop_k.root();
    }

    // Check if at least one padding value is non-zero or unknown
    bool has_padding = false;
    for (auto& pad : this->pads_) {
        if (SymEngine::is_a<SymEngine::Integer>(*pad)) {
            if (SymEngine::rcp_static_cast<const SymEngine::Integer>(pad)->as_int() != 0) {
                // We found a non-zero padding value
                has_padding = true;
                break;
            }
        } else {
            // We just don't know if the convolution is padded and assume that it is as a fall-back
            has_padding = true;
            break;
        }
    }

    // Compute spatial input dimensions
    symbolic::MultiExpression is;
    is.reserve(dims);
    for (size_t i = 0; i < dims; i++) {
        is.push_back(symbolic::
                         add(symbolic::sub(symbolic::mul(os[i], this->strides_[i]), this->pads_[i]),
                             symbolic::mul(ks[i], this->dilations_[i])));
    }

    // If convolution is padded, add branch to stay in bounds for computation
    if (has_padding) {
        symbolic::Condition comp_condition = symbolic::__true__();
        for (size_t i = 0; i < dims; i++) {
            comp_condition = symbolic::
                And(comp_condition,
                    symbolic::And(symbolic::Lt(is[i], this->shape_[i + 2]), symbolic::Ge(is[i], symbolic::zero())));
        }
        auto& branch = builder.add_if_else(*current_seq, b.block->debug_info());
        current_seq = &builder.add_case(branch, comp_condition, b.block->debug_info());
    }

    // Determine subsets for computation
    auto out_channels_per_group = symbolic::div(this->output_channels_, this->group_);
    auto group_idx = symbolic::div(l, out_channels_per_group);
    auto input_channel_idx = symbolic::add(symbolic::mul(group_idx, channels_per_group), c);
    data_flow::Subset X_subset;
    X_subset.push_back(n);
    X_subset.push_back(input_channel_idx);
    X_subset.insert(X_subset.end(), is.begin(), is.end());
    data_flow::Subset W_subset;
    W_subset.push_back(l);
    W_subset.push_back(c);
    W_subset.insert(W_subset.end(), ks.begin(), ks.end());

    // Create computation block
    auto& comp_block = builder.add_block(*current_seq, {}, b.block->debug_info());
    {
        auto& X_access = standalone->add_indirect_read_access(comp_block, X_INPUT_IDX);
        auto& W_access = standalone->add_indirect_read_access(comp_block, W_INPUT_IDX);
        auto& accum_access_in = builder.add_access(comp_block, accum_container, this->debug_info_);
        auto& accum_access_out = builder.add_access(comp_block, accum_container, this->debug_info_);
        auto& tasklet = builder.add_tasklet(
            comp_block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"}, this->debug_info_
        );
        builder.add_computational_memlet(
            comp_block, X_access, tasklet, "_in1", X_subset, b.iedge_X->base_type(), b.iedge_X->debug_info()
        );
        builder.add_computational_memlet(
            comp_block, W_access, tasklet, "_in2", W_subset, b.iedge_W->base_type(), b.iedge_W->debug_info()
        );
        builder.add_computational_memlet(comp_block, accum_access_in, tasklet, "_in3", {}, base_type, this->debug_info_);
        builder
            .add_computational_memlet(comp_block, tasklet, "_out", accum_access_out, {}, base_type, this->debug_info_);
    }

    // Determine subsets for output
    data_flow::Subset Y_subset;
    Y_subset.push_back(n);
    Y_subset.push_back(l);
    Y_subset.insert(Y_subset.end(), os.begin(), os.end());

    // Create output block, i.e., write accumulation back to output
    auto& output_block = builder.add_block(*accum_seq, {}, b.block->debug_info());
    if (b.has_bias) {
        auto& accum_access = builder.add_access(output_block, accum_container, this->debug_info_);
        auto& B_access = standalone->add_indirect_read_access(output_block, B_INPUT_IDX);
        auto& Y_access = standalone->add_indirect_write_access(output_block, Y_INPUT_IDX);
        auto& tasklet =
            builder
                .add_tasklet(output_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info_);
        builder.add_computational_memlet(output_block, accum_access, tasklet, "_in1", {}, base_type, this->debug_info_);
        builder.add_computational_memlet(
            output_block, B_access, tasklet, "_in2", {l}, b.iedge_B->base_type(), b.iedge_B->debug_info()
        );
        builder.add_computational_memlet(
            output_block, tasklet, "_out", Y_access, Y_subset, b.iedge_Y->base_type(), b.iedge_Y->debug_info()
        );
    } else {
        auto& accum_access = builder.add_access(output_block, accum_container, this->debug_info_);
        auto& Y_access = standalone->add_indirect_write_access(output_block, Y_INPUT_IDX);
        auto& tasklet =
            builder.add_tasklet(output_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info_);
        builder.add_computational_memlet(output_block, accum_access, tasklet, "_in", {}, base_type, this->debug_info_);
        builder.add_computational_memlet(
            output_block, tasklet, "_out", Y_access, Y_subset, b.iedge_Y->base_type(), b.iedge_Y->debug_info()
        );
    }

    return standalone->successfully_expanded();
}

symbolic::SymbolSet ConvNode::symbols() const {
    auto syms = SpatialTensorNode::symbols();
    for (auto& atom : symbolic::atoms(output_channels_)) {
        syms.insert(atom);
    }
    for (auto& atom : symbolic::atoms(group_)) {
        syms.insert(atom);
    }

    return syms;
}

void ConvNode::replace(const symbolic::Expression old_expression, const symbolic::Expression new_expression) {
    SpatialTensorNode::replace(old_expression, new_expression);
    output_channels_ = symbolic::subs(output_channels_, old_expression, new_expression);
    group_ = symbolic::subs(group_, old_expression, new_expression);
}

void ConvNode::replace(const symbolic::ExpressionMapping& replacements) {
    SpatialTensorNode::replace(replacements);
    output_channels_ = symbolic::subs(output_channels_, replacements);
    group_ = symbolic::subs(group_, replacements);
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
        group_,
        with_bias_,
        fixed_quantization_,
        implementation_type_
    ));
}

std::string ConvNode::toStr() const {
    std::stringstream result;
    result << "Conv(";
    SpatialTensorNode::operator<<(result);

    result << ", output_channels=" + output_channels_->__str__();
    result << ", group=" + group_->__str__() + ")";
    return result.str();
}

symbolic::Expression ConvNode::flop() const {
    // Total FLOPs = output_elements * K_conv (multiplications)
    //             + output_elements * (K_conv - 1) (additions)
    auto output_elems = num_output_elements();
    auto k_conv = kernel_iteration_count();

    auto mul_ops = symbolic::mul(output_elems, k_conv);
    auto add_ops = symbolic::mul(output_elems, symbolic::sub(k_conv, symbolic::one()));
    return symbolic::add(mul_ops, add_ops);
}

data_flow::PointerAccessType ConvNode::pointer_access_type(int input_idx) const {
    if (input_idx == 0) {
        return data_flow::PointerAccessMeta::create_full_write_only(symbolic::__nullptr__(), true);
    } else if (input_idx >= 1 && input_idx < inputs_.size()) {
        return data_flow::PointerAccessMeta::create_read_only(symbolic::__nullptr__(), true);
    } else {
        return TensorNode::pointer_access_type(input_idx);
    }
}

symbolic::Expression ConvNode::num_output_elements() const {
    // N * C_out * prod(output_spatial_dim(i))
    return symbolic::mul(symbolic::mul(shape_[0], output_channels_), output_spatial_volume());
}

symbolic::Expression ConvNode::kernel_iteration_count() const {
    // (C_in / group) * prod(kernel_shape_[i])
    return symbolic::mul(symbolic::div(shape_[1], group_), kernel_volume());
}

nlohmann::json ConvNodeSerializer::serialize(const data_flow::LibraryNode& library_node) {
    const ConvNode& conv_node = static_cast<const ConvNode&>(library_node);
    nlohmann::json j;

    serializer::JSONSerializer serializer;
    j["output_channels"] = serializer.expression(conv_node.output_channels());
    j["group"] = serializer.expression(conv_node.group());
    j["with_bias"] = conv_node.has_bias();

    fill_base_values(conv_node, j);

    return j;
}

data_flow::LibraryNode& ConvNodeSerializer::deserialize(
    const nlohmann::json& j, builder::StructuredSDFGBuilder& builder, structured_control_flow::Block& parent
) {
    assert(j.contains("kernel_shape"));

    auto base = deserialize_base_values(j);

    auto bias_it = j.find("with_bias");
    bool with_bias = false;
    if (bias_it != j.end()) {
        with_bias = bias_it->get<bool>();
    }

    symbolic::Expression output_channels = symbolic::one();
    if (j.contains("output_channels")) {
        output_channels = symbolic::parse(j["output_channels"].get<std::string>());
    }

    symbolic::Expression group = symbolic::one();
    if (j.contains("group")) {
        group = symbolic::parse(j["group"].get<std::string>());
    }

    return builder.add_library_node<ConvNode>(
        parent,
        base.debug_info,
        base.shape,
        base.kernel_shape,
        base.strides,
        base.pads,
        base.dilations,
        output_channels,
        group,
        with_bias,
        base.quantization
    );
}

} // namespace tensor
} // namespace math
} // namespace sdfg
