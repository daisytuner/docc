#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

#include <map>
#include <sstream>
#include <utility>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/memset.h"
#include "sdfg/data_flow/memlet.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
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

bool ConvNode::check_expandable(
    data_flow::DataFlowGraph& dfg, analysis::AnalysisManager& analysis_manager, ConvExpandPrerequisits& boundary
) const {
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
    boundary.block = dynamic_cast<structured_control_flow::Block*>(dfg.get_parent());
    if (!boundary.block) {
        return false;
    }

    auto& scope_analysis = analysis_manager.get<analysis::ScopeAnalysis>();
    boundary.block_parent = dynamic_cast<structured_control_flow::Sequence*>(scope_analysis.parent_scope(boundary.block)
    );
    if (!boundary.block_parent) {
        return false;
    }

    boundary.block_index = boundary.block_parent->index(*boundary.block);
    if (boundary.block_index >= boundary.block_parent->size()) {
        return false;
    }

    return true;
}

bool ConvNode::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    // Validate nodes are standalone in the data flow graph
    auto& dfg = this->get_parent();
    ConvExpandPrerequisits b;
    if (!check_expandable(dfg, analysis_manager, b)) {
        return false;
    }

    // Determine BLAS precision

    types::Scalar base_type(this->primitive_type(dfg));
    blas::BLAS_Precision precision = get_blas_precision(base_type);
    if (precision == blas::BLAS_Precision::invalid) {
        return false;
    }

    // Create new sequence for expansion
    auto& new_sequence = builder.add_sequence_before(
        *b.block_parent, *b.block, b.block_parent->at(b.block_index).second.assignments(), b.block->debug_info()
    );

    // Dimensions, i.e., 1D, 2D, 3D, ...
    size_t dims = this->kernel_shape_.size();
    symbolic::MultiExpression out_shape = get_out_shape();
    types::Scalar indvar_type(types::PrimitiveType::Int64);

    auto in_channels = symbolic::div(this->shape_[1], this->group_);
    auto out_channels = symbolic::div(this->output_channels_, this->group_);

    // Add loop over batch size
    auto n_container = builder.find_new_name("_n");
    builder.add_container(n_container, indvar_type);
    auto n = symbolic::symbol(n_container);
    auto& loop_n = builder.add_map(
        new_sequence,
        n,
        symbolic::Lt(n, this->shape_[0]),
        symbolic::zero(),
        symbolic::add(n, symbolic::one()),
        ScheduleType_Sequential::create(),
        {},
        b.block->debug_info()
    );

    // Add loop over groups
    auto g_container = builder.find_new_name("_g");
    builder.add_container(g_container, indvar_type);
    auto g = symbolic::symbol(g_container);
    auto& loop_g = builder.add_map(
        loop_n.root(),
        g,
        symbolic::Lt(g, this->group_),
        symbolic::zero(),
        symbolic::add(g, symbolic::one()),
        ScheduleType_Sequential::create(),
        {},
        b.block->debug_info()
    );

    // Add patches container with malloc
    symbolic::Expression patches_size = in_channels;
    for (size_t i = 0; i < dims; i++) {
        patches_size = symbolic::mul(patches_size, symbolic::mul(this->kernel_shape_[i], out_shape[i]));
    }
    types::Pointer patches_type(base_type);
    auto patches_container = builder.find_new_name("_patches");
    builder.add_container(patches_container, patches_type);
    auto [patches_malloc_block, patches_malloc_node] = stdlib::add_malloc_block(
        builder,
        loop_g.root(),
        patches_container,
        symbolic::mul(patches_size, symbolic::size_of_type(base_type)),
        patches_type,
        this->debug_info()
    );

    // Add loop over channels
    structured_control_flow::Sequence* current_seq = &loop_g.root();
    auto c_container = builder.find_new_name("_c");
    builder.add_container(c_container, indvar_type);
    auto c = symbolic::symbol(c_container);
    auto& loop_c = builder.add_map(
        *current_seq,
        c,
        symbolic::Lt(c, in_channels),
        symbolic::zero(),
        symbolic::add(c, symbolic::one()),
        ScheduleType_Sequential::create(),
        {},
        b.block->debug_info()
    );
    current_seq = &loop_c.root();

    // Add loops over kernel shape
    symbolic::SymbolVec ks;
    ks.reserve(dims);
    for (size_t i = 0; i < dims; i++) {
        auto k_container = builder.find_new_name("_k");
        builder.add_container(k_container, indvar_type);
        auto k = symbolic::symbol(k_container);
        ks.push_back(k);
        auto& loop_k = builder.add_map(
            *current_seq,
            k,
            symbolic::Lt(k, this->kernel_shape_[i]),
            symbolic::zero(),
            symbolic::add(k, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            b.block->debug_info()
        );
        current_seq = &loop_k.root();
    }

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
            {},
            b.block->debug_info()
        );
        current_seq = &loop_o.root();
    }

    // Add if/else to stay in bounds for copying
    symbolic::MultiExpression is;
    is.reserve(dims);
    symbolic::Condition copy_condition = symbolic::__true__();
    symbolic::Condition zero_condition = symbolic::__false__();
    for (size_t i = 0; i < dims; i++) {
        auto i_expr = symbolic::
            add(symbolic::sub(symbolic::mul(os[i], this->strides_[i]), this->pads_[i]),
                symbolic::mul(ks[i], this->dilations_[i]));
        is.push_back(i_expr);
        copy_condition = symbolic::
            And(copy_condition,
                symbolic::And(symbolic::Lt(i_expr, this->shape_[i + 2]), symbolic::Ge(i_expr, symbolic::zero())));
        zero_condition = symbolic::
            Or(zero_condition,
               symbolic::Or(symbolic::Ge(i_expr, this->shape_[i + 2]), symbolic::Lt(i_expr, symbolic::zero())));
    }
    auto& branch = builder.add_if_else(*current_seq, {}, b.block->debug_info());
    auto& copy_case = builder.add_case(branch, copy_condition, b.block->debug_info());
    auto& zero_case = builder.add_case(branch, zero_condition, b.block->debug_info());

    // Determine patches subset & tensor type
    data_flow::Subset patches_subset;
    patches_subset.push_back(c);
    patches_subset.insert(patches_subset.end(), ks.begin(), ks.end());
    patches_subset.insert(patches_subset.end(), os.begin(), os.end());
    symbolic::MultiExpression patches_shape;
    patches_shape.push_back(in_channels);
    patches_shape.insert(patches_shape.end(), this->kernel_shape_.begin(), this->kernel_shape_.end());
    patches_shape.insert(patches_shape.end(), out_shape.begin(), out_shape.end());
    types::Tensor patches_tensor_type(base_type, patches_shape);

    // Determine subset for X
    data_flow::Subset subset_X;
    subset_X.push_back(n);
    subset_X.push_back(symbolic::add(symbolic::mul(in_channels, g), c));
    subset_X.insert(subset_X.end(), is.begin(), is.end());

    // Add copy from X to patches
    auto& copy_block = builder.add_block(copy_case, {}, b.block->debug_info());
    {
        auto& X_access = builder.add_access(copy_block, b.access_X->data(), b.access_X->debug_info());
        auto& patches_access = builder.add_access(copy_block, patches_container, this->debug_info());
        auto& tasklet =
            builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
        builder.add_computational_memlet(
            copy_block, X_access, tasklet, "_in", subset_X, b.iedge_X->base_type(), b.iedge_X->debug_info()
        );
        builder.add_computational_memlet(
            copy_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
        );
    }

    // Add zero assignment to patches
    auto& zero_block = builder.add_block(zero_case, {}, b.block->debug_info());
    {
        auto& constant_zero = builder.add_constant(zero_block, "0.0", base_type, this->debug_info());
        auto& patches_access = builder.add_access(zero_block, patches_container, this->debug_info());
        auto& tasklet =
            builder.add_tasklet(zero_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
        builder.add_computational_memlet(zero_block, constant_zero, tasklet, "_in", {}, base_type, this->debug_info());
        builder.add_computational_memlet(
            zero_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
        );
    }

    // Add reference to W
    auto ref_W_container = builder.find_new_name("_ref_W");
    types::Scalar ref_W_base_type(builder.subject().type(b.access_W->data()).primitive_type());
    types::Pointer ref_W_type(ref_W_base_type);
    builder.add_container(ref_W_container, ref_W_type);
    auto ref_W_subset = symbolic::mul(symbolic::mul(out_channels, g), in_channels);
    for (size_t i = 0; i < dims; i++) {
        ref_W_subset = symbolic::mul(ref_W_subset, this->kernel_shape_[i]);
    }
    auto& ref_W_block = builder.add_block(loop_g.root(), {}, b.block->debug_info());
    {
        auto& W_access = builder.add_access(ref_W_block, b.access_W->data(), b.access_W->debug_info());
        auto& ref_W_access = builder.add_access(ref_W_block, ref_W_container, b.access_W->debug_info());
        builder.add_reference_memlet(ref_W_block, W_access, ref_W_access, {ref_W_subset}, ref_W_type);
    }

    // Add reference to Y
    auto ref_Y_container = builder.find_new_name("_ref_Y");
    types::Scalar ref_Y_base_type(builder.subject().type(b.access_Y->data()).primitive_type());
    types::Pointer ref_Y_type(ref_Y_base_type);
    builder.add_container(ref_Y_container, ref_Y_type);
    auto ref_Y_subset = symbolic::add(symbolic::mul(this->output_channels_, n), symbolic::mul(out_channels, g));
    for (size_t i = 0; i < dims; i++) {
        ref_Y_subset = symbolic::mul(ref_Y_subset, out_shape[i]);
    }
    auto& ref_Y_block = builder.add_block(loop_g.root(), {}, b.block->debug_info());
    {
        auto& Y_access = builder.add_access(ref_Y_block, b.access_Y->data(), b.access_Y->debug_info());
        auto& ref_Y_access = builder.add_access(ref_Y_block, ref_Y_container, b.access_Y->debug_info());
        builder.add_reference_memlet(ref_Y_block, Y_access, ref_Y_access, {ref_Y_subset}, ref_Y_type);
    }

    // Add GEMM node
    auto& gemm_block = builder.add_block(loop_g.root(), {}, b.block->debug_info());
    {
        auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, this->debug_info());
        auto& beta = builder.add_constant(gemm_block, "0.0", base_type, this->debug_info());
        auto& ref_W_access = builder.add_access(gemm_block, ref_W_container, b.access_W->debug_info());
        auto& patches_access = builder.add_access(gemm_block, patches_container, this->debug_info());
        auto& ref_Y_access_in = builder.add_access(gemm_block, ref_Y_container, b.access_Y->debug_info());
        symbolic::Expression gemm_m = out_channels;
        symbolic::Expression gemm_n = symbolic::one();
        symbolic::Expression gemm_k = in_channels;
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
            gemm_m, // m
            gemm_n, // n
            gemm_k, // k
            gemm_k, // lda
            gemm_n, // ldb
            gemm_n // ldc
        );
        builder.add_computational_memlet(gemm_block, alpha, libnode, "__alpha", {}, base_type, this->debug_info());
        builder.add_computational_memlet(gemm_block, beta, libnode, "__beta", {}, base_type, this->debug_info());
        builder
            .add_computational_memlet(gemm_block, ref_W_access, libnode, "__A", {}, ref_W_type, b.iedge_W->debug_info());
        builder
            .add_computational_memlet(gemm_block, patches_access, libnode, "__B", {}, patches_type, this->debug_info());
        builder.add_computational_memlet(
            gemm_block, ref_Y_access_in, libnode, "__C", {}, ref_Y_type, b.iedge_Y->debug_info()
        );
    }

    // Add bias if available
    if (b.has_bias) {
        // Add loop over output channels
        auto l_container = builder.find_new_name("_l");
        builder.add_container(l_container, indvar_type);
        auto l = symbolic::symbol(l_container);
        auto& loop_l = builder.add_map(
            loop_g.root(),
            l,
            symbolic::Lt(l, out_channels),
            symbolic::zero(),
            symbolic::add(l, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            b.block->debug_info()
        );
        current_seq = &loop_l.root();

        // Add loops over output dimensions (again)
        for (size_t i = 0; i < dims; i++) {
            auto o_container = builder.find_new_name("_o");
            builder.add_container(o_container, indvar_type);
            auto o = symbolic::symbol(o_container);
            auto& loop_o = builder.add_map(
                *current_seq,
                o,
                symbolic::Lt(o, out_shape[i]),
                symbolic::zero(),
                symbolic::add(o, symbolic::one()),
                ScheduleType_Sequential::create(),
                {},
                b.block->debug_info()
            );
            current_seq = &loop_o.root();
            os[i] = o;
        }

        // Add bias to Y
        data_flow::Subset Y_subset;
        Y_subset.push_back(n);
        Y_subset.push_back(symbolic::add(symbolic::mul(out_channels, g), l));
        Y_subset.insert(Y_subset.end(), os.begin(), os.end());
        auto B_subset = symbolic::add(symbolic::mul(out_channels, g), l);
        auto& bias_block = builder.add_block(*current_seq, {}, b.block->debug_info());
        {
            auto& B_access = builder.add_access(bias_block, b.access_B->data(), b.access_B->debug_info());
            auto& Y_access_in = builder.add_access(bias_block, b.access_Y->data(), b.access_Y->debug_info());
            auto& Y_access_out = builder.add_access(bias_block, b.access_Y->data(), b.access_Y->debug_info());
            auto& tasklet =
                builder
                    .add_tasklet(bias_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info());
            builder.add_computational_memlet(
                bias_block, Y_access_in, tasklet, "_in1", Y_subset, b.iedge_Y->base_type(), this->debug_info()
            );
            builder.add_computational_memlet(
                bias_block, B_access, tasklet, "_in2", {B_subset}, b.iedge_B->base_type(), b.iedge_B->debug_info()
            );
            builder.add_computational_memlet(
                bias_block, tasklet, "_out", Y_access_out, Y_subset, b.iedge_Y->base_type(), b.iedge_Y->debug_info()
            );
        }
    }

    // Add free for patches container
    auto& patches_free_block = builder.add_block(loop_g.root(), {}, b.block->debug_info());
    {
        auto& patches_access_in = builder.add_access(patches_free_block, patches_container, this->debug_info());
        auto& libnode = builder.add_library_node<stdlib::FreeNode>(patches_free_block, this->debug_info());
        builder.add_computational_memlet(
            patches_free_block, patches_access_in, libnode, "_ptr", {}, patches_type, this->debug_info()
        );
    }

    // Clean up the original block
    builder.clear_code_node_legacy(*b.block, *this);
    // WARNING: this has been deallocated at this point!!
    builder.remove_child(*b.block_parent, b.block_index + 1);

    return true;
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
