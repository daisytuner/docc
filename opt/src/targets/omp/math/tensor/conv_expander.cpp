#include "sdfg/targets/omp/math/tensor/conv_expander.h"

#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/blas/blas_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/passes/expansion/lib_node_expander.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"
#include "sdfg/types/type.h"

namespace sdfg {
namespace omp {

passes::LibNodeExpander::ExpandOutcome OMPConvExpander::handle_expand(
    passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::ConvNode& node
) const {
    return handle_expand_im2col(context, block, node);
}

passes::LibNodeExpander::ExpandOutcome OMPConvExpander::handle_expand_im2col(
    passes::LibNodeExpander::ExpandContext& context, structured_control_flow::Block& block, math::tensor::ConvNode& node
) {
    auto& dfg = node.get_parent();
    math::tensor::ConvNode::ConvExpandPrerequisits b;
    if (!node.check_expandable(dfg, b)) {
        return context.unable();
    }

    // Determine BLAS precision
    types::Scalar base_type(node.primitive_type(dfg));
    math::blas::BLAS_Precision precision = node.get_blas_precision(base_type);
    if (precision == math::blas::BLAS_Precision::invalid) {
        return context.unable();
    }

    // Unable to perform im2col if the output tensor is not contiguous
    if (b.iedge_Y->base_type().type_id() != types::TypeID::Tensor) {
        return context.unable();
    }
    const types::Tensor& y_tensor = static_cast<const types::Tensor&>(b.iedge_Y->base_type());
    if (!y_tensor.is_contiguous()) {
        return context.unable();
    }

    using Use = passes::LibNodeExpander::InputUse;
    std::vector<Use> req_inputs = {Use::IndirectReadWrite, Use::IndirectRead, Use::IndirectRead};
    if (node.has_bias()) {
        req_inputs.push_back(Use::IndirectRead);
    }
    auto standalone = context.replacement_requires_access_nodes(req_inputs);

    if (!standalone) {
        return context.unable();
    }

    // Create new sequence for expansion
    auto& new_sequence = standalone->replace_with_sequence();
    auto& builder = standalone->builder();

    // Shapes
    const auto& shape = node.shape();
    const auto& kernel_shape = node.kernel_shape();
    const auto& strides = node.strides();
    const auto& pads = node.pads();
    const auto& dilations = node.dilations();
    const auto& output_channels = node.output_channels();
    const auto& group = node.group();
    symbolic::MultiExpression out_shape = node.get_out_shape();

    // Dimensions, i.e., 1D, 2D, 3D, ...
    size_t dims = kernel_shape.size();
    types::Scalar indvar_type(types::PrimitiveType::Int64);

    auto in_channels = symbolic::div(shape[1], group);
    auto out_channels = symbolic::div(output_channels, group);

    // Add loop over batch size
    auto n_container = builder.find_new_name("_n");
    builder.add_container(n_container, indvar_type);
    auto n = symbolic::symbol(n_container);
    auto& loop_n = builder.add_map(
        new_sequence,
        n,
        symbolic::Lt(n, shape[0]),
        symbolic::zero(),
        symbolic::add(n, symbolic::one()),
        ScheduleType_Sequential::create(),
        block.debug_info()
    );

    // Add loop over groups
    auto g_container = builder.find_new_name("_g");
    builder.add_container(g_container, indvar_type);
    auto g = symbolic::symbol(g_container);
    auto& loop_g = builder.add_map(
        loop_n.root(),
        g,
        symbolic::Lt(g, group),
        symbolic::zero(),
        symbolic::add(g, symbolic::one()),
        ScheduleType_Sequential::create(),
        block.debug_info()
    );

    // Add patches container with malloc
    symbolic::Expression patches_size = in_channels;
    for (size_t i = 0; i < dims; i++) {
        patches_size = symbolic::mul(patches_size, symbolic::mul(kernel_shape[i], out_shape[i]));
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
        node.debug_info()
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
        block.debug_info()
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
            symbolic::Lt(k, kernel_shape[i]),
            symbolic::zero(),
            symbolic::add(k, symbolic::one()),
            ScheduleType_Sequential::create(),
            block.debug_info()
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
            block.debug_info()
        );
        current_seq = &loop_o.root();
    }

    // Add if/else to stay in bounds for copying
    symbolic::MultiExpression is;
    is.reserve(dims);
    symbolic::Condition copy_condition = symbolic::__true__();
    symbolic::Condition zero_condition = symbolic::__false__();
    for (size_t i = 0; i < dims; i++) {
        auto i_expr =
            symbolic::add(symbolic::sub(symbolic::mul(os[i], strides[i]), pads[i]), symbolic::mul(ks[i], dilations[i]));
        is.push_back(i_expr);
        copy_condition = symbolic::
            And(copy_condition,
                symbolic::And(symbolic::Lt(i_expr, shape[i + 2]), symbolic::Ge(i_expr, symbolic::zero())));
        zero_condition = symbolic::
            Or(zero_condition,
               symbolic::Or(symbolic::Ge(i_expr, shape[i + 2]), symbolic::Lt(i_expr, symbolic::zero())));
    }
    auto& branch = builder.add_if_else(*current_seq, block.debug_info());
    auto& copy_case = builder.add_case(branch, copy_condition, block.debug_info());
    auto& zero_case = builder.add_case(branch, zero_condition, block.debug_info());

    // Determine patches subset & tensor type
    data_flow::Subset patches_subset;
    patches_subset.push_back(c);
    patches_subset.insert(patches_subset.end(), ks.begin(), ks.end());
    patches_subset.insert(patches_subset.end(), os.begin(), os.end());
    symbolic::MultiExpression patches_shape;
    patches_shape.push_back(in_channels);
    patches_shape.insert(patches_shape.end(), kernel_shape.begin(), kernel_shape.end());
    patches_shape.insert(patches_shape.end(), out_shape.begin(), out_shape.end());
    types::Tensor patches_tensor_type(base_type, patches_shape);

    // Determine subset for X
    data_flow::Subset subset_X;
    subset_X.push_back(n);
    subset_X.push_back(symbolic::add(symbolic::mul(in_channels, g), c));
    subset_X.insert(subset_X.end(), is.begin(), is.end());

    // Add copy from X to patches
    auto& copy_block = builder.add_block(copy_case, {}, block.debug_info());
    {
        auto& X_access = standalone->add_indirect_read_access(copy_block, math::tensor::ConvNode::X_INPUT_IDX);
        auto& patches_access = builder.add_access(copy_block, patches_container, node.debug_info());
        auto& tasklet =
            builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node.debug_info());
        builder.add_computational_memlet(
            copy_block, X_access, tasklet, "_in", subset_X, b.iedge_X->base_type(), b.iedge_X->debug_info()
        );
        builder.add_computational_memlet(
            copy_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, node.debug_info()
        );
    }

    // Add zero assignment to patches
    auto& zero_block = builder.add_block(zero_case, {}, block.debug_info());
    {
        auto& constant_zero = builder.add_constant(zero_block, "0.0", base_type, node.debug_info());
        auto& patches_access = builder.add_access(zero_block, patches_container, node.debug_info());
        auto& tasklet =
            builder.add_tasklet(zero_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node.debug_info());
        builder.add_computational_memlet(zero_block, constant_zero, tasklet, "_in", {}, base_type, node.debug_info());
        builder.add_computational_memlet(
            zero_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, node.debug_info()
        );
    }

    // Add reference to W
    auto ref_W_container = builder.find_new_name("_ref_W");
    auto& ref_W_block = builder.add_block(loop_g.root(), {}, block.debug_info());
    auto& W_access = standalone->add_scalar_input_access(ref_W_block, math::tensor::ConvNode::W_INPUT_IDX);
    types::Scalar ref_W_base_type(builder.subject().type(W_access.data()).primitive_type());
    types::Pointer ref_W_type(ref_W_base_type);
    builder.add_container(ref_W_container, ref_W_type);
    auto ref_W_subset = symbolic::mul(symbolic::mul(out_channels, g), in_channels);
    for (size_t i = 0; i < dims; i++) {
        ref_W_subset = symbolic::mul(ref_W_subset, kernel_shape[i]);
    }
    {
        auto& ref_W_access = builder.add_access(ref_W_block, ref_W_container, W_access.debug_info());
        builder.add_reference_memlet(ref_W_block, W_access, ref_W_access, {ref_W_subset}, ref_W_type);
    }

    // Add reference to Y
    auto& ref_Y_block = builder.add_block(loop_g.root(), {}, block.debug_info());
    auto& Y_access = standalone->add_scalar_input_access(ref_Y_block, math::tensor::ConvNode::Y_INPUT_IDX);
    auto ref_Y_container = builder.find_new_name("_ref_Y");
    types::Scalar ref_Y_base_type(builder.subject().type(Y_access.data()).primitive_type());
    types::Pointer ref_Y_type(ref_Y_base_type);
    builder.add_container(ref_Y_container, ref_Y_type);
    auto ref_Y_subset = symbolic::add(symbolic::mul(output_channels, n), symbolic::mul(out_channels, g));
    for (size_t i = 0; i < dims; i++) {
        ref_Y_subset = symbolic::mul(ref_Y_subset, out_shape[i]);
    }
    {
        auto& ref_Y_access = builder.add_access(ref_Y_block, ref_Y_container, Y_access.debug_info());
        builder.add_reference_memlet(ref_Y_block, Y_access, ref_Y_access, {ref_Y_subset}, ref_Y_type);
    }

    // Add GEMM node
    auto& gemm_block = builder.add_block(loop_g.root(), {}, block.debug_info());
    {
        auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, node.debug_info());
        auto& beta = builder.add_constant(gemm_block, "0.0", base_type, node.debug_info());
        auto& ref_W_access = builder.add_access(gemm_block, ref_W_container, W_access.debug_info());
        auto& patches_access = builder.add_access(gemm_block, patches_container, node.debug_info());
        auto& ref_Y_access_in = builder.add_access(gemm_block, ref_Y_container, Y_access.debug_info());
        symbolic::Expression gemm_m = out_channels;
        symbolic::Expression gemm_n = symbolic::one();
        symbolic::Expression gemm_k = in_channels;
        for (size_t i = 0; i < dims; i++) {
            gemm_n = symbolic::mul(gemm_n, out_shape[i]);
            gemm_k = symbolic::mul(gemm_k, kernel_shape[i]);
        }
        auto& libnode = builder.add_library_node<math::blas::GEMMNode>(
            gemm_block,
            node.debug_info(),
            math::blas::ImplementationType_BLAS,
            precision, // precision
            math::blas::BLAS_Layout::RowMajor, // layout
            math::blas::BLAS_Transpose::No, // transA
            math::blas::BLAS_Transpose::No, // transB
            gemm_m, // m
            gemm_n, // n
            gemm_k, // k
            gemm_k, // lda
            gemm_n, // ldb
            gemm_n // ldc
        );
        builder.add_computational_memlet(gemm_block, alpha, libnode, "__alpha", {}, base_type, node.debug_info());
        builder.add_computational_memlet(gemm_block, beta, libnode, "__beta", {}, base_type, node.debug_info());
        builder
            .add_computational_memlet(gemm_block, ref_W_access, libnode, "__A", {}, ref_W_type, b.iedge_W->debug_info());
        builder
            .add_computational_memlet(gemm_block, patches_access, libnode, "__B", {}, patches_type, node.debug_info());
        builder.add_computational_memlet(
            gemm_block, ref_Y_access_in, libnode, "__C", {}, ref_Y_type, b.iedge_Y->debug_info()
        );
    }

    // Add bias if available
    if (node.has_bias()) {
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
            block.debug_info()
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
                block.debug_info()
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
        auto& bias_block = builder.add_block(*current_seq, {}, block.debug_info());
        {
            auto& B_access = standalone->add_indirect_read_access(bias_block, math::tensor::ConvNode::B_INPUT_IDX);
            auto& Y_access_in = standalone->add_indirect_read_access(bias_block, math::tensor::ConvNode::Y_INPUT_IDX);
            auto& Y_access_out = standalone->add_indirect_write_access(bias_block, math::tensor::ConvNode::Y_INPUT_IDX);
            auto& tasklet =
                builder
                    .add_tasklet(bias_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, node.debug_info());
            builder.add_computational_memlet(
                bias_block, Y_access_in, tasklet, "_in1", Y_subset, b.iedge_Y->base_type(), node.debug_info()
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
    auto [patches_free_block, patches_free_node] =
        stdlib::add_free_block(builder, loop_g.root(), patches_container, patches_type, node.debug_info());

    return standalone->successfully_expanded();
}

} // namespace omp
} // namespace sdfg
