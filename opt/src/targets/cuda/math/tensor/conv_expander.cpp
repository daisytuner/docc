#include "sdfg/targets/cuda/math/tensor/conv_expander.h"
#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/math/blas/gemm_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"
#include "sdfg/types/tensor.h"

namespace sdfg {
namespace offloading {

bool CudaConvExpander::expand(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    return expand_conv(builder, analysis_manager, node_);
}

bool CudaConvExpander::expand_conv(
    builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager, math::tensor::ConvNode& node
) {
    auto& dfg = node.get_parent();
    math::tensor::ConvNode::ConvExpandPrerequisits b;
    if (!node.check_expandable(dfg, analysis_manager, b)) {
        return false;
    }

    if (!symbolic::eq(node.group(), symbolic::one())) {
        // If there are no groups (i.e., group == 1), then we can do im2row with one GEMM.
        // Else, no CUDA specific expand is needed
        return false;
    }


    types::Scalar base_type(node.primitive_type(dfg));
    math::blas::BLAS_Precision precision = node.get_blas_precision(base_type);

    // Create new sequence for expansion
    auto& new_sequence = builder.add_sequence_before(
        *b.block_parent, *b.block, b.block_parent->at(b.block_index).second.assignments(), b.block->debug_info()
    );

    // Dimensions, i.e., 1D, 2D, 3D, ...
    size_t dims = node.kernel_shape().size();
    symbolic::MultiExpression out_shape = node.get_out_shape();
    types::Scalar indvar_type(types::PrimitiveType::Int64);


    /* ===== No groups ====================================================================== */

    // Add patches container with malloc
    symbolic::Expression patches_size = symbolic::mul(node.shape()[0], node.shape()[1]);
    for (size_t i = 0; i < dims; i++) {
        patches_size = symbolic::mul(patches_size, symbolic::mul(node.kernel_shape()[i], out_shape[i]));
    }
    types::Pointer patches_type(base_type);
    auto patches_container = builder.find_new_name("_patches");
    builder.add_container(patches_container, patches_type);
    auto [patches_malloc_block, patches_malloc_node] = stdlib::add_malloc_block(
        builder,
        new_sequence,
        patches_container,
        symbolic::mul(patches_size, symbolic::size_of_type(base_type)),
        patches_type,
        node.debug_info()
    );

    // Add malloc for temporary GEMM output
    symbolic::Expression tmp_Y_size = symbolic::mul(node.output_channels(), node.shape()[0]);
    for (size_t i = 0; i < dims; i++) {
        tmp_Y_size = symbolic::mul(tmp_Y_size, out_shape[i]);
    }
    auto tmp_Y_container = builder.find_new_name("_tmp_Y");
    types::Scalar tmp_Y_base_type(builder.subject().type(b.access_Y->data()).primitive_type());
    types::Pointer tmp_Y_type(tmp_Y_base_type);
    builder.add_container(tmp_Y_container, tmp_Y_type);
    auto [tmp_Y_malloc_block, tmp_Y_malloc_node] = stdlib::add_malloc_block(
        builder,
        new_sequence,
        tmp_Y_container,
        symbolic::mul(tmp_Y_size, symbolic::size_of_type(tmp_Y_base_type)),
        tmp_Y_type,
        node.debug_info()
    );

    // Add loop over batch size
    auto n_container = builder.find_new_name("_n");
    builder.add_container(n_container, indvar_type);
    auto n = symbolic::symbol(n_container);
    auto& loop_n = builder.add_map(
        new_sequence,
        n,
        symbolic::Lt(n, node.shape()[0]),
        symbolic::zero(),
        symbolic::add(n, symbolic::one()),
        ScheduleType_Sequential::create(),
        {},
        b.block->debug_info()
    );
    structured_control_flow::Sequence* current_seq = &loop_n.root();

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

    // Add loop over channels
    auto c_container = builder.find_new_name("_c");
    builder.add_container(c_container, indvar_type);
    auto c = symbolic::symbol(c_container);
    auto& loop_c = builder.add_map(
        *current_seq,
        c,
        symbolic::Lt(c, node.shape()[1]),
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
            symbolic::Lt(k, node.kernel_shape()[i]),
            symbolic::zero(),
            symbolic::add(k, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            b.block->debug_info()
        );
        current_seq = &loop_k.root();
    }

    // Add if/else to stay in bounds for copying
    symbolic::MultiExpression is;
    is.reserve(dims);
    symbolic::Condition copy_condition = symbolic::__true__();
    symbolic::Condition zero_condition = symbolic::__false__();
    for (size_t i = 0; i < dims; i++) {
        auto i_expr = symbolic::
            add(symbolic::sub(symbolic::mul(os[i], node.strides()[i]), node.pads()[i]),
                symbolic::mul(ks[i], node.dilations()[i]));
        is.push_back(i_expr);
        copy_condition = symbolic::
            And(copy_condition,
                symbolic::And(symbolic::Lt(i_expr, node.shape()[i + 2]), symbolic::Ge(i_expr, symbolic::zero())));
        zero_condition = symbolic::
            Or(zero_condition,
               symbolic::Or(symbolic::Ge(i_expr, node.shape()[i + 2]), symbolic::Lt(i_expr, symbolic::zero())));
    }
    auto& branch = builder.add_if_else(*current_seq, {}, b.block->debug_info());
    auto& copy_case = builder.add_case(branch, copy_condition, b.block->debug_info());
    auto& zero_case = builder.add_case(branch, zero_condition, b.block->debug_info());

    // Determine patches subset & tensor type
    data_flow::Subset patches_subset;
    patches_subset.push_back(n);
    patches_subset.insert(patches_subset.end(), os.begin(), os.end());
    patches_subset.push_back(c);
    patches_subset.insert(patches_subset.end(), ks.begin(), ks.end());
    symbolic::MultiExpression patches_shape;
    patches_shape.push_back(node.shape()[0]);
    patches_shape.insert(patches_shape.end(), out_shape.begin(), out_shape.end());
    patches_shape.push_back(node.shape()[1]);
    patches_shape.insert(patches_shape.end(), node.kernel_shape().begin(), node.kernel_shape().end());
    types::Tensor patches_tensor_type(base_type, patches_shape);

    // Determine subset for X
    data_flow::Subset subset_X;
    subset_X.push_back(n);
    subset_X.push_back(c);
    subset_X.insert(subset_X.end(), is.begin(), is.end());

    // Add copy from X to patches
    auto& copy_block = builder.add_block(copy_case, {}, b.block->debug_info());
    {
        auto& X_access = builder.add_access(copy_block, b.access_X->data(), b.access_X->debug_info());
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
    auto& zero_block = builder.add_block(zero_case, {}, b.block->debug_info());
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

    // Add GEMM node
    auto& gemm_block = builder.add_block(new_sequence, {}, b.block->debug_info());
    {
        auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, node.debug_info());
        auto& beta = builder.add_constant(gemm_block, "0.0", base_type, node.debug_info());
        symbolic::Expression gemm_m = node.output_channels();
        symbolic::Expression gemm_n = node.shape()[0];
        symbolic::Expression gemm_k = node.shape()[1];
        for (size_t i = 0; i < dims; i++) {
            gemm_n = symbolic::mul(gemm_n, out_shape[i]);
            gemm_k = symbolic::mul(gemm_k, node.kernel_shape()[i]);
        }
        auto& libnode = math::blas::add_gemm_node(
            builder,
            gemm_block,
            b.access_W->data(),
            patches_container,
            tmp_Y_container,
            alpha,
            beta,
            precision,
            math::blas::BLAS_Layout::RowMajor, // layout
            math::blas::BLAS_Transpose::No, // transA
            math::blas::BLAS_Transpose::Trans, // transB
            gemm_m, // m
            gemm_n, // n
            gemm_k, // k
            gemm_k, // lda
            gemm_k, // ldb
            gemm_n, // ldc
            types::Pointer(types::Scalar(b.iedge_W->base_type().primitive_type())),
            patches_type,
            tmp_Y_type,
            base_type,
            node.debug_info(),
            b.access_W->debug_info(),
            node.debug_info(),
            b.access_Y->debug_info(),
            b.iedge_W->debug_info(),
            node.debug_info(),
            b.iedge_Y->debug_info(),
            math::blas::ImplementationType_BLAS
        );
    }

    // Add loop over batch size (again)
    auto& loop_n_2 = builder.add_map(
        new_sequence,
        n,
        symbolic::Lt(n, node.shape()[0]),
        symbolic::zero(),
        symbolic::add(n, symbolic::one()),
        ScheduleType_Sequential::create(),
        {},
        b.block->debug_info()
    );
    current_seq = &loop_n_2.root();

    // Add loop over output channels
    auto l_container = builder.find_new_name("_l");
    builder.add_container(l_container, indvar_type);
    auto l = symbolic::symbol(l_container);
    auto& loop_l = builder.add_map(
        *current_seq,
        l,
        symbolic::Lt(l, node.output_channels()),
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

    // Add transposed copy from temporary GEMM output to Y + add bias if available
    data_flow::Subset tmp_Y_subset;
    tmp_Y_subset.push_back(l);
    tmp_Y_subset.push_back(n);
    tmp_Y_subset.insert(tmp_Y_subset.end(), os.begin(), os.end());
    symbolic::MultiExpression tmp_Y_shape;
    tmp_Y_shape.push_back(node.output_channels());
    tmp_Y_shape.push_back(node.shape()[0]);
    tmp_Y_shape.insert(tmp_Y_shape.end(), out_shape.begin(), out_shape.end());
    types::Tensor tmp_Y_tensor_type(tmp_Y_base_type, tmp_Y_shape);
    data_flow::Subset Y_subset;
    Y_subset.push_back(n);
    Y_subset.push_back(l);
    Y_subset.insert(Y_subset.end(), os.begin(), os.end());
    auto& transpose_block = builder.add_block(*current_seq, {}, b.block->debug_info());
    if (b.has_bias) {
        auto& tmp_Y_access = builder.add_access(transpose_block, tmp_Y_container, node.debug_info());
        auto& B_access = builder.add_access(transpose_block, b.access_B->data(), b.access_B->debug_info());
        auto& Y_access = builder.add_access(transpose_block, b.access_Y->data(), b.access_Y->debug_info());
        auto& tasklet =
            builder
                .add_tasklet(transpose_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, node.debug_info());
        builder.add_computational_memlet(
            transpose_block, tmp_Y_access, tasklet, "_in1", tmp_Y_subset, tmp_Y_tensor_type, node.debug_info()
        );
        builder.add_computational_memlet(
            transpose_block, B_access, tasklet, "_in2", {l}, b.iedge_B->base_type(), b.iedge_B->debug_info()
        );
        builder.add_computational_memlet(
            transpose_block, tasklet, "_out", Y_access, Y_subset, b.iedge_Y->base_type(), b.iedge_Y->debug_info()
        );
    } else {
        auto& tmp_Y_access = builder.add_access(transpose_block, tmp_Y_container, node.debug_info());
        auto& Y_access = builder.add_access(transpose_block, b.access_Y->data(), b.access_Y->debug_info());
        auto& tasklet =
            builder.add_tasklet(transpose_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node.debug_info());
        builder.add_computational_memlet(
            transpose_block, tmp_Y_access, tasklet, "_in", tmp_Y_subset, tmp_Y_tensor_type, node.debug_info()
        );
        builder.add_computational_memlet(
            transpose_block, tasklet, "_out", Y_access, Y_subset, b.iedge_Y->base_type(), b.iedge_Y->debug_info()
        );
    }

    // Add free for patches container
    auto [patches_free_block, patches_free_node] =
        stdlib::add_free_block(builder, new_sequence, patches_container, patches_type, node.debug_info());

    // Add free for temporary GEMM output
    auto [tmp_Y_free_block, tmp_Y_free_node] =
        stdlib::add_free_block(builder, new_sequence, tmp_Y_container, tmp_Y_type, node.debug_info());

    // Clean up the original block
    builder.clear_code_node_legacy(*b.block, node);
    builder.remove_child(*b.block_parent, b.block_index + 1);

    return true;
}
} // namespace offloading
} // namespace sdfg
