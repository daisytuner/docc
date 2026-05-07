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
    // Custom CUDA expansion for convolution
    auto& dfg = node_.get_parent();
    if ((dfg.nodes().size() != 4 || dfg.edges().size() != 3) && (dfg.nodes().size() != 5 || dfg.edges().size() != 4)) {
        return false;
    }

    // Get edges
    auto iedges = dfg.in_edges_by_connector(node_);
    auto oedges = dfg.out_edges_by_connector(node_);
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
    math::blas::BLAS_Precision precision;
    types::Scalar base_type(node_.primitive_type(dfg));
    switch (base_type.primitive_type()) {
        case types::PrimitiveType::Half:
            precision = math::blas::BLAS_Precision::h;
            break;
        case types::PrimitiveType::Float:
            precision = math::blas::BLAS_Precision::s;
            break;
        case types::PrimitiveType::Double:
            precision = math::blas::BLAS_Precision::d;
            break;
        default:
            return false;
    }

    // Create new sequence for expansion
    auto& new_sequence = builder.add_sequence_before(
        *block_parent, *block, block_parent->at(block_index).second.assignments(), block->debug_info()
    );

    // Dimensions, i.e., 1D, 2D, 3D, ...
    size_t dims = node_.kernel_shape().size();
    symbolic::MultiExpression out_shape;
    out_shape.reserve(dims);
    // out_shape[i] = (shape[i + 2] + pads[i] + pads[dims + i] - dilations[i] * (kernel_shape[i] - 1) - 1)
    //                 / strides[i] + 1
    for (size_t i = 0; i < dims; i++) {
        out_shape.push_back(symbolic::add(
            symbolic::div(
                symbolic::
                    sub(symbolic::sub(
                            symbolic::add(node_.shape()[i + 2], symbolic::add(node_.pads()[i], node_.pads()[dims + i])),
                            symbolic::mul(node_.dilations()[i], symbolic::sub(node_.kernel_shape()[i], symbolic::one()))
                        ),
                        symbolic::one()),
                node_.strides()[i]
            ),
            symbolic::one()
        ));
    }
    types::Scalar indvar_type(types::PrimitiveType::Int64);

    // If there are no groups (i.e., group == 1), then we can do im2row with one GEMM.
    // Else, we do naïve im2col with multiple GEMM's.
    if (symbolic::eq(node_.group(), symbolic::one())) {
        /* ===== No groups ====================================================================== */

        // Add patches container with malloc
        symbolic::Expression patches_size = symbolic::mul(node_.shape()[0], node_.shape()[1]);
        for (size_t i = 0; i < dims; i++) {
            patches_size = symbolic::mul(patches_size, symbolic::mul(node_.kernel_shape()[i], out_shape[i]));
        }
        types::Pointer patches_type(base_type);
        auto patches_container = builder.find_new_name("_patches");
        builder.add_container(patches_container, patches_type);
        auto& patches_malloc_block = builder.add_block(new_sequence, {}, block->debug_info());
        {
            auto& patches_access = builder.add_access(patches_malloc_block, patches_container, node_.debug_info());
            auto& libnode = builder.add_library_node<stdlib::MallocNode>(
                patches_malloc_block, node_.debug_info(), symbolic::mul(patches_size, symbolic::size_of_type(base_type))
            );
            builder.add_computational_memlet(
                patches_malloc_block, libnode, "_ret", patches_access, {}, patches_type, node_.debug_info()
            );
        }

        // Add malloc for temporary GEMM output
        symbolic::Expression tmp_Y_size = symbolic::mul(node_.output_channels(), node_.shape()[0]);
        for (size_t i = 0; i < dims; i++) {
            tmp_Y_size = symbolic::mul(tmp_Y_size, out_shape[i]);
        }
        auto tmp_Y_container = builder.find_new_name("_tmp_Y");
        types::Scalar tmp_Y_base_type(builder.subject().type(access_Y->data()).primitive_type());
        types::Pointer tmp_Y_type(tmp_Y_base_type);
        builder.add_container(tmp_Y_container, tmp_Y_type);
        auto& tmp_Y_malloc_block = builder.add_block(new_sequence, {}, block->debug_info());
        {
            auto& tmp_Y_access = builder.add_access(tmp_Y_malloc_block, tmp_Y_container, node_.debug_info());
            auto& libnode = builder.add_library_node<stdlib::MallocNode>(
                tmp_Y_malloc_block,
                node_.debug_info(),
                symbolic::mul(tmp_Y_size, symbolic::size_of_type(tmp_Y_base_type))
            );
            builder.add_computational_memlet(
                tmp_Y_malloc_block, libnode, "_ret", tmp_Y_access, {}, tmp_Y_type, node_.debug_info()
            );
        }

        // Add loop over batch size
        auto n_container = builder.find_new_name("_n");
        builder.add_container(n_container, indvar_type);
        auto n = symbolic::symbol(n_container);
        auto& loop_n = builder.add_map(
            new_sequence,
            n,
            symbolic::Lt(n, node_.shape()[0]),
            symbolic::zero(),
            symbolic::add(n, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
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
                block->debug_info()
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
            symbolic::Lt(c, node_.shape()[1]),
            symbolic::zero(),
            symbolic::add(c, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
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
                symbolic::Lt(k, node_.kernel_shape()[i]),
                symbolic::zero(),
                symbolic::add(k, symbolic::one()),
                ScheduleType_Sequential::create(),
                {},
                block->debug_info()
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
                add(symbolic::sub(symbolic::mul(os[i], node_.strides()[i]), node_.pads()[i]),
                    symbolic::mul(ks[i], node_.dilations()[i]));
            is.push_back(i_expr);
            copy_condition = symbolic::
                And(copy_condition,
                    symbolic::And(symbolic::Lt(i_expr, node_.shape()[i + 2]), symbolic::Ge(i_expr, symbolic::zero())));
            zero_condition = symbolic::
                Or(zero_condition,
                   symbolic::Or(symbolic::Ge(i_expr, node_.shape()[i + 2]), symbolic::Lt(i_expr, symbolic::zero())));
        }
        auto& branch = builder.add_if_else(*current_seq, {}, block->debug_info());
        auto& copy_case = builder.add_case(branch, copy_condition, block->debug_info());
        auto& zero_case = builder.add_case(branch, zero_condition, block->debug_info());

        // Determine patches subset & tensor type
        data_flow::Subset patches_subset;
        patches_subset.push_back(n);
        patches_subset.insert(patches_subset.end(), os.begin(), os.end());
        patches_subset.push_back(c);
        patches_subset.insert(patches_subset.end(), ks.begin(), ks.end());
        symbolic::MultiExpression patches_shape;
        patches_shape.push_back(node_.shape()[0]);
        patches_shape.insert(patches_shape.end(), out_shape.begin(), out_shape.end());
        patches_shape.push_back(node_.shape()[1]);
        patches_shape.insert(patches_shape.end(), node_.kernel_shape().begin(), node_.kernel_shape().end());
        types::Tensor patches_tensor_type(base_type, patches_shape);

        // Determine subset for X
        data_flow::Subset subset_X;
        subset_X.push_back(n);
        subset_X.push_back(c);
        subset_X.insert(subset_X.end(), is.begin(), is.end());

        // Add copy from X to patches
        auto& copy_block = builder.add_block(copy_case, {}, block->debug_info());
        {
            auto& X_access = builder.add_access(copy_block, access_X->data(), access_X->debug_info());
            auto& patches_access = builder.add_access(copy_block, patches_container, node_.debug_info());
            auto& tasklet =
                builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node_.debug_info());
            builder.add_computational_memlet(
                copy_block, X_access, tasklet, "_in", subset_X, iedge_X->base_type(), iedge_X->debug_info()
            );
            builder.add_computational_memlet(
                copy_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, node_.debug_info()
            );
        }

        // Add zero assignment to patches
        auto& zero_block = builder.add_block(zero_case, {}, block->debug_info());
        {
            auto& constant_zero = builder.add_constant(zero_block, "0.0", base_type, node_.debug_info());
            auto& patches_access = builder.add_access(zero_block, patches_container, node_.debug_info());
            auto& tasklet =
                builder.add_tasklet(zero_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node_.debug_info());
            builder
                .add_computational_memlet(zero_block, constant_zero, tasklet, "_in", {}, base_type, node_.debug_info());
            builder.add_computational_memlet(
                zero_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, node_.debug_info()
            );
        }

        // Add GEMM node
        auto& gemm_block = builder.add_block(new_sequence, {}, block->debug_info());
        {
            auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, node_.debug_info());
            auto& beta = builder.add_constant(gemm_block, "0.0", base_type, node_.debug_info());
            auto& W_access = builder.add_access(gemm_block, access_W->data(), access_W->debug_info());
            auto& patches_access = builder.add_access(gemm_block, patches_container, node_.debug_info());
            auto& tmp_Y_access_in = builder.add_access(gemm_block, tmp_Y_container, access_Y->debug_info());
            auto& tmp_Y_access_out = builder.add_access(gemm_block, tmp_Y_container, access_Y->debug_info());
            symbolic::Expression gemm_m = node_.output_channels();
            symbolic::Expression gemm_n = node_.shape()[0];
            symbolic::Expression gemm_k = node_.shape()[1];
            for (size_t i = 0; i < dims; i++) {
                gemm_n = symbolic::mul(gemm_n, out_shape[i]);
                gemm_k = symbolic::mul(gemm_k, node_.kernel_shape()[i]);
            }
            auto& libnode = builder.add_library_node<math::blas::GEMMNode>(
                gemm_block,
                node_.debug_info(),
                math::blas::ImplementationType_BLAS,
                precision, // precision
                math::blas::BLAS_Layout::RowMajor, // layout
                math::blas::BLAS_Transpose::No, // transA
                math::blas::BLAS_Transpose::Trans, // transB
                gemm_m, // m
                gemm_n, // n
                gemm_k, // k
                gemm_k, // lda
                gemm_k, // ldb
                gemm_n // ldc
            );
            builder.add_computational_memlet(gemm_block, alpha, libnode, "__alpha", {}, base_type, node_.debug_info());
            builder.add_computational_memlet(gemm_block, beta, libnode, "__beta", {}, base_type, node_.debug_info());
            builder.add_computational_memlet(
                gemm_block,
                W_access,
                libnode,
                "__A",
                {},
                types::Pointer(types::Scalar(iedge_W->base_type().primitive_type())),
                iedge_W->debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, patches_access, libnode, "__B", {}, patches_type, node_.debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, tmp_Y_access_in, libnode, "__C", {}, tmp_Y_type, oedge_Y->debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, libnode, "__C", tmp_Y_access_out, {}, tmp_Y_type, oedge_Y->debug_info()
            );
        }

        // Add loop over batch size (again)
        auto& loop_n_2 = builder.add_map(
            new_sequence,
            n,
            symbolic::Lt(n, node_.shape()[0]),
            symbolic::zero(),
            symbolic::add(n, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
        );
        current_seq = &loop_n_2.root();

        // Add loop over output channels
        auto l_container = builder.find_new_name("_l");
        builder.add_container(l_container, indvar_type);
        auto l = symbolic::symbol(l_container);
        auto& loop_l = builder.add_map(
            *current_seq,
            l,
            symbolic::Lt(l, node_.output_channels()),
            symbolic::zero(),
            symbolic::add(l, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
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
                block->debug_info()
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
        tmp_Y_shape.push_back(node_.output_channels());
        tmp_Y_shape.push_back(node_.shape()[0]);
        tmp_Y_shape.insert(tmp_Y_shape.end(), out_shape.begin(), out_shape.end());
        types::Tensor tmp_Y_tensor_type(tmp_Y_base_type, tmp_Y_shape);
        data_flow::Subset Y_subset;
        Y_subset.push_back(n);
        Y_subset.push_back(l);
        Y_subset.insert(Y_subset.end(), os.begin(), os.end());
        auto& transpose_block = builder.add_block(*current_seq, {}, block->debug_info());
        if (has_bias) {
            auto& tmp_Y_access = builder.add_access(transpose_block, tmp_Y_container, node_.debug_info());
            auto& B_access = builder.add_access(transpose_block, access_B->data(), access_B->debug_info());
            auto& Y_access = builder.add_access(transpose_block, access_Y->data(), access_Y->debug_info());
            auto& tasklet = builder.add_tasklet(
                transpose_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, node_.debug_info()
            );
            builder.add_computational_memlet(
                transpose_block, tmp_Y_access, tasklet, "_in1", tmp_Y_subset, tmp_Y_tensor_type, node_.debug_info()
            );
            builder.add_computational_memlet(
                transpose_block, B_access, tasklet, "_in2", {l}, iedge_B->base_type(), iedge_B->debug_info()
            );
            builder.add_computational_memlet(
                transpose_block, tasklet, "_out", Y_access, Y_subset, oedge_Y->base_type(), oedge_Y->debug_info()
            );
        } else {
            auto& tmp_Y_access = builder.add_access(transpose_block, tmp_Y_container, node_.debug_info());
            auto& Y_access = builder.add_access(transpose_block, access_Y->data(), access_Y->debug_info());
            auto& tasklet =
                builder
                    .add_tasklet(transpose_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node_.debug_info());
            builder.add_computational_memlet(
                transpose_block, tmp_Y_access, tasklet, "_in", tmp_Y_subset, tmp_Y_tensor_type, node_.debug_info()
            );
            builder.add_computational_memlet(
                transpose_block, tasklet, "_out", Y_access, Y_subset, oedge_Y->base_type(), oedge_Y->debug_info()
            );
        }

        // Add free for patches container
        auto& patches_free_block = builder.add_block(new_sequence, {}, block->debug_info());
        {
            auto& patches_access_in = builder.add_access(patches_free_block, patches_container, node_.debug_info());
            auto& patches_access_out = builder.add_access(patches_free_block, patches_container, node_.debug_info());
            auto& libnode = builder.add_library_node<stdlib::FreeNode>(patches_free_block, node_.debug_info());
            builder.add_computational_memlet(
                patches_free_block, patches_access_in, libnode, "_ptr", {}, patches_type, node_.debug_info()
            );
            builder.add_computational_memlet(
                patches_free_block, libnode, "_ptr", patches_access_out, {}, patches_type, node_.debug_info()
            );
        }

        // Add free for temporary GEMM output
        auto& tmp_Y_free_block = builder.add_block(new_sequence, {}, block->debug_info());
        {
            auto& tmp_Y_access_in = builder.add_access(tmp_Y_free_block, tmp_Y_container, node_.debug_info());
            auto& tmp_Y_access_out = builder.add_access(tmp_Y_free_block, tmp_Y_container, node_.debug_info());
            auto& libnode = builder.add_library_node<stdlib::FreeNode>(tmp_Y_free_block, node_.debug_info());
            builder.add_computational_memlet(
                tmp_Y_free_block, tmp_Y_access_in, libnode, "_ptr", {}, tmp_Y_type, node_.debug_info()
            );
            builder.add_computational_memlet(
                tmp_Y_free_block, libnode, "_ptr", tmp_Y_access_out, {}, tmp_Y_type, node_.debug_info()
            );
        }

        /* ===== No groups ====================================================================== */

    } else {
        /* ===== Groups ========================================================================= */

        auto in_channels = symbolic::div(node_.shape()[1], node_.group());
        auto out_channels = symbolic::div(node_.output_channels(), node_.group());

        // Add loop over batch size
        auto n_container = builder.find_new_name("_n");
        builder.add_container(n_container, indvar_type);
        auto n = symbolic::symbol(n_container);
        auto& loop_n = builder.add_map(
            new_sequence,
            n,
            symbolic::Lt(n, node_.shape()[0]),
            symbolic::zero(),
            symbolic::add(n, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
        );

        // Add loop over groups
        auto g_container = builder.find_new_name("_g");
        builder.add_container(g_container, indvar_type);
        auto g = symbolic::symbol(g_container);
        auto& loop_g = builder.add_map(
            loop_n.root(),
            g,
            symbolic::Lt(g, node_.group()),
            symbolic::zero(),
            symbolic::add(g, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
        );

        // Add patches container with malloc
        symbolic::Expression patches_size = in_channels;
        for (size_t i = 0; i < dims; i++) {
            patches_size = symbolic::mul(patches_size, symbolic::mul(node_.kernel_shape()[i], out_shape[i]));
        }
        types::Pointer patches_type(base_type);
        auto patches_container = builder.find_new_name("_patches");
        builder.add_container(patches_container, patches_type);
        auto& patches_malloc_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& patches_access = builder.add_access(patches_malloc_block, patches_container, node_.debug_info());
            auto& libnode = builder.add_library_node<stdlib::MallocNode>(
                patches_malloc_block, node_.debug_info(), symbolic::mul(patches_size, symbolic::size_of_type(base_type))
            );
            builder.add_computational_memlet(
                patches_malloc_block, libnode, "_ret", patches_access, {}, patches_type, node_.debug_info()
            );
        }

        // Add loops over output dimensions
        structured_control_flow::Sequence* current_seq = &loop_g.root();
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
                block->debug_info()
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
            symbolic::Lt(c, in_channels),
            symbolic::zero(),
            symbolic::add(c, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
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
                symbolic::Lt(k, node_.kernel_shape()[i]),
                symbolic::zero(),
                symbolic::add(k, symbolic::one()),
                ScheduleType_Sequential::create(),
                {},
                block->debug_info()
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
                add(symbolic::sub(symbolic::mul(os[i], node_.strides()[i]), node_.pads()[i]),
                    symbolic::mul(ks[i], node_.dilations()[i]));
            is.push_back(i_expr);
            copy_condition = symbolic::
                And(copy_condition,
                    symbolic::And(symbolic::Lt(i_expr, node_.shape()[i + 2]), symbolic::Ge(i_expr, symbolic::zero())));
            zero_condition = symbolic::
                Or(zero_condition,
                   symbolic::Or(symbolic::Ge(i_expr, node_.shape()[i + 2]), symbolic::Lt(i_expr, symbolic::zero())));
        }
        auto& branch = builder.add_if_else(*current_seq, {}, block->debug_info());
        auto& copy_case = builder.add_case(branch, copy_condition, block->debug_info());
        auto& zero_case = builder.add_case(branch, zero_condition, block->debug_info());

        // Determine patches subset & tensor type
        data_flow::Subset patches_subset;
        patches_subset.push_back(c);
        patches_subset.insert(patches_subset.end(), ks.begin(), ks.end());
        patches_subset.insert(patches_subset.end(), os.begin(), os.end());
        symbolic::MultiExpression patches_shape;
        patches_shape.push_back(in_channels);
        patches_shape.insert(patches_shape.end(), node_.kernel_shape().begin(), node_.kernel_shape().end());
        patches_shape.insert(patches_shape.end(), out_shape.begin(), out_shape.end());
        types::Tensor patches_tensor_type(base_type, patches_shape);

        // Determine subset for X
        data_flow::Subset subset_X;
        subset_X.push_back(n);
        subset_X.push_back(symbolic::add(symbolic::mul(in_channels, g), c));
        subset_X.insert(subset_X.end(), is.begin(), is.end());

        // Add copy from X to patches
        auto& copy_block = builder.add_block(copy_case, {}, block->debug_info());
        {
            auto& X_access = builder.add_access(copy_block, access_X->data(), access_X->debug_info());
            auto& patches_access = builder.add_access(copy_block, patches_container, node_.debug_info());
            auto& tasklet =
                builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node_.debug_info());
            builder.add_computational_memlet(
                copy_block, X_access, tasklet, "_in", subset_X, iedge_X->base_type(), iedge_X->debug_info()
            );
            builder.add_computational_memlet(
                copy_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, node_.debug_info()
            );
        }

        // Add zero assignment to patches
        auto& zero_block = builder.add_block(zero_case, {}, block->debug_info());
        {
            auto& constant_zero = builder.add_constant(zero_block, "0.0", base_type, node_.debug_info());
            auto& patches_access = builder.add_access(zero_block, patches_container, node_.debug_info());
            auto& tasklet =
                builder.add_tasklet(zero_block, data_flow::TaskletCode::assign, "_out", {"_in"}, node_.debug_info());
            builder
                .add_computational_memlet(zero_block, constant_zero, tasklet, "_in", {}, base_type, node_.debug_info());
            builder.add_computational_memlet(
                zero_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, node_.debug_info()
            );
        }

        // Add reference to W
        auto ref_W_container = builder.find_new_name("_ref_W");
        types::Scalar ref_W_base_type(builder.subject().type(access_W->data()).primitive_type());
        types::Pointer ref_W_type(ref_W_base_type);
        builder.add_container(ref_W_container, ref_W_type);
        auto ref_W_subset = symbolic::mul(symbolic::mul(out_channels, g), in_channels);
        for (size_t i = 0; i < dims; i++) {
            ref_W_subset = symbolic::mul(ref_W_subset, node_.kernel_shape()[i]);
        }
        auto& ref_W_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& W_access = builder.add_access(ref_W_block, access_W->data(), access_W->debug_info());
            auto& ref_W_access = builder.add_access(ref_W_block, ref_W_container, access_W->debug_info());
            builder.add_reference_memlet(ref_W_block, W_access, ref_W_access, {ref_W_subset}, ref_W_type);
        }

        // Add reference to Y
        auto ref_Y_container = builder.find_new_name("_ref_Y");
        types::Scalar ref_Y_base_type(builder.subject().type(access_Y->data()).primitive_type());
        types::Pointer ref_Y_type(ref_Y_base_type);
        builder.add_container(ref_Y_container, ref_Y_type);
        auto ref_Y_subset = symbolic::add(symbolic::mul(node_.output_channels(), n), symbolic::mul(out_channels, g));
        for (size_t i = 0; i < dims; i++) {
            ref_Y_subset = symbolic::mul(ref_Y_subset, out_shape[i]);
        }
        auto& ref_Y_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& Y_access = builder.add_access(ref_Y_block, access_Y->data(), access_Y->debug_info());
            auto& ref_Y_access = builder.add_access(ref_Y_block, ref_Y_container, access_Y->debug_info());
            builder.add_reference_memlet(ref_Y_block, Y_access, ref_Y_access, {ref_Y_subset}, ref_Y_type);
        }

        // Add GEMM node
        auto& gemm_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, node_.debug_info());
            auto& beta = builder.add_constant(gemm_block, "0.0", base_type, node_.debug_info());
            auto& ref_W_access = builder.add_access(gemm_block, ref_W_container, access_W->debug_info());
            auto& patches_access = builder.add_access(gemm_block, patches_container, node_.debug_info());
            auto& ref_Y_access_in = builder.add_access(gemm_block, ref_Y_container, access_Y->debug_info());
            auto& ref_Y_access_out = builder.add_access(gemm_block, ref_Y_container, access_Y->debug_info());
            symbolic::Expression gemm_m = out_channels;
            symbolic::Expression gemm_n = symbolic::one();
            symbolic::Expression gemm_k = in_channels;
            for (size_t i = 0; i < dims; i++) {
                gemm_n = symbolic::mul(gemm_n, out_shape[i]);
                gemm_k = symbolic::mul(gemm_k, node_.kernel_shape()[i]);
            }
            auto& libnode = builder.add_library_node<math::blas::GEMMNode>(
                gemm_block,
                node_.debug_info(),
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
            builder.add_computational_memlet(gemm_block, alpha, libnode, "__alpha", {}, base_type, node_.debug_info());
            builder.add_computational_memlet(gemm_block, beta, libnode, "__beta", {}, base_type, node_.debug_info());
            builder
                .add_computational_memlet(gemm_block, ref_W_access, libnode, "__A", {}, ref_W_type, iedge_W->debug_info());
            builder.add_computational_memlet(
                gemm_block, patches_access, libnode, "__B", {}, patches_type, node_.debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, ref_Y_access_in, libnode, "__C", {}, ref_Y_type, oedge_Y->debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, libnode, "__C", ref_Y_access_out, {}, ref_Y_type, oedge_Y->debug_info()
            );
        }

        // Add bias if available
        if (has_bias) {
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
                block->debug_info()
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
                    block->debug_info()
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
            auto& bias_block = builder.add_block(*current_seq, {}, block->debug_info());
            {
                auto& B_access = builder.add_access(bias_block, access_B->data(), access_B->debug_info());
                auto& Y_access_in = builder.add_access(bias_block, access_Y->data(), access_Y->debug_info());
                auto& Y_access_out = builder.add_access(bias_block, access_Y->data(), access_Y->debug_info());
                auto& tasklet = builder.add_tasklet(
                    bias_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, node_.debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, Y_access_in, tasklet, "_in1", Y_subset, oedge_Y->base_type(), node_.debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, B_access, tasklet, "_in2", {B_subset}, iedge_B->base_type(), iedge_B->debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, tasklet, "_out", Y_access_out, Y_subset, oedge_Y->base_type(), oedge_Y->debug_info()
                );
            }
        }

        // Add free for patches container
        auto& patches_free_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& patches_access_in = builder.add_access(patches_free_block, patches_container, node_.debug_info());
            auto& patches_access_out = builder.add_access(patches_free_block, patches_container, node_.debug_info());
            auto& libnode = builder.add_library_node<stdlib::FreeNode>(patches_free_block, node_.debug_info());
            builder.add_computational_memlet(
                patches_free_block, patches_access_in, libnode, "_ptr", {}, patches_type, node_.debug_info()
            );
            builder.add_computational_memlet(
                patches_free_block, libnode, "_ptr", patches_access_out, {}, patches_type, node_.debug_info()
            );
        }

        /* ===== Groups ========================================================================= */
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
    builder.remove_node(*block, node_);
    builder.remove_child(*block_parent, block_index + 1);

    return true;
}
} // namespace offloading
} // namespace sdfg
