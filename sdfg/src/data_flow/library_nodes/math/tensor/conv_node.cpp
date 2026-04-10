#include "sdfg/data_flow/library_nodes/math/tensor/conv_node.h"

#include <map>
#include <sstream>

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

    // Compute spatial dimensions: K = C_in * prod(kernel_shape), spatial = prod(out_shape)
    // Patches layout: [K, spatial] = [C_in, kernel..., out_shape...]
    // GEMM: Y[F, spatial] = W[F, K] × patches[K, spatial]  (NoTrans × NoTrans)
    // Output Y is [N, F, out_shape...] — GEMM writes directly per batch via pointer reference

    if (symbolic::eq(this->group_, symbolic::one())) {
        /* ===== No groups (H-tiled im2col + GEMM) ============================================= */

        // Compute K = C_in * prod(kernel_shape) and spatial = prod(out_shape)
        symbolic::Expression gemm_k = this->shape_[1]; // C_in
        for (size_t i = 0; i < dims; i++) {
            gemm_k = symbolic::mul(gemm_k, this->kernel_shape_[i]);
        }
        symbolic::Expression spatial = symbolic::one();
        for (size_t i = 0; i < dims; i++) {
            spatial = symbolic::mul(spatial, out_shape[i]);
        }

        // Compute H-tile parameters for the first spatial output dimension
        // tail_spatial = prod(out_shape[1:])
        symbolic::Expression tail_spatial = symbolic::one();
        for (size_t i = 1; i < dims; i++) {
            tail_spatial = symbolic::mul(tail_spatial, out_shape[i]);
        }

        // Compute tile size targeting ~2MB L2 cache
        symbolic::Expression h_tile = out_shape[0]; // default: no tiling (full H)
        symbolic::Expression num_h_tiles = symbolic::one();
        {
            bool can_compute = true;
            int64_t K_val = 0, tail_val = 1, h_out_val = 0;
            size_t elem_size = (base_type.primitive_type() == types::PrimitiveType::Double)  ? 8
                               : (base_type.primitive_type() == types::PrimitiveType::Float) ? 4
                                                                                             : 2;

            if (SymEngine::is_a<SymEngine::Integer>(*gemm_k)) {
                K_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(gemm_k)->as_int();
            } else {
                can_compute = false;
            }
            if (SymEngine::is_a<SymEngine::Integer>(*out_shape[0])) {
                h_out_val = SymEngine::rcp_static_cast<const SymEngine::Integer>(out_shape[0])->as_int();
            } else {
                can_compute = false;
            }
            for (size_t i = 1; i < dims && can_compute; i++) {
                if (SymEngine::is_a<SymEngine::Integer>(*out_shape[i])) {
                    tail_val *= SymEngine::rcp_static_cast<const SymEngine::Integer>(out_shape[i])->as_int();
                } else {
                    can_compute = false;
                }
            }

            if (can_compute && K_val > 0 && tail_val > 0 && h_out_val > 0) {
                int64_t target_bytes = 2 * 1024 * 1024; // 2 MB
                int64_t h_tile_val = target_bytes / (K_val * tail_val * (int64_t) elem_size);
                h_tile_val = (h_tile_val < 1) ? 1 : ((h_tile_val > h_out_val) ? h_out_val : h_tile_val);
                // Find largest divisor of h_out_val <= h_tile_val
                while (h_tile_val > 1 && h_out_val % h_tile_val != 0) {
                    h_tile_val--;
                }
                h_tile = SymEngine::integer(h_tile_val);
                num_h_tiles = SymEngine::integer(h_out_val / h_tile_val);
            }
        }

        // tile_n = h_tile × tail_spatial (number of spatial positions per tile)
        symbolic::Expression tile_n = symbolic::mul(h_tile, tail_spatial);

        // Patches buffer: K × tile_n (tiled, reused across batches and tiles)
        symbolic::Expression patches_size = symbolic::mul(gemm_k, tile_n);
        types::Pointer patches_type(base_type);
        auto patches_container = builder.find_new_name("_patches");
        builder.add_container(patches_container, patches_type);

        // Batch loop
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
            block->debug_info()
        );

        // H-tile loop (inside batch)
        auto t_container = builder.find_new_name("_t");
        builder.add_container(t_container, indvar_type);
        auto t = symbolic::symbol(t_container);
        auto& loop_t = builder.add_map(
            loop_n.root(),
            t,
            symbolic::Lt(t, num_h_tiles),
            symbolic::zero(),
            symbolic::add(t, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
        );

        // Malloc patches (outside batch loop, single allocation reused)
        auto& patches_malloc_block = builder.add_block(loop_t.root(), {}, block->debug_info());
        {
            auto& patches_access = builder.add_access(patches_malloc_block, patches_container, this->debug_info());
            auto& libnode = builder.add_library_node<stdlib::MallocNode>(
                patches_malloc_block, this->debug_info(), symbolic::mul(patches_size, symbolic::size_of_type(base_type))
            );
            builder.add_computational_memlet(
                patches_malloc_block, libnode, "_ret", patches_access, {}, patches_type, this->debug_info()
            );
        }

        // Im2col: c, k[0..dims-1], o0_inner ∈ [0,h_tile), o[1..dims-1]
        // Patches layout: [C_in, kernel..., h_tile, out_shape[1:]...] — row-major [K, tile_n]
        structured_control_flow::Sequence* current_seq = &loop_t.root();

        // Channel loop
        auto c_container = builder.find_new_name("_c");
        builder.add_container(c_container, indvar_type);
        auto c = symbolic::symbol(c_container);
        auto& loop_c = builder.add_map(
            *current_seq,
            c,
            symbolic::Lt(c, this->shape_[1]),
            symbolic::zero(),
            symbolic::add(c, symbolic::one()),
            ScheduleType_Sequential::create(),
            {},
            block->debug_info()
        );
        current_seq = &loop_c.root();

        // Kernel dimension loops
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
                block->debug_info()
            );
            current_seq = &loop_k.root();
        }

        // Output spatial dimension loops
        // Dim 0: tiled inner loop o0_inner ∈ [0, h_tile)
        // Dim 1+: full range o_i ∈ [0, out_shape[i])
        symbolic::SymbolVec os;
        os.reserve(dims);
        for (size_t i = 0; i < dims; i++) {
            auto o_container = builder.find_new_name("_o");
            builder.add_container(o_container, indvar_type);
            auto o = symbolic::symbol(o_container);
            os.push_back(o);
            symbolic::Expression bound = (i == 0) ? h_tile : out_shape[i];
            auto& loop_o = builder.add_map(
                *current_seq,
                o,
                symbolic::Lt(o, bound),
                symbolic::zero(),
                symbolic::add(o, symbolic::one()),
                ScheduleType_Sequential::create(),
                {},
                block->debug_info()
            );
            current_seq = &loop_o.root();
        }

        // Compute input indices using actual output position
        // Dim 0: actual_o0 = t * h_tile + o0_inner
        // Dim i>0: actual_o_i = o_i
        symbolic::MultiExpression is;
        is.reserve(dims);
        symbolic::Condition copy_condition = symbolic::__true__();
        symbolic::Condition zero_condition = symbolic::__false__();
        for (size_t i = 0; i < dims; i++) {
            symbolic::Expression actual_o = (i == 0) ? symbolic::add(symbolic::mul(t, h_tile), os[0])
                                                     : symbolic::Expression(os[i]);
            auto i_expr = symbolic::
                add(symbolic::sub(symbolic::mul(actual_o, this->strides_[i]), this->pads_[i]),
                    symbolic::mul(ks[i], this->dilations_[i]));
            is.push_back(i_expr);
            copy_condition = symbolic::
                And(copy_condition,
                    symbolic::And(symbolic::Lt(i_expr, this->shape_[i + 2]), symbolic::Ge(i_expr, symbolic::zero())));
            zero_condition = symbolic::
                Or(zero_condition,
                   symbolic::Or(symbolic::Ge(i_expr, this->shape_[i + 2]), symbolic::Lt(i_expr, symbolic::zero())));
        }
        auto& branch = builder.add_if_else(*current_seq, {}, block->debug_info());
        auto& copy_case = builder.add_case(branch, copy_condition, block->debug_info());
        auto& zero_case = builder.add_case(branch, zero_condition, block->debug_info());

        // Patches subset: [c, k0, k1, ..., o0_inner, o1, ...]
        // Patches shape: [C_in, kH, kW, ..., h_tile, out_shape[1], ...]
        data_flow::Subset patches_subset;
        patches_subset.push_back(c);
        patches_subset.insert(patches_subset.end(), ks.begin(), ks.end());
        patches_subset.insert(patches_subset.end(), os.begin(), os.end());
        symbolic::MultiExpression patches_shape;
        patches_shape.push_back(this->shape_[1]); // C_in
        patches_shape.insert(patches_shape.end(), this->kernel_shape_.begin(), this->kernel_shape_.end());
        patches_shape.push_back(h_tile); // tiled first spatial dim
        for (size_t i = 1; i < dims; i++) {
            patches_shape.push_back(out_shape[i]);
        }
        types::Tensor patches_tensor_type(base_type, patches_shape);

        // X subset: [n, c, i0, i1, ...]
        data_flow::Subset subset_X;
        subset_X.push_back(n);
        subset_X.push_back(c);
        subset_X.insert(subset_X.end(), is.begin(), is.end());

        // Copy X → patches
        auto& copy_block = builder.add_block(copy_case, {}, block->debug_info());
        {
            auto& X_access = builder.add_access(copy_block, access_X->data(), access_X->debug_info());
            auto& patches_access = builder.add_access(copy_block, patches_container, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
            builder.add_computational_memlet(
                copy_block, X_access, tasklet, "_in", subset_X, iedge_X->base_type(), iedge_X->debug_info()
            );
            builder.add_computational_memlet(
                copy_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
            );
        }

        // Zero → patches (out-of-bounds padding positions)
        auto& zero_block = builder.add_block(zero_case, {}, block->debug_info());
        {
            auto& constant_zero = builder.add_constant(zero_block, "0.0", base_type, this->debug_info());
            auto& patches_access = builder.add_access(zero_block, patches_container, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(zero_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
            builder
                .add_computational_memlet(zero_block, constant_zero, tasklet, "_in", {}, base_type, this->debug_info());
            builder.add_computational_memlet(
                zero_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
            );
        }

        // Reference to Y at tile position: Y + n*F*spatial + t*tile_n
        auto ref_Y_container = builder.find_new_name("_ref_Y");
        types::Scalar ref_Y_base_type(builder.subject().type(access_Y->data()).primitive_type());
        types::Pointer ref_Y_type(ref_Y_base_type);
        builder.add_container(ref_Y_container, ref_Y_type);
        auto ref_Y_offset =
            symbolic::add(symbolic::mul(n, symbolic::mul(this->output_channels_, spatial)), symbolic::mul(t, tile_n));
        auto& ref_Y_block = builder.add_block(loop_t.root(), {}, block->debug_info());
        {
            auto& Y_access = builder.add_access(ref_Y_block, access_Y->data(), access_Y->debug_info());
            auto& ref_Y_access = builder.add_access(ref_Y_block, ref_Y_container, access_Y->debug_info());
            builder.add_reference_memlet(ref_Y_block, Y_access, ref_Y_access, {ref_Y_offset}, ref_Y_type);
        }

        // GEMM: Y_tile[F, tile_n] = W[F, K] × patches[K, tile_n]
        // ldb = tile_n (patches row stride), ldc = spatial (output row stride in Y)
        auto& gemm_block = builder.add_block(loop_t.root(), {}, block->debug_info());
        {
            auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, this->debug_info());
            auto& beta = builder.add_constant(gemm_block, "0.0", base_type, this->debug_info());
            auto& W_access = builder.add_access(gemm_block, access_W->data(), access_W->debug_info());
            auto& patches_access = builder.add_access(gemm_block, patches_container, this->debug_info());
            auto& ref_Y_access_in = builder.add_access(gemm_block, ref_Y_container, access_Y->debug_info());
            auto& ref_Y_access_out = builder.add_access(gemm_block, ref_Y_container, access_Y->debug_info());
            auto& libnode = builder.add_library_node<blas::GEMMNode>(
                gemm_block,
                this->debug_info(),
                blas::ImplementationType_BLAS,
                precision,
                blas::BLAS_Layout::RowMajor,
                blas::BLAS_Transpose::No, // transA
                blas::BLAS_Transpose::No, // transB
                this->output_channels_, // m = F
                tile_n, // n = tile_n (tiled!)
                gemm_k, // k = K
                gemm_k, // lda = K
                tile_n, // ldb = tile_n
                spatial // ldc = spatial (full output row stride!)
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
            builder.add_computational_memlet(
                gemm_block, patches_access, libnode, "__B", {}, patches_type, this->debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, ref_Y_access_in, libnode, "__C", {}, ref_Y_type, oedge_Y->debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, libnode, "__C", ref_Y_access_out, {}, ref_Y_type, oedge_Y->debug_info()
            );
        }

        // Bias (inside batch loop, outside tile loop): Y[n, f, o...] += B[f]
        if (has_bias) {
            auto l_container = builder.find_new_name("_l");
            builder.add_container(l_container, indvar_type);
            auto l = symbolic::symbol(l_container);
            auto& loop_l = builder.add_map(
                loop_n.root(),
                l,
                symbolic::Lt(l, this->output_channels_),
                symbolic::zero(),
                symbolic::add(l, symbolic::one()),
                ScheduleType_Sequential::create(),
                {},
                block->debug_info()
            );
            structured_control_flow::Sequence* bias_seq = &loop_l.root();

            symbolic::SymbolVec bias_os;
            bias_os.reserve(dims);
            for (size_t i = 0; i < dims; i++) {
                auto o_container = builder.find_new_name("_o");
                builder.add_container(o_container, indvar_type);
                auto o = symbolic::symbol(o_container);
                bias_os.push_back(o);
                auto& loop_o = builder.add_map(
                    *bias_seq,
                    o,
                    symbolic::Lt(o, out_shape[i]),
                    symbolic::zero(),
                    symbolic::add(o, symbolic::one()),
                    ScheduleType_Sequential::create(),
                    {},
                    block->debug_info()
                );
                bias_seq = &loop_o.root();
            }

            data_flow::Subset Y_subset;
            Y_subset.push_back(n);
            Y_subset.push_back(l);
            Y_subset.insert(Y_subset.end(), bias_os.begin(), bias_os.end());

            auto& bias_block = builder.add_block(*bias_seq, {}, block->debug_info());
            {
                auto& B_access = builder.add_access(bias_block, access_B->data(), access_B->debug_info());
                auto& Y_access_in = builder.add_access(bias_block, access_Y->data(), access_Y->debug_info());
                auto& Y_access_out = builder.add_access(bias_block, access_Y->data(), access_Y->debug_info());
                auto& tasklet = builder.add_tasklet(
                    bias_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, Y_access_in, tasklet, "_in1", Y_subset, oedge_Y->base_type(), this->debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, B_access, tasklet, "_in2", {l}, iedge_B->base_type(), iedge_B->debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, tasklet, "_out", Y_access_out, Y_subset, oedge_Y->base_type(), oedge_Y->debug_info()
                );
            }
        }

        // Free patches
        auto& patches_free_block = builder.add_block(loop_t.root(), {}, block->debug_info());
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

        /* ===== No groups (H-tiled im2col + GEMM) ============================================= */

    } else {
        /* ===== Groups ========================================================================= */

        auto in_channels = symbolic::div(this->shape_[1], this->group_);
        auto out_channels = symbolic::div(this->output_channels_, this->group_);

        // Compute K_group = in_channels * prod(kernel_shape), spatial = prod(out_shape)
        symbolic::Expression gemm_k = in_channels;
        for (size_t i = 0; i < dims; i++) {
            gemm_k = symbolic::mul(gemm_k, this->kernel_shape_[i]);
        }
        symbolic::Expression spatial = symbolic::one();
        for (size_t i = 0; i < dims; i++) {
            spatial = symbolic::mul(spatial, out_shape[i]);
        }

        // Patches buffer: K_group × spatial (per group iteration, reused)
        symbolic::Expression patches_size = symbolic::mul(gemm_k, spatial);
        types::Pointer patches_type(base_type);
        auto patches_container = builder.find_new_name("_patches");
        builder.add_container(patches_container, patches_type);

        // Batch loop
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
            block->debug_info()
        );

        // Group loop
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
            block->debug_info()
        );


        // Malloc patches (outside loops)
        auto& patches_malloc_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& patches_access = builder.add_access(patches_malloc_block, patches_container, this->debug_info());
            auto& libnode = builder.add_library_node<stdlib::MallocNode>(
                patches_malloc_block, this->debug_info(), symbolic::mul(patches_size, symbolic::size_of_type(base_type))
            );
            builder.add_computational_memlet(
                patches_malloc_block, libnode, "_ret", patches_access, {}, patches_type, this->debug_info()
            );
        }

        // Im2col loops: c, k[0..dims-1], o[0..dims-1]
        structured_control_flow::Sequence* current_seq = &loop_g.root();

        // Channel loop (over in_channels per group)
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

        // Kernel dimension loops
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
                block->debug_info()
            );
            current_seq = &loop_k.root();
        }

        // Output spatial loops (with bounds check for padding/dilation)
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
        auto& branch = builder.add_if_else(*current_seq, {}, block->debug_info());
        auto& copy_case = builder.add_case(branch, copy_condition, block->debug_info());
        auto& zero_case = builder.add_case(branch, zero_condition, block->debug_info());

        // Patches subset: [c, k0, k1, ..., o0, o1, ...]
        // Patches shape: [in_channels, kernel..., out_shape...]
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

        // X subset: [n, g * in_channels + c, i0, i1, ...]
        data_flow::Subset subset_X;
        subset_X.push_back(n);
        subset_X.push_back(symbolic::add(symbolic::mul(in_channels, g), c));
        subset_X.insert(subset_X.end(), is.begin(), is.end());

        // Add copy from X to patches
        auto& copy_block = builder.add_block(copy_case, {}, block->debug_info());
        {
            auto& X_access = builder.add_access(copy_block, access_X->data(), access_X->debug_info());
            auto& patches_access = builder.add_access(copy_block, patches_container, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(copy_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
            builder.add_computational_memlet(
                copy_block, X_access, tasklet, "_in", subset_X, iedge_X->base_type(), iedge_X->debug_info()
            );
            builder.add_computational_memlet(
                copy_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
            );
        }

        // Add zero assignment to patches
        auto& zero_block = builder.add_block(zero_case, {}, block->debug_info());
        {
            auto& constant_zero = builder.add_constant(zero_block, "0.0", base_type, this->debug_info());
            auto& patches_access = builder.add_access(zero_block, patches_container, this->debug_info());
            auto& tasklet =
                builder.add_tasklet(zero_block, data_flow::TaskletCode::assign, "_out", {"_in"}, this->debug_info());
            builder
                .add_computational_memlet(zero_block, constant_zero, tasklet, "_in", {}, base_type, this->debug_info());
            builder.add_computational_memlet(
                zero_block, tasklet, "_out", patches_access, patches_subset, patches_tensor_type, this->debug_info()
            );
        }

        // Reference to W[g, :, :, ...] — offset = g * out_channels * K_group
        auto ref_W_container = builder.find_new_name("_ref_W");
        types::Scalar ref_W_base_type(builder.subject().type(access_W->data()).primitive_type());
        types::Pointer ref_W_type(ref_W_base_type);
        builder.add_container(ref_W_container, ref_W_type);
        auto ref_W_offset = symbolic::mul(g, symbolic::mul(out_channels, gemm_k));
        auto& ref_W_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& W_access = builder.add_access(ref_W_block, access_W->data(), access_W->debug_info());
            auto& ref_W_access = builder.add_access(ref_W_block, ref_W_container, access_W->debug_info());
            builder.add_reference_memlet(ref_W_block, W_access, ref_W_access, {ref_W_offset}, ref_W_type);
        }

        // Reference to Y[n, g*out_channels, 0, ...] — offset = (n * F + g * out_channels) * spatial
        auto ref_Y_container = builder.find_new_name("_ref_Y");
        types::Scalar ref_Y_base_type(builder.subject().type(access_Y->data()).primitive_type());
        types::Pointer ref_Y_type(ref_Y_base_type);
        builder.add_container(ref_Y_container, ref_Y_type);
        auto ref_Y_offset =
            symbolic::mul(symbolic::add(symbolic::mul(this->output_channels_, n), symbolic::mul(out_channels, g)), spatial);
        auto& ref_Y_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& Y_access = builder.add_access(ref_Y_block, access_Y->data(), access_Y->debug_info());
            auto& ref_Y_access = builder.add_access(ref_Y_block, ref_Y_container, access_Y->debug_info());
            builder.add_reference_memlet(ref_Y_block, Y_access, ref_Y_access, {ref_Y_offset}, ref_Y_type);
        }

        // GEMM: Y_ref[out_channels, spatial] = W_ref[out_channels, K_group] × patches[K_group, spatial]
        auto& gemm_block = builder.add_block(loop_g.root(), {}, block->debug_info());
        {
            auto& alpha = builder.add_constant(gemm_block, "1.0", base_type, this->debug_info());
            auto& beta = builder.add_constant(gemm_block, "0.0", base_type, this->debug_info());
            auto& ref_W_access = builder.add_access(gemm_block, ref_W_container, access_W->debug_info());
            auto& patches_access = builder.add_access(gemm_block, patches_container, this->debug_info());
            auto& ref_Y_access_in = builder.add_access(gemm_block, ref_Y_container, access_Y->debug_info());
            auto& ref_Y_access_out = builder.add_access(gemm_block, ref_Y_container, access_Y->debug_info());
            auto& libnode = builder.add_library_node<blas::GEMMNode>(
                gemm_block,
                this->debug_info(),
                blas::ImplementationType_BLAS,
                precision,
                blas::BLAS_Layout::RowMajor,
                blas::BLAS_Transpose::No, // transA
                blas::BLAS_Transpose::No, // transB
                out_channels, // m
                spatial, // n
                gemm_k, // k
                gemm_k, // lda
                spatial, // ldb
                spatial // ldc
            );
            builder.add_computational_memlet(gemm_block, alpha, libnode, "__alpha", {}, base_type, this->debug_info());
            builder.add_computational_memlet(gemm_block, beta, libnode, "__beta", {}, base_type, this->debug_info());
            builder
                .add_computational_memlet(gemm_block, ref_W_access, libnode, "__A", {}, ref_W_type, iedge_W->debug_info());
            builder.add_computational_memlet(
                gemm_block, patches_access, libnode, "__B", {}, patches_type, this->debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, ref_Y_access_in, libnode, "__C", {}, ref_Y_type, oedge_Y->debug_info()
            );
            builder.add_computational_memlet(
                gemm_block, libnode, "__C", ref_Y_access_out, {}, ref_Y_type, oedge_Y->debug_info()
            );
        }

        // Add bias if available: Y[n, g*out_channels + l, o...] += B[g*out_channels + l]
        if (has_bias) {
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
            structured_control_flow::Sequence* bias_seq = &loop_l.root();

            symbolic::SymbolVec bias_os;
            bias_os.reserve(dims);
            for (size_t i = 0; i < dims; i++) {
                auto o_container = builder.find_new_name("_o");
                builder.add_container(o_container, indvar_type);
                auto o = symbolic::symbol(o_container);
                bias_os.push_back(o);
                auto& loop_o = builder.add_map(
                    *bias_seq,
                    o,
                    symbolic::Lt(o, out_shape[i]),
                    symbolic::zero(),
                    symbolic::add(o, symbolic::one()),
                    ScheduleType_Sequential::create(),
                    {},
                    block->debug_info()
                );
                bias_seq = &loop_o.root();
            }

            data_flow::Subset Y_subset;
            Y_subset.push_back(n);
            Y_subset.push_back(symbolic::add(symbolic::mul(out_channels, g), l));
            Y_subset.insert(Y_subset.end(), bias_os.begin(), bias_os.end());
            auto B_subset = symbolic::add(symbolic::mul(out_channels, g), l);

            auto& bias_block = builder.add_block(*bias_seq, {}, block->debug_info());
            {
                auto& B_access = builder.add_access(bias_block, access_B->data(), access_B->debug_info());
                auto& Y_access_in = builder.add_access(bias_block, access_Y->data(), access_Y->debug_info());
                auto& Y_access_out = builder.add_access(bias_block, access_Y->data(), access_Y->debug_info());
                auto& tasklet = builder.add_tasklet(
                    bias_block, data_flow::TaskletCode::fp_add, "_out", {"_in1", "_in2"}, this->debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, Y_access_in, tasklet, "_in1", Y_subset, oedge_Y->base_type(), this->debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, B_access, tasklet, "_in2", {B_subset}, iedge_B->base_type(), iedge_B->debug_info()
                );
                builder.add_computational_memlet(
                    bias_block, tasklet, "_out", Y_access_out, Y_subset, oedge_Y->base_type(), oedge_Y->debug_info()
                );
            }
        }

        // Free patches
        auto& patches_free_block = builder.add_block(loop_g.root(), {}, block->debug_info());
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
    builder.remove_node(*block, *this);
    builder.remove_child(*block_parent, block_index + 1);

    {
        // CPU-specific transformations on the canonical expansion:
        //
        // 1. LoopTiling on h_out inside im2col
        //    - Split the output spatial H-loop into tiles of H_TILE rows
        //    - Reduces patches buffer from K×spatial to K×(H_TILE×W_out)
        //    - Tile size chosen to fit patches + weights in L2 cache
        // 2. TileFusion of im2col + GEMM
        //    - Fuse the tiled im2col block with the per-tile GEMM
        //    - Each tile: extract patches → GEMM → next tile
        //    - Both weights A and patches B reside in L2 when GEMM runs
        //
        // 3. MapCollapse on batch × h_tile
        //    - Flatten (batch, h_tile) into a single parallel dimension
        //    - Maximizes parallel work items (e.g., 32 batches × 4 tiles = 128)
        //
        // 4. Malloc privatization
        //    - Move patches malloc/free inside the parallel region
        //    - Each thread allocates its tile buffer once, reuses across tiles
    }
    {
        // GPU-specific transformations on the canonical expansion:
        //
        // 1. LoopTiling on output spatial dimensions
        //    - Tile H_out and W_out for thread block mapping
        //    - Tile sizes chosen for shared memory capacity
        //
        // 2. Map outer dimensions to GPU grid
        //    - batch → grid.z, output_channel_tile → grid.y, spatial_tile → grid.x
        //    - Apply CUDAScheduler to outer maps
        //
        // 3. Shared memory for patches tile
        //    - Replace malloc with __shared__ memory allocation
        //    - Cooperative loading: threads in a block load patches tile together
        //
        // 4. Register tiling for GEMM accumulation
        //    - Each thread computes a small tile of the output
        //    - Use fp_fma tasklets for multiply-accumulate
        //
        // 5. Memory coalescing
        //    - Ensure im2col reads and output writes are coalesced
        //    - Thread indexing aligned with contiguous memory dimension
    }

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
