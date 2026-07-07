// =============================================================================
// GPU Kernel Tests
// =============================================================================
//
// The pipeline demonstrated here is the canonical way to bring a sequential,
// flat-pointer matmul kernel onto the GPU using a CUTLASS-style
// thread-coarsened layout: each thread owns a CY*CX register tile of C and
// the inner k-loop is an outer product over (iI, jI) that consumes A/B
// fragments staged in shared memory.
//
// Transformations are grouped by phase: tiling -> interchange -> offload ->
// local storage. Each phase only reshapes the loop nest; the offload phase
// is the single point where the kernel gets bound to CUDA grid/block dims.
//
//   sequential Map(i) Map(j) For(k)                            // host-side scalar kernel
//     |
//     |  === TILING ===
//     |  (T1) LoopTiling(for_i, CY)                            // thread coarsening, rows
//     |  (T2) LoopTiling(for_j, CX)                            // thread coarsening, cols
//     |  (T3) LoopTiling(for_k, TK)                            // strip-mine k
//     v
//   Map(iO,CY) Map(iI) Map(jO,CX) Map(jI) For(kk,TK) For(kI)
//     |
//     |  === INTERCHANGE ===
//     |  (I1) LoopInterchange(iI, jO)                          // put block loops adjacent
//     |  (I2) LoopInterchange(iO, jO)                          // coalesce X-dim along j
//     |  (I3) LoopInterchange(jI, kk)                          // sink kk above (iI, jI)
//     |  (I4) LoopInterchange(iI, kk)
//     |  (I5) LoopInterchange(jI, kI)                          // sink kI above (iI, jI)
//     |  (I6) LoopInterchange(iI, kI)
//     v
//   Map(jO) Map(iO) For(kk) For(kI) Map(iI) Map(jI) { fma }
//     |
//     |  === OFFLOAD ===
//     |  (O1) cuda::CUDATransform(jO, TX=8)                    // offload to CUDA X-dim
//     |  (O2) CUDAParallelizeNestedMap(iO, TY=4)               // nested Y-dim
//     v
//   Map_X(jO,TX) Map_Y(iO,TY) For(kk) For(kI) Map(iI) Map(jI) { fma }
//     |
//     |  === LOCAL STORAGE ===
//     |  (L1) InLocalStorage(for_kI, A, NV_Shared)             // A tile: TY*CY*TK
//     |  (L2) InLocalStorage(for_kI, B, NV_Shared)             // B tile: TX*CX*TK
//     |  (L3) OutLocalStorage(for_kk, C)                       // C reg tile: CY*CX
//     v
//   Map_X Map_Y { C_reg[CY*CX] = C[..],
//                 for(kk) {
//                   barriers + coop copy_A (TY*CY*TK SMEM),
//                   barriers + coop copy_B (TX*CX*TK SMEM),
//                   for(kI) { for(iI) { for(jI) { fma:
//                       C_reg[iI,jI] += A_local[iI,kI] * B_local[kI,jI] } } }
//                 },
//                 C[..] = C_reg }
//
// =============================================================================

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/offloading/sync_condition_propagation.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/passes/structured_control_flow/sequence_fusion.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/transformations/in_local_storage.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/offloading/cuda_parallelize_nested_map.h"
#include "sdfg/transformations/offloading/cuda_transform.h"
#include "sdfg/transformations/out_local_storage.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void cleanup(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& am) {
    passes::SequenceFusion sequence_fusion;
    passes::DeadCFGElimination dead_cfg;
    bool applies;
    do {
        applies = false;
        applies |= dead_cfg.run(builder, am);
        applies |= sequence_fusion.run(builder, am);
    } while (applies);
}

// Find an access node for any container whose name matches `predicate`,
// reachable from `node` (recursively). Used to look up access nodes after
// CUDATransform renames pointer arguments (e.g. A -> __daisy_cuda_<id>_A).
data_flow::AccessNode* find_access_by_suffix(structured_control_flow::ControlFlowNode& node, const std::string& suffix) {
    // Walk the node recursively. We only need a single match - first wins.
    std::function<data_flow::AccessNode*(structured_control_flow::ControlFlowNode&)> walk =
        [&](structured_control_flow::ControlFlowNode& n) -> data_flow::AccessNode* {
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&n)) {
            for (auto& dn : block->dataflow().nodes()) {
                if (auto* an = dynamic_cast<data_flow::AccessNode*>(&dn)) {
                    const auto& data = an->data();
                    if (data.size() >= suffix.size() &&
                        data.compare(data.size() - suffix.size(), suffix.size(), suffix) == 0) {
                        return an;
                    }
                }
            }
        } else if (auto* seq = dyn_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); ++i) {
                if (auto* hit = walk(seq->at(i).first)) {
                    return hit;
                }
            }
        } else if (auto* ifelse = dyn_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ifelse->size(); ++i) {
                if (auto* hit = walk(ifelse->at(i).first)) {
                    return hit;
                }
            }
        } else if (auto* loop_node = dyn_cast<structured_control_flow::StructuredLoop*>(&n)) {
            if (auto* hit = walk(loop_node->root())) {
                return hit;
            }
        }
        return nullptr;
    };
    // For a StructuredLoop, walk its body; for everything else, walk the node
    // itself (Sequence/Block/IfElse search recursively).
    if (auto* loop = dyn_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return walk(loop->root());
    }
    return walk(node);
}

// Find the unique container whose name starts with `prefix`.
std::string find_container_by_prefix(const sdfg::Function& fn, const std::string& prefix) {
    for (auto& name : fn.containers()) {
        if (name.size() >= prefix.size() && name.compare(0, prefix.size(), prefix) == 0) {
            return name;
        }
    }
    return {};
}

// Find the unique container name ending in `suffix` (e.g. "_A0" for the SMEM
// buffer that ILS created for the A pointer argument).
std::string find_container_by_suffix(const sdfg::Function& fn, const std::string& suffix) {
    for (auto& name : fn.containers()) {
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return name;
        }
    }
    return {};
}

// Find the first For loop under `seq`.
structured_control_flow::For* find_first_for(structured_control_flow::Sequence& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
        if (auto* f = dyn_cast<structured_control_flow::For*>(&seq.at(i).first)) {
            return f;
        }
    }
    return nullptr;
}

// Find the first Map under `seq`.
structured_control_flow::Map* find_first_map(structured_control_flow::Sequence& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
        if (auto* m = dyn_cast<structured_control_flow::Map*>(&seq.at(i).first)) {
            return m;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Matmul fixture: flat-pointer, linearized accesses, fixed bounds
// ---------------------------------------------------------------------------
//
// Builds:
//   for (i = 0; i < M; ++i)
//     for (j = 0; j < N; ++j)
//       for (k = 0; k < K; ++k)
//         C[i*N+j] = A[i*K+k] * B[k*N+j] + C[i*N+j]   // fma
//
// Dimensions are integer constants so the argument-size analysis used by
// CUDATransform can succeed without `allow_dynamic_sizes=true`.

struct MatmulFixture {
    static constexpr long M = 128;
    static constexpr long N = 128;
    static constexpr long K = 64;

    std::unique_ptr<builder::StructuredSDFGBuilder> builder;
    structured_control_flow::Map* for_i = nullptr;
    structured_control_flow::Map* for_j = nullptr;
    structured_control_flow::For* for_k = nullptr;

    void build() {
        builder = std::make_unique<builder::StructuredSDFGBuilder>("gpu_kernel_gemm", FunctionType_CPU);

        types::Scalar idx_desc(types::PrimitiveType::Int64);
        types::Scalar elem_desc(types::PrimitiveType::Double);
        types::Pointer ptr_desc(elem_desc);

        builder->add_container("i", idx_desc);
        builder->add_container("j", idx_desc);
        builder->add_container("k", idx_desc);
        builder->add_container("A", ptr_desc, /*is_argument=*/true);
        builder->add_container("B", ptr_desc, /*is_argument=*/true);
        builder->add_container("C", ptr_desc, /*is_argument=*/true);

        auto i = symbolic::symbol("i");
        auto j = symbolic::symbol("j");
        auto k = symbolic::symbol("k");
        auto M_e = symbolic::integer(M);
        auto N_e = symbolic::integer(N);
        auto K_e = symbolic::integer(K);

        auto& root = builder->subject().root();

        for_i = &builder->add_map(
            root,
            i,
            symbolic::Lt(i, M_e),
            symbolic::integer(0),
            symbolic::add(i, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        for_j = &builder->add_map(
            for_i->root(),
            j,
            symbolic::Lt(j, N_e),
            symbolic::integer(0),
            symbolic::add(j, symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        for_k = &builder->add_for(
            for_j->root(), k, symbolic::Lt(k, K_e), symbolic::integer(0), symbolic::add(k, symbolic::integer(1))
        );

        // fma: C[i*N+j] = A[i*K+k] * B[k*N+j] + C[i*N+j]
        auto& block = builder->add_block(for_k->root());
        auto& a_in = builder->add_access(block, "A");
        auto& b_in = builder->add_access(block, "B");
        auto& c_in = builder->add_access(block, "C");
        auto& c_out = builder->add_access(block, "C");

        auto& tasklet = builder->add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});

        builder
            ->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::add(symbolic::mul(i, K_e), k)}, ptr_desc);
        builder
            ->add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::add(symbolic::mul(k, N_e), j)}, ptr_desc);
        builder
            ->add_computational_memlet(block, c_in, tasklet, "_in3", {symbolic::add(symbolic::mul(i, N_e), j)}, ptr_desc);
        builder
            ->add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::add(symbolic::mul(i, N_e), j)}, ptr_desc);
    }
};

} // namespace

// =============================================================================
// Test
// =============================================================================

TEST(GPUKernelTest, GEMM_CudaTilingILS) {
    // Thread block & register-tile geometry.
    //   Block tile  = TY*CY rows  x  TX*CX cols  =  16 x 32
    //   Grid        = M/(TY*CY) x N/(TX*CX)      =   8 x  4
    //   Reg tile    = CY x CX per thread         =   4 x  4
    //   SMEM A tile = TY*CY x TK                 =  16 x  8 = 128 elems
    //   SMEM B tile = TX*CX x TK                 =  32 x  8 = 256 elems
    constexpr int TX = 8; // threads per block, X (coalesced along j)
    constexpr int TY = 4; // threads per block, Y
    constexpr int CX = 4; // register-tile cols per thread
    constexpr int CY = 4; // register-tile rows per thread
    constexpr int TK = 8; // K-tile depth

    MatmulFixture fix;
    fix.build();
    auto& builder = *fix.builder;
    analysis::AnalysisManager am(builder.subject());

    // -------------------------------------------------------------------------
    // Phase 1: TILING. Strip-mine i, j, k. We tile all three before doing any
    // structural reordering so the rest of the pipeline can refer to all six
    // resulting loops (iO/iI, jO/jI, kk/kI) by name.
    //
    //   Why tile BEFORE offload: once Map(i)/Map(j) are bound to CUDA grid
    //   dims, there is no transform that splits them back into a "block
    //   loop + sequential thread-tile loop" pair. The inner loops (iI, jI)
    //   need to end up INSIDE the k-traversal so each thread accumulates a
    //   CY*CX register tile of C via an outer product over k — the
    //   structural pattern that defines a CUTLASS-style GEMM.
    //
    //   LoopTiling renames the OUTER loop to "<orig>_tile0" and leaves the
    //   original indvar on the INNER per-thread tile loop. So after T1+T2+T3
    //   the indvars are:
    //         iO="i_tile0", iI="i", jO="j_tile0", jI="j",
    //         kk="k_tile0", kI="k"
    //
    //   Structure after this phase:
    //         iO { iI { jO { jI { kk { kI { fma } } } } } }
    // -------------------------------------------------------------------------
    // (T1) LoopTiling(for_i, CY)
    {
        transformations::LoopTiling tile_i(*fix.for_i, /*tile_size=*/CY);
        ASSERT_TRUE(tile_i.can_be_applied(builder, am)) << "LoopTiling(for_i, CY) must apply";
        tile_i.apply(builder, am);
        am.invalidate_all();
    }
    // (T2) LoopTiling(for_j, CX): for_j now sits inside the new iI.
    {
        auto* iO_re = find_first_map(builder.subject().root());
        ASSERT_NE(iO_re, nullptr);
        auto* iI_re = find_first_map(iO_re->root());
        ASSERT_NE(iI_re, nullptr);
        auto* j_re = find_first_map(iI_re->root());
        ASSERT_NE(j_re, nullptr);
        EXPECT_EQ(j_re->indvar()->get_name(), "j");

        transformations::LoopTiling tile_j(*j_re, /*tile_size=*/CX);
        ASSERT_TRUE(tile_j.can_be_applied(builder, am)) << "LoopTiling(for_j, CX) must apply";
        tile_j.apply(builder, am);
        am.invalidate_all();
    }
    // (T3) LoopTiling(for_k, TK): for_k now sits at iO { iI { jO { jI { for_k } } } }.
    {
        auto* iO_re = find_first_map(builder.subject().root());
        auto* iI_re = find_first_map(iO_re->root());
        auto* jO_re = find_first_map(iI_re->root());
        auto* jI_re = find_first_map(jO_re->root());
        auto* k_re = find_first_for(jI_re->root());
        ASSERT_NE(k_re, nullptr);
        EXPECT_EQ(k_re->indvar()->get_name(), "k");

        transformations::LoopTiling tile_k(*k_re, /*tile_size=*/TK);
        ASSERT_TRUE(tile_k.can_be_applied(builder, am)) << "LoopTiling(for_k, TK) must apply";
        tile_k.apply(builder, am);
        am.invalidate_all();
    }

    // -------------------------------------------------------------------------
    // Phase 2: INTERCHANGE. Reorder the nest to the CUTLASS mainloop shape
    //   jO { iO { kk { kI { iI { jI { fma } } } } } }
    //
    // Why this exact order:
    //   * jO outermost  -> coalesced X-dim along the fast axis of row-major
    //                       C / B (after offload, jO becomes Map_X).
    //   * iO next       -> Y-dim (after offload, iO becomes Map_Y).
    //   * kk between (iO, kI) -> OLS(C) at for_kk projects C's subset over
    //                       the enclosed (iI, jI) and yields a CY*CX
    //                       per-thread register tile.
    //   * kI between (kk, iI) -> ILS(A) at for_kI produces an SMEM A tile
    //                       of shape (TY*CY) x TK; ILS(B) a tile of shape
    //                       (TX*CX) x TK. The inner (iI, jI) loops are
    //                       then a pure outer product reading
    //                       A[iI,kI]*B[kI,jI] from SMEM into the C
    //                       register tile — the CUTLASS mainloop pattern.
    // -------------------------------------------------------------------------
    // (I1) Interchange(iI, jO): iO { iI { jO { ... } } } -> iO { jO { iI { ... } } }
    {
        auto* iO_re = find_first_map(builder.subject().root());
        auto* iI_re = find_first_map(iO_re->root());
        auto* jO_re = find_first_map(iI_re->root());
        ASSERT_NE(jO_re, nullptr);
        EXPECT_EQ(jO_re->indvar()->get_name(), "j_tile0");

        transformations::LoopInterchange swap(*iI_re, *jO_re);
        ASSERT_TRUE(swap.can_be_applied(builder, am)) << "Interchange(iI, jO) must apply";
        swap.apply(builder, am);
        am.invalidate_all();
    }
    // (I2) Interchange(iO, jO): iO { jO { ... } } -> jO { iO { ... } }
    {
        auto* iO_re = find_first_map(builder.subject().root());
        auto* jO_re = find_first_map(iO_re->root());
        ASSERT_EQ(iO_re->indvar()->get_name(), "i_tile0");
        ASSERT_EQ(jO_re->indvar()->get_name(), "j_tile0");

        transformations::LoopInterchange interchange(*iO_re, *jO_re);
        ASSERT_TRUE(interchange.can_be_applied(builder, am)) << "Interchange(iO, jO) must apply";
        interchange.apply(builder, am);
        am.invalidate_all();
    }
    // After I1+I2: jO { iO { iI { jI { kk { kI { fma } } } } } }.
    auto* for_j_outer = find_first_map(builder.subject().root());
    ASSERT_NE(for_j_outer, nullptr);
    EXPECT_EQ(for_j_outer->indvar()->get_name(), "j_tile0");

    auto* for_i_inner = find_first_map(for_j_outer->root());
    ASSERT_NE(for_i_inner, nullptr);
    EXPECT_EQ(for_i_inner->indvar()->get_name(), "i_tile0");

    // (I3) Interchange(jI, kk): iI { jI { kk } } -> iI { kk { jI } }
    {
        auto* iI_re = find_first_map(for_i_inner->root());
        auto* jI_re = find_first_map(iI_re->root());
        auto* kk_re = find_first_for(jI_re->root());
        ASSERT_NE(kk_re, nullptr);
        transformations::LoopInterchange swap(*jI_re, *kk_re);
        ASSERT_TRUE(swap.can_be_applied(builder, am)) << "Interchange(jI, kk) must apply";
        swap.apply(builder, am);
        am.invalidate_all();
    }
    // (I4) Interchange(iI, kk): iI { kk { ... } } -> kk { iI { ... } }
    {
        auto* iI_re = find_first_map(for_i_inner->root());
        auto* kk_re = find_first_for(iI_re->root());
        ASSERT_NE(kk_re, nullptr);
        transformations::LoopInterchange swap(*iI_re, *kk_re);
        ASSERT_TRUE(swap.can_be_applied(builder, am)) << "Interchange(iI, kk) must apply";
        swap.apply(builder, am);
        am.invalidate_all();
    }
    // After I3+I4: kk { iI { jI { kI { fma } } } } under for_i_inner.
    // (I5) Interchange(jI, kI): iI { jI { kI } } -> iI { kI { jI } }
    {
        auto* kk_re = find_first_for(for_i_inner->root());
        auto* iI_re = find_first_map(kk_re->root());
        auto* jI_re = find_first_map(iI_re->root());
        auto* kI_re = find_first_for(jI_re->root());
        ASSERT_NE(kI_re, nullptr);
        transformations::LoopInterchange swap(*jI_re, *kI_re);
        ASSERT_TRUE(swap.can_be_applied(builder, am)) << "Interchange(jI, kI) must apply";
        swap.apply(builder, am);
        am.invalidate_all();
    }
    // (I6) Interchange(iI, kI): iI { kI { ... } } -> kI { iI { ... } }
    {
        auto* kk_re = find_first_for(for_i_inner->root());
        auto* iI_re = find_first_map(kk_re->root());
        auto* kI_re = find_first_for(iI_re->root());
        ASSERT_NE(kI_re, nullptr);
        transformations::LoopInterchange swap(*iI_re, *kI_re);
        ASSERT_TRUE(swap.can_be_applied(builder, am)) << "Interchange(iI, kI) must apply";
        swap.apply(builder, am);
        am.invalidate_all();
    }
    // After I5+I6: jO { iO { kk { kI { iI { jI { fma } } } } } }.

    // -------------------------------------------------------------------------
    // Phase 3: OFFLOAD. Bind the two block-mapped loops to CUDA grid dims.
    //   (O1) CUDATransform on Map(jO): X-dim CUDA kernel, TX=8 threads/block.
    //        jO has step CX, so the X block tile is TX*CX = 32 cols. Inserts
    //        H2D/D2H blocks around the kernel and renames pointer arguments
    //        A, B, C -> __daisy_cuda_<id>_{A,B,C}.
    //   (O2) Promote Map(iO) to the Y dim (TY=4 threads/block).
    //        iO has step CY, so the Y block tile is TY*CY = 16 rows.
    // -------------------------------------------------------------------------
    // (O1) CUDATransform(jO, TX)
    {
        cuda::CUDATransform offload(*for_j_outer, /*block_size=*/TX);
        ASSERT_TRUE(offload.can_be_applied(builder, am)) << "CUDATransform on Map(jO) must apply";
        offload.apply(builder, am);
        am.invalidate_all();
    }
    EXPECT_EQ(for_j_outer->schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(cuda::ScheduleType_CUDA::dimension(for_j_outer->schedule_type()), cuda::CUDADimension::X);
    EXPECT_TRUE(symbolic::eq(cuda::ScheduleType_CUDA::block_size(for_j_outer->schedule_type()), symbolic::integer(TX)));

    // (O2) CUDAParallelizeNestedMap(iO, TY)
    {
        transformations::CUDAParallelizeNestedMap parallelize(*for_i_inner, /*block_size=*/TY);
        ASSERT_TRUE(parallelize.can_be_applied(builder, am)) << "CUDAParallelizeNestedMap on Map(iO) must apply";
        parallelize.apply(builder, am);
        am.invalidate_all();
    }
    EXPECT_EQ(for_i_inner->schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(cuda::ScheduleType_CUDA::dimension(for_i_inner->schedule_type()), cuda::CUDADimension::Y);
    EXPECT_TRUE(symbolic::eq(cuda::ScheduleType_CUDA::block_size(for_i_inner->schedule_type()), symbolic::integer(TY)));

    // -------------------------------------------------------------------------
    // Phase 4: LOCAL STORAGE. Stage A/B in SMEM at for_kI; promote C to a
    // per-thread CY*CX REGISTER TILE at for_kk.
    //
    //   ILS at for_kI lands the cooperative load BETWEEN for_kk and for_kI;
    //   per kk iter, all threads in the block cooperatively load the
    //   (TY*CY) x TK A tile (resp. (TX*CX) x TK B tile) into SMEM once.
    //
    //   OLS at for_kk hoists the C load BEFORE for_kk and the writeback
    //   AFTER for_kk. With (iI, jI) now nested INSIDE for_kk (Phase 2),
    //   OLS projects C's index subset over those loops and materialises a
    //   CY*CX per-thread local — the canonical CUTLASS accumulator
    //   fragment — instead of the 1-elem scalar from the un-coarsened
    //   pipeline. Each thread now performs CY*CX*TK FMAs per kk-iteration
    //   while issuing only CY+CX SMEM loads per kI step (after backend
    //   scalarization of the outer product). Global-memory traffic on C
    //   drops to 2 transactions per output element (one load, one store).
    //
    //   Default OLS storage = CPU_Stack: on GPU codegen this lowers to a
    //   per-thread local (register file / local memory, at the compiler's
    //   discretion).
    // -------------------------------------------------------------------------
    auto* for_kk = find_first_for(for_i_inner->root());
    ASSERT_NE(for_kk, nullptr);
    auto* for_k_inner_tiled = find_first_for(for_kk->root());
    ASSERT_NE(for_k_inner_tiled, nullptr);
    EXPECT_EQ(for_k_inner_tiled->indvar()->get_name(), "k");

    // (L1) ILS(A, NV_Shared) at for_kI
    {
        auto* a_access = find_access_by_suffix(*for_k_inner_tiled, "_A");
        ASSERT_NE(a_access, nullptr) << "Renamed A access not found under for_k_inner_tiled";

        transformations::InLocalStorage ils_a(*for_k_inner_tiled, *a_access, types::StorageType::NV_Shared());
        ASSERT_TRUE(ils_a.can_be_applied(builder, am)) << "ILS on A must apply";
        ils_a.apply(builder, am);
        am.invalidate_all();
    }
    // (L2) ILS(B, NV_Shared) at for_kI
    {
        auto* b_access = find_access_by_suffix(*for_k_inner_tiled, "_B");
        ASSERT_NE(b_access, nullptr) << "Renamed B access not found under for_k_inner_tiled";

        transformations::InLocalStorage ils_b(*for_k_inner_tiled, *b_access, types::StorageType::NV_Shared());
        ASSERT_TRUE(ils_b.can_be_applied(builder, am)) << "ILS on B must apply";
        ils_b.apply(builder, am);
        am.invalidate_all();
    }
    // (L3) OLS(C) at for_kk — CY*CX per-thread register tile.
    {
        auto* c_access = find_access_by_suffix(*for_kk, "_C");
        ASSERT_NE(c_access, nullptr) << "Renamed C access not found under for_kk";

        transformations::OutLocalStorage ols_c(*for_kk, *c_access);
        ASSERT_TRUE(ols_c.can_be_applied(builder, am)) << "OLS on C must apply";
        ols_c.apply(builder, am);
        am.invalidate_all();
    }

    // -------------------------------------------------------------------------
    // Post-passes: SyncConditionPropagation propagates the grid-condition
    // through the kernel so that out-of-bounds threads still hit every
    // barrier but skip non-barrier work. For perfectly divisible bounds
    // (M%(TY*CY)==0, N%(TX*CX)==0) this is a no-op w.r.t. semantics, but it
    // sets the `nested_sync` schedule property and is part of the canonical
    // pipeline.
    // -------------------------------------------------------------------------
    {
        passes::SyncConditionPropagation sync_prop;
        sync_prop.run_pass(builder, am);
        am.invalidate_all();
    }

    cleanup(builder, am);

    // -------------------------------------------------------------------------
    // Verification: SMEM buffers exist with correct sizes.
    //   A_local: TY*CY rows x TK k-depths = 4*4*8 = 128 elements per block
    //   B_local: TX*CX cols x TK k-depths = 8*4*8 = 256 elements per block
    //
    // ILS names its buffer `__daisy_in_local_storage_<container>` (where
    // <container> is the renamed device arg, e.g. `__daisy_cuda_0_A`). The
    // cooperative copy indvar uses a similar suffix (`__daisy_ils_coop_...`),
    // so we look up by the buffer prefix to disambiguate.
    // -------------------------------------------------------------------------
    auto a_local = find_container_by_prefix(builder.subject(), "__daisy_in_local_storage_");
    ASSERT_FALSE(a_local.empty()) << "First SMEM container not created";
    std::string b_local;
    for (auto& name : builder.subject().containers()) {
        if (name == a_local) continue;
        if (name.compare(0, std::strlen("__daisy_in_local_storage_"), "__daisy_in_local_storage_") == 0) {
            b_local = name;
            break;
        }
    }
    ASSERT_FALSE(b_local.empty()) << "Second SMEM container not created";

    // Pin A vs B by the container suffix (which carries the original arg name).
    if (a_local.find("_B") != std::string::npos) {
        std::swap(a_local, b_local);
    }
    EXPECT_NE(a_local.find("_A"), std::string::npos) << "A buffer name: " << a_local;
    EXPECT_NE(b_local.find("_B"), std::string::npos) << "B buffer name: " << b_local;

    const auto& a_type = builder.subject().type(a_local);
    const auto& b_type = builder.subject().type(b_local);

    EXPECT_EQ(a_type.storage_type(), types::StorageType::NV_Shared());
    EXPECT_EQ(b_type.storage_type(), types::StorageType::NV_Shared());

    const auto* a_arr = dynamic_cast<const types::Array*>(&a_type);
    const auto* b_arr = dynamic_cast<const types::Array*>(&b_type);

    ASSERT_NE(a_arr, nullptr) << "A_local is not an Array";
    ASSERT_NE(b_arr, nullptr) << "B_local is not an Array";

    EXPECT_TRUE(symbolic::eq(a_arr->num_elements(), symbolic::integer(TY * CY * TK)))
        << "A_local size mismatch: got " << a_arr->num_elements()->__str__() << ", expected " << (TY * CY * TK);
    EXPECT_TRUE(symbolic::eq(b_arr->num_elements(), symbolic::integer(TX * CX * TK)))
        << "B_local size mismatch: got " << b_arr->num_elements()->__str__() << ", expected " << (TX * CX * TK);

    // -------------------------------------------------------------------------
    // Verification: C register-tile accumulator. After coarsening, OLS at
    // for_kk projects C's subset over (iI, jI), giving a CY*CX per-thread
    // local — the CUTLASS-style accumulator fragment. Storage is CPU_Stack:
    // on GPU codegen this lowers to a register-file allocation.
    // -------------------------------------------------------------------------
    auto c_local = find_container_by_prefix(builder.subject(), "__daisy_out_local_storage_");
    ASSERT_FALSE(c_local.empty()) << "C register-tile container not created";
    EXPECT_NE(c_local.find("_C"), std::string::npos) << "OLS buffer name: " << c_local;

    const auto& c_type = builder.subject().type(c_local);
    EXPECT_EQ(c_type.storage_type(), types::StorageType::CPU_Stack())
        << "C accumulator should be a per-thread local (CPU_Stack), got " << c_type.storage_type().value();

    const auto* c_arr = dynamic_cast<const types::Array*>(&c_type);
    ASSERT_NE(c_arr, nullptr) << "C_local is not an Array";
    EXPECT_TRUE(symbolic::eq(c_arr->num_elements(), symbolic::integer(CY * CX)))
        << "C_local should be a CY*CX register tile, got " << c_arr->num_elements()->__str__() << ", expected "
        << (CY * CX);

    // -------------------------------------------------------------------------
    // Loop structure: map_j(X, TX) > map_i(Y, TY) > for_kk > [prologue
    // blocks/maps..., for_kI > iI > jI > fma_block]. ILS at for_kI lands its
    // copies + barriers INSIDE for_kk, ahead of for_kI. map_i_body should
    // still hold for_kk as its only non-trivial loop child.
    // -------------------------------------------------------------------------
    auto& map_i_body = for_i_inner->root();

    // Find the (now tiled) outer kk loop under map_i_body. After ILS+cleanup,
    // it should be the only For directly under map_i_body.
    structured_control_flow::For* kk_loop = nullptr;
    for (size_t i = 0; i < map_i_body.size(); ++i) {
        if (auto* f = dyn_cast<structured_control_flow::For*>(&map_i_body.at(i).first)) {
            const std::string indvar = f->indvar()->get_name();
            if (indvar.compare(0, std::strlen("k_tile"), "k_tile") == 0) {
                kk_loop = f;
                break;
            }
        }
    }
    ASSERT_NE(kk_loop, nullptr) << "k_tile loop not found under map_i_body";

    auto& kk_body = kk_loop->root();
    EXPECT_GE(kk_body.size(), 4u) << "for_kk body should hold prologue + inner k loop";

    // SyncConditionPropagation wraps each non-barrier child of a GPU Map body
    // in an `if (map.condition()) { ... }` guard. The pass runs once per GPU
    // ancestor (Map(j) X, then Map(i) Y), so each non-barrier child of for_kk
    // ends up wrapped in TWO nested single-case IfElse nodes. Unwrap to a
    // fixpoint.
    auto unwrap_if_else = [](structured_control_flow::ControlFlowNode* node
                          ) -> structured_control_flow::ControlFlowNode* {
        while (auto* ie = dyn_cast<structured_control_flow::IfElse*>(node)) {
            if (ie->size() != 1) return nullptr;
            auto& inner = ie->at(0).first;
            if (inner.size() != 1) return nullptr;
            node = &inner.at(0).first;
        }
        return node;
    };

    // Last child of kk_body must be (a wrapper around) for_kI.
    auto* last_raw = &kk_body.at(kk_body.size() - 1).first;
    auto* last_unwrapped = unwrap_if_else(last_raw);
    ASSERT_NE(last_unwrapped, nullptr) << "last kk_body child does not unwrap to a single inner node";

    auto* last_child = dyn_cast<structured_control_flow::For*>(last_unwrapped);
    ASSERT_NE(last_child, nullptr) << "last unwrapped child is not a For";
    EXPECT_EQ(last_child->indvar()->get_name(), "k");

    // The fma body (now nested under for_kI > iI > jI) references the SMEM
    // buffers (no longer the renamed device pointers for A/B) and the C
    // register tile (no longer the C device pointer).
    auto* a_in_body = find_access_by_suffix(*last_child, a_local);
    auto* b_in_body = find_access_by_suffix(*last_child, b_local);
    EXPECT_NE(a_in_body, nullptr) << "fma body does not read A from SMEM";
    EXPECT_NE(b_in_body, nullptr) << "fma body does not read B from SMEM";

    // OLS rewrote all C accesses inside for_kk to point at C_local. The
    // device pointer "_C" (i.e. `__daisy_cuda_0_C`) should no longer appear
    // anywhere inside for_kI. C_local must appear instead.
    auto* c_dev_in_body = find_access_by_suffix(*last_child, "_C");
    EXPECT_EQ(c_dev_in_body, nullptr) << "C device pointer still present inside for_kI after OLS";

    auto* c_reg_in_body = find_access_by_suffix(*last_child, c_local);
    EXPECT_NE(c_reg_in_body, nullptr) << "fma body does not reference C register tile";

    // The C device pointer must still appear OUTSIDE for_kk — once for the
    // initial load (C_local = C[..]) and once for the writeback (C[..] =
    // C_local). Both live as siblings of for_kk under map_i_body, wrapped in
    // IfElse by SyncConditionPropagation.
    auto* c_dev_outside = find_access_by_suffix(map_i_body, "_C");
    EXPECT_NE(c_dev_outside, nullptr) << "C device pointer load/store vanished from kernel";

    // Exactly two cooperative copy Maps must exist in kk_body, possibly wrapped
    // in IfElse by SyncConditionPropagation.
    int copy_map_count = 0;
    for (size_t i = 0; i < kk_body.size(); ++i) {
        auto* unwrapped = unwrap_if_else(&kk_body.at(i).first);
        if (unwrapped && dyn_cast<structured_control_flow::Map*>(unwrapped)) {
            ++copy_map_count;
        }
    }
    EXPECT_EQ(copy_map_count, 2) << "Expected exactly two cooperative copy Maps inside for_kk";

    // Barrier blocks: at least one per staged buffer pair (before & after copy)
    // -> 4 total. Barriers are NOT wrapped (they must execute for all threads).
    int barrier_block_count = 0;
    for (size_t i = 0; i < kk_body.size(); ++i) {
        auto* blk = dyn_cast<structured_control_flow::Block*>(&kk_body.at(i).first);
        if (!blk) continue;
        for (auto& n : blk->dataflow().nodes()) {
            if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(&n)) {
                if (lib->code() == data_flow::LibraryNodeType_BarrierLocal) {
                    ++barrier_block_count;
                    break;
                }
            }
        }
    }
    EXPECT_GE(barrier_block_count, 2) << "Expected at least 2 barrier blocks (one per copy boundary pair)";
}
