// =============================================================================
// GPU Kernel Tests
// =============================================================================
//
// The pipeline demonstrated here is the canonical way to bring a sequential,
// flat-pointer matmul kernel onto the GPU:
//
//   sequential Map(i) Map(j) For(k)               // host-side scalar kernel
//     |
//     |  (1) LoopInterchange(i, j)                // coalesce X-dim along j
//     v
//   Map(j) Map(i) For(k)
//     |
//     |  (2) cuda::CUDATransform(map_j, BX=32)    // offload to CUDA, X-dim
//     |  (3) CUDAParallelizeNestedMap(map_i, BY)  // nested Y-dim
//     v
//   Map_X(j, BX=32) Map_Y(i, BY=8) For(k)        // kernel skeleton
//     |
//     |  (4) LoopTiling(for_k, TK=8)              // strip-mine k
//     v
//   Map_X Map_Y For(kk, step=8) For(k_in)
//     |
//     |  (5) InLocalStorage(for_k_inner, A, NV_Shared) // stage A tile in SMEM
//     |  (6) InLocalStorage(for_k_inner, B, NV_Shared) // stage B tile in SMEM
//     |  (7) OutLocalStorage(for_kk, C)                 // promote C to register
//     v
//   Map_X Map_Y { C_reg = C[..],
//                 for(kk) {
//                   barriers + coop copy_A,
//                   barriers + coop copy_B,
//                   for(k_in) [fma reads A_local, B_local; accumulates C_reg]
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
        if (auto* block = dynamic_cast<structured_control_flow::Block*>(&n)) {
            for (auto& dn : block->dataflow().nodes()) {
                if (auto* an = dynamic_cast<data_flow::AccessNode*>(&dn)) {
                    const auto& data = an->data();
                    if (data.size() >= suffix.size() &&
                        data.compare(data.size() - suffix.size(), suffix.size(), suffix) == 0) {
                        return an;
                    }
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(&n)) {
            for (size_t i = 0; i < seq->size(); ++i) {
                if (auto* hit = walk(seq->at(i).first)) {
                    return hit;
                }
            }
        } else if (auto* ifelse = dynamic_cast<structured_control_flow::IfElse*>(&n)) {
            for (size_t i = 0; i < ifelse->size(); ++i) {
                if (auto* hit = walk(ifelse->at(i).first)) {
                    return hit;
                }
            }
        } else if (auto* loop_node = dynamic_cast<structured_control_flow::StructuredLoop*>(&n)) {
            if (auto* hit = walk(loop_node->root())) {
                return hit;
            }
        }
        return nullptr;
    };
    // For a StructuredLoop, walk its body; for everything else, walk the node
    // itself (Sequence/Block/IfElse search recursively).
    if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
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
        if (auto* f = dynamic_cast<structured_control_flow::For*>(&seq.at(i).first)) {
            return f;
        }
    }
    return nullptr;
}

// Find the first Map under `seq`.
structured_control_flow::Map* find_first_map(structured_control_flow::Sequence& seq) {
    for (size_t i = 0; i < seq.size(); ++i) {
        if (auto* m = dynamic_cast<structured_control_flow::Map*>(&seq.at(i).first)) {
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
    constexpr int BX = 32; // X-dim block size (warp width, coalesced along j)
    constexpr int BY = 8; // Y-dim block size
    constexpr int TK = 8; // K-tile size

    MatmulFixture fix;
    fix.build();
    auto& builder = *fix.builder;
    analysis::AnalysisManager am(builder.subject());

    // -------------------------------------------------------------------------
    // (1) Loop interchange: i <-> j so j becomes outermost.
    //     This makes the j-dim the X-dim (coalesced along the fast axis of
    //     row-major C/B) once we offload.
    // -------------------------------------------------------------------------
    {
        transformations::LoopInterchange interchange(*fix.for_i, *fix.for_j);
        ASSERT_TRUE(interchange.can_be_applied(builder, am)) << "LoopInterchange(i, j) must apply";
        interchange.apply(builder, am);
        am.invalidate_all();
    }

    // After interchange the new outer loop body is the (former) i loop, and
    // its body still contains the unchanged k loop. Re-find references.
    auto* for_j_outer = find_first_map(builder.subject().root());
    ASSERT_NE(for_j_outer, nullptr);
    EXPECT_EQ(for_j_outer->indvar()->get_name(), "j");

    auto* for_i_inner = find_first_map(for_j_outer->root());
    ASSERT_NE(for_i_inner, nullptr);
    EXPECT_EQ(for_i_inner->indvar()->get_name(), "i");

    auto* for_k_inner = find_first_for(for_i_inner->root());
    ASSERT_NE(for_k_inner, nullptr);
    EXPECT_EQ(for_k_inner->indvar()->get_name(), "k");

    // -------------------------------------------------------------------------
    // (2) CUDATransform on Map(j): X-dim CUDA kernel, BX=32.
    //     Also inserts H2D/D2H blocks around the kernel and renames pointer
    //     arguments A, B, C -> __daisy_cuda_<id>_{A,B,C}.
    // -------------------------------------------------------------------------
    {
        cuda::CUDATransform offload(*for_j_outer, /*block_size=*/BX);
        ASSERT_TRUE(offload.can_be_applied(builder, am)) << "CUDATransform on Map(j) must apply";
        offload.apply(builder, am);
        am.invalidate_all();
    }

    // for_j_outer is unchanged in identity but the schedule type was updated.
    EXPECT_EQ(for_j_outer->schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(cuda::ScheduleType_CUDA::dimension(for_j_outer->schedule_type()), cuda::CUDADimension::X);
    EXPECT_TRUE(symbolic::eq(cuda::ScheduleType_CUDA::block_size(for_j_outer->schedule_type()), symbolic::integer(BX)));

    // -------------------------------------------------------------------------
    // (3) Promote Map(i) to the Y dimension (BY=8).
    // -------------------------------------------------------------------------
    {
        transformations::CUDAParallelizeNestedMap parallelize(*for_i_inner, /*block_size=*/BY);
        ASSERT_TRUE(parallelize.can_be_applied(builder, am)) << "CUDAParallelizeNestedMap on Map(i) must apply";
        parallelize.apply(builder, am);
        am.invalidate_all();
    }

    EXPECT_EQ(for_i_inner->schedule_type().value(), cuda::ScheduleType_CUDA::value());
    EXPECT_EQ(cuda::ScheduleType_CUDA::dimension(for_i_inner->schedule_type()), cuda::CUDADimension::Y);
    EXPECT_TRUE(symbolic::eq(cuda::ScheduleType_CUDA::block_size(for_i_inner->schedule_type()), symbolic::integer(BY)));

    // -------------------------------------------------------------------------
    // (4) Strip-mine the k loop: For(k) -> For(k_tile, step=TK) For(k).
    // -------------------------------------------------------------------------
    auto* for_k_in_kernel = find_first_for(for_i_inner->root());
    ASSERT_NE(for_k_in_kernel, nullptr);
    EXPECT_EQ(for_k_in_kernel->indvar()->get_name(), "k");

    structured_control_flow::StructuredLoop* for_kk = nullptr;
    {
        transformations::LoopTiling tile(*for_k_in_kernel, /*tile_size=*/TK);
        ASSERT_TRUE(tile.can_be_applied(builder, am)) << "LoopTiling on for_k must apply";
        tile.apply(builder, am);
        for_kk = tile.outer_loop();
        ASSERT_NE(for_kk, nullptr);
        am.invalidate_all();
    }

    // for_kk should now be the only direct loop child of for_i_inner->root().
    // LoopTiling uses find_new_name for the outer indvar, so the actual name
    // includes a disambiguating suffix ("k_tile0").
    {
        const std::string indvar = for_kk->indvar()->get_name();
        EXPECT_EQ(indvar.compare(0, std::strlen("k_tile"), "k_tile"), 0) << "Unexpected k_tile indvar: " << indvar;
    }

    // -------------------------------------------------------------------------
    // (5) Stage A tile in SMEM. ILS is applied to the *inner* k-loop (post
    //     tiling), so the cooperative load lands BETWEEN for_kk and the inner
    //     for_k. Each for_kk iteration loads exactly TK elements per i thread.
    //     After CUDATransform, A is renamed - look it up by suffix.
    // -------------------------------------------------------------------------
    auto* for_k_inner_tiled = find_first_for(for_kk->root());
    ASSERT_NE(for_k_inner_tiled, nullptr);
    EXPECT_EQ(for_k_inner_tiled->indvar()->get_name(), "k");

    {
        auto* a_access = find_access_by_suffix(*for_k_inner_tiled, "_A");
        ASSERT_NE(a_access, nullptr) << "Renamed A access not found under for_k_inner_tiled";

        transformations::InLocalStorage ils_a(*for_k_inner_tiled, *a_access, types::StorageType::NV_Shared());
        ASSERT_TRUE(ils_a.can_be_applied(builder, am)) << "ILS on A must apply";
        ils_a.apply(builder, am);
        am.invalidate_all();
    }

    // -------------------------------------------------------------------------
    // (6) Stage B tile in SMEM.
    // -------------------------------------------------------------------------
    {
        auto* b_access = find_access_by_suffix(*for_k_inner_tiled, "_B");
        ASSERT_NE(b_access, nullptr) << "Renamed B access not found under for_k_inner_tiled";

        transformations::InLocalStorage ils_b(*for_k_inner_tiled, *b_access, types::StorageType::NV_Shared());
        ASSERT_TRUE(ils_b.can_be_applied(builder, am)) << "ILS on B must apply";
        ils_b.apply(builder, am);
        am.invalidate_all();
    }

    // -------------------------------------------------------------------------
    // (7) Promote C to a per-thread register accumulator.
    //
    //     OLS applied at the for_kk scope hoists the C load OUT of for_kk and
    //     the writeback AFTER for_kk. Each thread sees C[i*N + j] (a single
    //     scalar slot per thread), so the tile collapses to a 1-element local
    //     and the entire k-traversal accumulates in a register.
    //
    //     This cuts GMEM traffic on C from 2*M*N*(K/TK) accesses to just 2*M*N
    //     (one load + one store per output element), and is the prerequisite
    //     for any further inner-loop optimization (unrolling, vectorization,
    //     tensor-core MMA), since the inner k loop is now a pure FMA chain
    //     with no DRAM round-trip on the accumulator.
    //
    //     Default storage = CPU_Stack: on GPU codegen this lowers to a
    //     per-thread local (register or stack slot, at the compiler's
    //     discretion).
    // -------------------------------------------------------------------------
    {
        auto* c_access = find_access_by_suffix(*for_kk, "_C");
        ASSERT_NE(c_access, nullptr) << "Renamed C access not found under for_kk";

        transformations::OutLocalStorage ols_c(*for_kk, *c_access);
        ASSERT_TRUE(ols_c.can_be_applied(builder, am)) << "OLS on C must apply";
        ols_c.apply(builder, am);
        am.invalidate_all();
    }

    // -------------------------------------------------------------------------
    // (8) Propagate the grid-condition through the kernel so that out-of-bounds
    //     threads still hit every barrier but skip non-barrier work. For
    //     perfectly divisible bounds (M%BY == 0, N%BX == 0) this is a no-op
    //     w.r.t. semantics, but it sets the `nested_sync` schedule property
    //     and is part of the canonical pipeline.
    // -------------------------------------------------------------------------
    {
        passes::SyncConditionPropagation sync_prop;
        sync_prop.run_pass(builder, am);
        am.invalidate_all();
    }

    cleanup(builder, am);

    // -------------------------------------------------------------------------
    // Verification: SMEM buffers exist with correct sizes.
    //   A_local: per-thread on Y (BY) x varying TK = 8 * 8 = 64 elements
    //   B_local: per-thread on X (BX) x varying TK = 32 * 8 = 256 elements
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

    EXPECT_TRUE(symbolic::eq(a_arr->num_elements(), symbolic::integer(BY * TK)))
        << "A_local size mismatch: got " << a_arr->num_elements()->__str__() << ", expected " << (BY * TK);
    EXPECT_TRUE(symbolic::eq(b_arr->num_elements(), symbolic::integer(BX * TK)))
        << "B_local size mismatch: got " << b_arr->num_elements()->__str__() << ", expected " << (BX * TK);

    // -------------------------------------------------------------------------
    // Verification: C register accumulator exists as a 1-element per-thread
    // local (CPU_Stack on GPU = register/local slot). OLS names its buffer
    // `__daisy_out_local_storage_<container>`.
    // -------------------------------------------------------------------------
    auto c_local = find_container_by_prefix(builder.subject(), "__daisy_out_local_storage_");
    ASSERT_FALSE(c_local.empty()) << "C register-accumulator container not created";
    EXPECT_NE(c_local.find("_C"), std::string::npos) << "OLS buffer name: " << c_local;

    const auto& c_type = builder.subject().type(c_local);
    EXPECT_EQ(c_type.storage_type(), types::StorageType::CPU_Stack())
        << "C accumulator should be a per-thread local (CPU_Stack), got " << c_type.storage_type().value();

    const auto* c_arr = dynamic_cast<const types::Array*>(&c_type);
    ASSERT_NE(c_arr, nullptr) << "C_local is not an Array";
    EXPECT_TRUE(symbolic::eq(c_arr->num_elements(), symbolic::integer(1)))
        << "C_local should be a single-element register accumulator, got " << c_arr->num_elements()->__str__();

    // -------------------------------------------------------------------------
    // Loop structure: map_j(X, BX) > map_i(Y, BY) > for_kk > [prologue
    // blocks/maps..., for_k_inner > fma_block]. ILS was applied to the inner
    // for_k, so the cooperative copy + barriers live INSIDE for_kk's body,
    // ahead of the inner k loop. map_i_body should still hold just for_kk as
    // its only meaningful child.
    // -------------------------------------------------------------------------
    auto& map_i_body = for_i_inner->root();

    // Find the (now tiled) outer kk loop under map_i_body. After ILS+cleanup,
    // it should be the only For directly under map_i_body.
    structured_control_flow::For* kk_loop = nullptr;
    for (size_t i = 0; i < map_i_body.size(); ++i) {
        if (auto* f = dynamic_cast<structured_control_flow::For*>(&map_i_body.at(i).first)) {
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
        while (auto* ie = dynamic_cast<structured_control_flow::IfElse*>(node)) {
            if (ie->size() != 1) return nullptr;
            auto& inner = ie->at(0).first;
            if (inner.size() != 1) return nullptr;
            node = &inner.at(0).first;
        }
        return node;
    };

    // Last child of kk_body must be (a wrapper around) the inner k loop.
    auto* last_raw = &kk_body.at(kk_body.size() - 1).first;
    auto* last_unwrapped = unwrap_if_else(last_raw);
    ASSERT_NE(last_unwrapped, nullptr) << "last kk_body child does not unwrap to a single inner node";

    auto* last_child = dynamic_cast<structured_control_flow::For*>(last_unwrapped);
    ASSERT_NE(last_child, nullptr) << "last unwrapped child is not a For";
    EXPECT_EQ(last_child->indvar()->get_name(), "k");

    // The fma body references the SMEM buffers (no longer the renamed device
    // pointers for A/B) and the C register accumulator (no longer the C device
    // pointer).
    auto* a_in_body = find_access_by_suffix(*last_child, a_local);
    auto* b_in_body = find_access_by_suffix(*last_child, b_local);
    EXPECT_NE(a_in_body, nullptr) << "fma body does not read A from SMEM";
    EXPECT_NE(b_in_body, nullptr) << "fma body does not read B from SMEM";

    // OLS rewrote all C accesses inside for_kk to point at C_local. The
    // device pointer "_C" (i.e. `__daisy_cuda_0_C`) should no longer appear
    // anywhere inside the inner k loop. C_local must appear instead.
    auto* c_dev_in_body = find_access_by_suffix(*last_child, "_C");
    EXPECT_EQ(c_dev_in_body, nullptr) << "C device pointer still present inside inner k loop after OLS";

    auto* c_reg_in_body = find_access_by_suffix(*last_child, c_local);
    EXPECT_NE(c_reg_in_body, nullptr) << "fma body does not reference C register accumulator";

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
        if (unwrapped && dynamic_cast<structured_control_flow::Map*>(unwrapped)) {
            ++copy_map_count;
        }
    }
    EXPECT_EQ(copy_map_count, 2) << "Expected exactly two cooperative copy Maps inside for_kk";

    // Barrier blocks: at least one per staged buffer pair (before & after copy)
    // -> 4 total. Barriers are NOT wrapped (they must execute for all threads).
    int barrier_block_count = 0;
    for (size_t i = 0; i < kk_body.size(); ++i) {
        auto* blk = dynamic_cast<structured_control_flow::Block*>(&kk_body.at(i).first);
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
