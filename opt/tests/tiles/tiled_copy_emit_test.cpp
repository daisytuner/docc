#include "sdfg/tiles/tiled_copy_emit.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/tiled_copy.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;
namespace scf = sdfg::structured_control_flow;

namespace {

// A minimal fixture: a global source pointer "A" and a consumer block the copy is
// inserted *before*. Each test adds its own "buf" with a matching nested-array type.
struct EmitFixture {
    builder::StructuredSDFGBuilder builder;
    types::Pointer ptr;
    scf::ControlFlowNode* consumer;

    EmitFixture() : builder("tc_emit", FunctionType_CPU), ptr(types::Scalar(types::PrimitiveType::Float)) {
        builder.add_container("A", ptr, true);
        consumer = &builder.add_block(builder.subject().root());
    }
};

// A dense row-major nested-array buffer type over @p dims (empty = scalar [1]).
std::unique_ptr<types::IType> make_buffer(const std::vector<long long>& dims) {
    types::Scalar scalar(types::PrimitiveType::Float);
    if (dims.empty()) {
        return std::make_unique<types::Array>(scalar, symbolic::integer(1));
    }
    std::unique_ptr<types::IType> inner = scalar.clone();
    for (int a = static_cast<int>(dims.size()) - 1; a >= 1; --a) {
        inner = std::make_unique<types::Array>(*inner, symbolic::integer(dims[a]));
    }
    return std::make_unique<types::Array>(*inner, symbolic::integer(dims[0]));
}

// A row-major Layout of the given extents (colex strides so mode 0 is outermost).
tiles::Layout row_major(const std::vector<long long>& dims, long long offset = 0) {
    symbolic::MultiExpression shape, stride;
    long long acc = 1;
    std::vector<symbolic::Expression> strides_rev;
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        strides_rev.push_back(symbolic::integer(acc));
        acc *= dims[i];
    }
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.push_back(symbolic::integer(dims[i]));
        stride.push_back(strides_rev[dims.size() - 1 - i]);
    }
    return tiles::Layout(shape, stride, symbolic::integer(offset));
}

tiles::Layout dense(long long n) { return row_major({n}); }

// The scope sequence emit inserted before the consumer (root child 0).
scf::Sequence& emitted_scope(EmitFixture& fx) {
    return dynamic_cast<scf::Sequence&>(fx.builder.subject().root().at(0));
}

// First direct Map child of a sequence, or nullptr.
scf::Map* first_map(scf::Sequence& s) {
    for (size_t i = 0; i < s.size(); ++i) {
        if (auto* m = dynamic_cast<scf::Map*>(&s.at(i))) return m;
    }
    return nullptr;
}

// The chain of nested maps under a scope (outermost first).
std::vector<scf::Map*> map_chain(scf::Sequence& scope) {
    std::vector<scf::Map*> maps;
    scf::Sequence* cur = &scope;
    while (auto* m = first_map(*cur)) {
        maps.push_back(m);
        cur = &m->root();
    }
    return maps;
}

// Recursively find the first Block that has an access node named @p container.
scf::Block* find_copy_block(scf::ControlFlowNode& n, const std::string& container) {
    if (auto* b = dynamic_cast<scf::Block*>(&n)) {
        for (auto& node : b->dataflow().nodes()) {
            if (auto* acc = dynamic_cast<const data_flow::AccessNode*>(&node)) {
                if (acc->data() == container) return b;
            }
        }
        return nullptr;
    }
    if (auto* s = dynamic_cast<scf::Sequence*>(&n)) {
        for (size_t i = 0; i < s->size(); ++i) {
            if (auto* r = find_copy_block(s->at(i), container)) return r;
        }
    } else if (auto* l = dynamic_cast<scf::StructuredLoop*>(&n)) {
        return find_copy_block(l->root(), container);
    } else if (auto* ife = dynamic_cast<scf::IfElse*>(&n)) {
        for (size_t i = 0; i < ife->size(); ++i) {
            if (auto* r = find_copy_block(ife->at(i).first, container)) return r;
        }
    }
    return nullptr;
}

// The memlet subset of the edge touching @p container in @p block.
data_flow::Subset subset_touching(scf::Block& block, const std::string& container) {
    for (auto& edge : block.dataflow().edges()) {
        const auto* s = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        const auto* d = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
        if ((s && s->data() == container) || (d && d->data() == container)) return edge.subset();
    }
    return {};
}

// Recursively find the first IfElse under a node.
scf::IfElse* find_if_else(scf::ControlFlowNode& n) {
    if (auto* ife = dynamic_cast<scf::IfElse*>(&n)) return ife;
    if (auto* s = dynamic_cast<scf::Sequence*>(&n)) {
        for (size_t i = 0; i < s->size(); ++i) {
            if (auto* r = find_if_else(s->at(i))) return r;
        }
    } else if (auto* l = dynamic_cast<scf::StructuredLoop*>(&n)) {
        return find_if_else(l->root());
    }
    return nullptr;
}

bool is_barrier_block(scf::ControlFlowNode& n) {
    auto* b = dynamic_cast<scf::Block*>(&n);
    if (!b) return false;
    for (auto& node : b->dataflow().nodes()) {
        if (dynamic_cast<const data_flow::BarrierLocalNode*>(&node)) return true;
    }
    return false;
}

bool subset_eq(const data_flow::Subset& a, const std::vector<symbolic::Expression>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!symbolic::eq(a[i], b[i])) return false;
    }
    return true;
}

} // namespace

// ---- base case: rank-0 (scalar) tile ---------------------------------------
// No maps: a single copy block whose global address is the layout offset and whose
// buffer address is the degenerate index 0.
TEST(TiledCopyEmitTest, Rank0_ScalarTile_CopyIn) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = tiles::Layout({}, {}, symbolic::integer(42)); // scalar at offset 42
    plan.dst = tiles::Layout();

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::In, tiles::SyncPolicy::None);

    auto& scope = emitted_scope(fx);
    EXPECT_TRUE(map_chain(scope).empty());
    auto* block = find_copy_block(scope, "buf");
    ASSERT_NE(block, nullptr);
    EXPECT_TRUE(subset_eq(subset_touching(*block, "A"), {symbolic::integer(42)}));
    EXPECT_TRUE(subset_eq(subset_touching(*block, "buf"), {symbolic::integer(0)}));
}

// ---- base case: rank-1 dense copy-in ---------------------------------------
TEST(TiledCopyEmitTest, Rank1_Dense_CopyIn) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({8});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = dense(8);
    plan.dst = dense(8);

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::In, tiles::SyncPolicy::None);

    auto& scope = emitted_scope(fx);
    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 1u);
    auto i0 = maps[0]->indvar();
    EXPECT_TRUE(symbolic::eq(maps[0]->init(), symbolic::integer(0)));
    EXPECT_TRUE(SymEngine::eq(*maps[0]->condition(), *symbolic::Lt(i0, symbolic::integer(8))));

    auto* block = find_copy_block(scope, "buf");
    ASSERT_NE(block, nullptr);
    EXPECT_TRUE(subset_eq(subset_touching(*block, "A"), {plan.src.apply_coords({i0})}));
    EXPECT_TRUE(subset_eq(subset_touching(*block, "buf"), {i0}));
}

// ---- copy-out reverses the read/write roles, same addresses ----------------
TEST(TiledCopyEmitTest, Rank1_CopyOut_ReversesRoles) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({8});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = dense(8);
    plan.dst = dense(8);

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::Out, tiles::SyncPolicy::None);

    auto& scope = emitted_scope(fx);
    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 1u);
    auto i0 = maps[0]->indvar();
    auto* block = find_copy_block(scope, "buf");
    ASSERT_NE(block, nullptr);

    // Addresses are unchanged by direction; only the memlet orientation flips.
    EXPECT_TRUE(subset_eq(subset_touching(*block, "A"), {plan.src.apply_coords({i0})}));
    EXPECT_TRUE(subset_eq(subset_touching(*block, "buf"), {i0}));

    // Out: buffer is read (src side), global is written (dst side).
    for (auto& edge : block->dataflow().edges()) {
        const auto* s = dynamic_cast<const data_flow::AccessNode*>(&edge.src());
        const auto* d = dynamic_cast<const data_flow::AccessNode*>(&edge.dst());
        if (s && s->data() == "buf") SUCCEED();
        if (d && d->data() == "A") SUCCEED();
        if (s && s->data() == "A") FAIL() << "global must be written, not read, on copy-out";
    }
}

// ---- inductive step: rank-2 nested maps, multi-dim buffer ------------------
TEST(TiledCopyEmitTest, Rank2_NestedMaps_MultiDimBuffer) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({2, 4});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = row_major({2, 4}); // strides {4,1}
    plan.dst = row_major({2, 4});

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::In, tiles::SyncPolicy::None);

    auto& scope = emitted_scope(fx);
    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 2u);
    auto i0 = maps[0]->indvar();
    auto i1 = maps[1]->indvar();
    EXPECT_TRUE(SymEngine::eq(*maps[0]->condition(), *symbolic::Lt(i0, symbolic::integer(2))));
    EXPECT_TRUE(SymEngine::eq(*maps[1]->condition(), *symbolic::Lt(i1, symbolic::integer(4))));

    auto* block = find_copy_block(scope, "buf");
    ASSERT_NE(block, nullptr);
    // Global linear address = 4*i0 + i1; buffer multi-dim address = {i0, i1}.
    EXPECT_TRUE(subset_eq(subset_touching(*block, "A"), {plan.src.apply_coords({i0, i1})}));
    EXPECT_TRUE(subset_eq(subset_touching(*block, "buf"), {i0, i1}));
}

// ---- faithfulness: a strided/offset gather addresses via apply_coords -------
// The property that would have caught the ROCm subs()-scramble bug.
TEST(TiledCopyEmitTest, StridedOffsetGather_MatchesApplyCoords) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({3, 5});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    // A non-dense gather: shape {3,5}, strides {100,1}, offset 7.
    plan.src = tiles::Layout(
        {symbolic::integer(3), symbolic::integer(5)},
        {symbolic::integer(100), symbolic::integer(1)},
        symbolic::integer(7)
    );
    plan.dst = row_major({3, 5});

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::In, tiles::SyncPolicy::None);

    auto& scope = emitted_scope(fx);
    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 2u);
    auto i0 = maps[0]->indvar();
    auto i1 = maps[1]->indvar();
    auto* block = find_copy_block(scope, "buf");
    ASSERT_NE(block, nullptr);
    // 7 + 100*i0 + i1, straight from the algebra.
    EXPECT_TRUE(subset_eq(subset_touching(*block, "A"), {plan.src.apply_coords({i0, i1})}));
}

// ---- boundary guard wraps the body in an if with the given condition -------
TEST(TiledCopyEmitTest, BoundaryGuard_WrapsBodyInIf) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({8});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = dense(8);
    plan.dst = dense(8);

    // Only copy the first 5 elements (a ragged block).
    tiles::BoundaryGuard guard = [](const std::vector<symbolic::Expression>& coords) {
        return symbolic::Le(coords[0], symbolic::integer(4));
    };

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::In, tiles::SyncPolicy::None, guard);

    auto& scope = emitted_scope(fx);
    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 1u);
    auto* ife = find_if_else(scope);
    ASSERT_NE(ife, nullptr);
    ASSERT_EQ(ife->size(), 1u);
    EXPECT_TRUE(SymEngine::eq(*ife->at(0).second, *symbolic::Le(maps[0]->indvar(), symbolic::integer(4))));
    // The copy still happens, inside the guarded case.
    EXPECT_NE(find_copy_block(scope, "buf"), nullptr);
}

// ---- a provably-true guard emits no if (the interior copy stays vectorizable) --
TEST(TiledCopyEmitTest, BoundaryGuard_TrueDropsIf) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({8});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = dense(8);
    plan.dst = dense(8);

    tiles::BoundaryGuard guard = [](const std::vector<symbolic::Expression>&) {
        return symbolic::Condition(SymEngine::boolTrue);
    };

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::In, tiles::SyncPolicy::None, guard);

    auto& scope = emitted_scope(fx);
    EXPECT_EQ(find_if_else(scope), nullptr);
    EXPECT_NE(find_copy_block(scope, "buf"), nullptr);
}

// ---- SingleStage brackets the copy with leading + trailing barriers ---------
TEST(TiledCopyEmitTest, SingleStage_WrapsWithBarriers) {
    EmitFixture fx;
    auto& root = fx.builder.subject().root();
    auto buf_type = make_buffer({4});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = dense(4);
    plan.dst = dense(4);

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    tiles::emit(fx.builder, root, *fx.consumer, plan, c, tiles::CopyDirection::In, tiles::SyncPolicy::SingleStage);

    auto& scope = emitted_scope(fx);
    // [pre-barrier][map][post-barrier] inside the scope.
    ASSERT_EQ(scope.size(), 3u);
    EXPECT_TRUE(is_barrier_block(scope.at(0)));
    EXPECT_NE(dynamic_cast<scf::Map*>(&scope.at(1)), nullptr);
    EXPECT_TRUE(is_barrier_block(scope.at(2)));
    // The consumer still follows the whole scope.
    EXPECT_EQ(&root.at(1), fx.consumer);
}

// ---- emit_into: a slot prefix prepends to the buffer address -----------------
// The mixed per-thread+cooperative case: buf[slot][tile] instead of buf[tile].
TEST(TiledCopyEmitTest, EmitInto_SlotPrefix_PrependsBufferAddress) {
    EmitFixture fx;
    fx.builder.add_container("s", types::Scalar(types::PrimitiveType::UInt64));
    auto buf_type = make_buffer({4, 8}); // [slot(4)][tile(8)]
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = dense(8);
    plan.dst = dense(8);

    auto s = symbolic::symbol("s");
    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    auto& scope = fx.builder.add_sequence_before(fx.builder.subject().root(), *fx.consumer, DebugInfo());
    // [slot(4)][tile(8)] MultiDim buffer, addressed with the slot index prefix.
    tiles::PackedBuffer dst_buffer{{symbolic::integer(4)}, {symbolic::integer(8)}, tiles::BufferKind::MultiDim};
    tiles::emit_into(fx.builder, scope, plan, c, tiles::CopyDirection::In, dst_buffer, nullptr, {s});

    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 1u);
    auto i0 = maps[0]->indvar();
    auto* block = find_copy_block(scope, "buf");
    ASSERT_NE(block, nullptr);
    // Buffer subset is the slot index followed by the tile coordinate.
    EXPECT_TRUE(subset_eq(subset_touching(*block, "buf"), {s, i0}));
    // The global gather is unaffected by the slot prefix.
    EXPECT_TRUE(subset_eq(subset_touching(*block, "A"), {plan.src.apply_coords({i0})}));
}

// ---- emit_into: the given schedule is placed on the coverage map -------------
// A GPU offload schedule turns the copy cooperative.
TEST(TiledCopyEmitTest, EmitInto_Schedule_PlacedOnCoverageMap) {
    EmitFixture fx;
    auto buf_type = make_buffer({8});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = dense(8);
    plan.dst = dense(8);

    auto sched = gpu::ScheduleType_GPU_Offload::create<
        cuda::ScheduleType_CUDA_Offload>(gpu::TargetLevel::X_BLOCK, symbolic::integer(32));
    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    auto& scope = fx.builder.add_sequence_before(fx.builder.subject().root(), *fx.consumer, DebugInfo());
    tiles::PackedBuffer dst_buffer{{}, {symbolic::integer(8)}, tiles::BufferKind::MultiDim};
    tiles::emit_into(fx.builder, scope, plan, c, tiles::CopyDirection::In, dst_buffer, &sched);

    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 1u);
    EXPECT_EQ(maps[0]->schedule_type().category(), scf::ScheduleTypeCategory::Offloader);
}

// ---- emit_into: flat coverage is a single map + row-major delinearize ---------
// The cooperative shape: a rank-2 tile becomes ONE map over the linearized tile,
// with src/buffer addressed by the delinearized coordinate.
TEST(TiledCopyEmitTest, EmitInto_FlatCoverage_SingleMapDelinearized) {
    EmitFixture fx;
    auto buf_type = make_buffer({2, 4});
    fx.builder.add_container("buf", *buf_type);

    tiles::TiledCopy plan;
    plan.src = row_major({2, 4}); // strides {4,1}
    plan.dst = row_major({2, 4});

    tiles::CopyContainers c{"A", "buf", &fx.ptr, buf_type.get()};
    auto& scope = fx.builder.add_sequence_before(fx.builder.subject().root(), *fx.consumer, DebugInfo());
    tiles::PackedBuffer dst_buffer{{}, {symbolic::integer(2), symbolic::integer(4)}, tiles::BufferKind::MultiDim};
    tiles::emit_into(
        fx.builder, scope, plan, c, tiles::CopyDirection::In, dst_buffer, nullptr, {}, {}, tiles::Coverage::Flat
    );

    // A single flat map over the whole tile (size 8), not two nested maps.
    auto maps = map_chain(scope);
    ASSERT_EQ(maps.size(), 1u);
    auto cvar = maps[0]->indvar();
    EXPECT_TRUE(SymEngine::eq(*maps[0]->condition(), *symbolic::Lt(cvar, symbolic::integer(8))));

    // Row-major delinearize: decomp = {c/4, c%4}.
    symbolic::MultiExpression decomp = {
        symbolic::div(cvar, symbolic::integer(4)), symbolic::mod(cvar, symbolic::integer(4))
    };
    auto* block = find_copy_block(scope, "buf");
    ASSERT_NE(block, nullptr);
    EXPECT_TRUE(subset_eq(subset_touching(*block, "A"), {plan.src.apply_coords(decomp)}));
    EXPECT_TRUE(subset_eq(subset_touching(*block, "buf"), decomp));
}
