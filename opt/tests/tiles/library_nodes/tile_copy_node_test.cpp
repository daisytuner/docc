#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/targets/cuda/tiles/tile_copy_node.h"
#include "sdfg/tiles/layout.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/tiles/tiled_copy.h"
#include "sdfg/tiles/transformations/tile_guard_normalization.h"
#include "sdfg/types/array.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

using namespace sdfg;

namespace {
builder::StructuredSDFGBuilder make_builder() {
    return builder::StructuredSDFGBuilder("tile_copy_test", FunctionType_CPU);
}

inline data_flow::ImplementationType ImplementationType_DUMMY{"DUMMY"};

tiles::TiledCopy make_plan() {
    tiles::TiledCopy plan;
    // src carries a free symbol `i` in its offset (the per-copy base address).
    plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::symbol("i"));
    plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::CpAsync;
    return plan;
}
} // namespace

TEST(TileCopyNodeTest, ConstructAndProperties) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), ImplementationType_DUMMY, make_plan(), tiles::CopyDirection::In, 16
    ));
    EXPECT_EQ(node.code().value(), "tile_copy");
    EXPECT_EQ(node.bytes(), 16u);
    EXPECT_EQ(node.atom(), tiles::CopyAtom::CpAsync);
    EXPECT_EQ(node.direction(), tiles::CopyDirection::In);
    EXPECT_EQ(node.inputs().size(), 2u); // {_dst, _src}
    EXPECT_EQ(node.outputs().size(), 0u);
}

TEST(TileCopyNodeTest, SymbolsReportsPlanFreeSymbols) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), ImplementationType_DUMMY, make_plan(), tiles::CopyDirection::In, 16
    ));
    auto syms = node.symbols();
    EXPECT_TRUE(syms.count(symbolic::symbol("i")) == 1);
}

TEST(TileCopyNodeTest, ReplaceRewritesPlanOffset) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), ImplementationType_DUMMY, make_plan(), tiles::CopyDirection::In, 16
    ));
    node.replace(symbolic::symbol("i"), symbolic::symbol("j"));
    auto syms = node.symbols();
    EXPECT_TRUE(syms.count(symbolic::symbol("i")) == 0);
    EXPECT_TRUE(syms.count(symbolic::symbol("j")) == 1);
    EXPECT_TRUE(symbolic::eq(node.plan().src.offset(), symbolic::symbol("j")));
}

TEST(TileCopyNodeTest, Clone) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), ImplementationType_DUMMY, make_plan(), tiles::CopyDirection::Out, 8
    ));
    auto cloned = node.clone(node.element_id(), node.vertex(), node.get_parent());
    auto* clone = dynamic_cast<tiles::TileCopyNode*>(cloned.get());
    ASSERT_NE(clone, nullptr);
    EXPECT_EQ(clone->bytes(), 8u);
    EXPECT_EQ(clone->direction(), tiles::CopyDirection::Out);
    EXPECT_TRUE(symbolic::eq(clone->plan().src.offset(), symbolic::symbol("i")));
}

TEST(TileCopyNodeTest, SerializeRoundTrip) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    builder.add_library_node<
        tiles::TileCopyNode>(block, DebugInfo(), ImplementationType_DUMMY, make_plan(), tiles::CopyDirection::In, 16);

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto restored = serializer.deserialize(j);

    auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
    size_t n = 0;
    for (auto& n_ : rblock.dataflow().nodes()) {
        if (auto* c = dynamic_cast<tiles::TileCopyNode*>(&n_)) {
            n++;
            EXPECT_EQ(c->bytes(), 16u);
            EXPECT_EQ(c->atom(), tiles::CopyAtom::CpAsync);
            EXPECT_EQ(c->direction(), tiles::CopyDirection::In);
            EXPECT_TRUE(symbolic::eq(c->plan().src.offset(), symbolic::symbol("i")));
            EXPECT_EQ(c->plan().src.dims(), 1);
        }
    }
    EXPECT_EQ(n, 1u);
}

TEST(TileCopyNodeTest, ReferenceDispatcherEmitsSequentialLoop) {
    auto builder = make_builder();
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Array buf_type(elem, symbolic::integer(4));
    builder.add_container("g", ptr);
    builder.add_container("buf", buf_type);

    auto& block = builder.add_block(builder.subject().root());
    auto& g = builder.add_access(block, "g");
    auto& buf = builder.add_access(block, "buf");
    // src = global geometry (offset c), dst = buffer geometry (offset c); direction In
    // wires _dst<-buf, _src<-g.
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::ScalarSync;
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 4
    ));
    builder.add_computational_memlet(block, buf, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, g, node, "_src", {}, ptr);

    codegen::CLanguageExtension le(builder.subject());
    tiles::TileCopyNodeDispatcher dispatcher(le, builder.subject(), block.dataflow(), node);
    codegen::PrettyPrinter stream, globals;
    codegen::CodeSnippetFactory snippets;
    dispatcher.dispatch(stream, globals, snippets);

    const std::string code = stream.str();
    EXPECT_NE(code.find("for (long long __tc_i0 = 0; __tc_i0 < 4; ++__tc_i0)"), std::string::npos) << code;
    EXPECT_NE(code.find("] = ((float *)"), std::string::npos) << code;
}

TEST(TileCopyNodeTest, MultiDimRowMajorAddressing) {
    // A 2x3 tile whose source has a non-dense row stride (10) distinguishes
    // row-major from colex delinearization; the buffer is dense (stride 3,1).
    auto builder = make_builder();
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("g", ptr);
    builder.add_container("buf", types::Array(elem, symbolic::integer(6)));

    auto& block = builder.add_block(builder.subject().root());
    auto& g = builder.add_access(block, "g");
    auto& buf = builder.add_access(block, "buf");
    tiles::TiledCopy plan;
    plan.src = tiles::Layout(
        {symbolic::integer(2), symbolic::integer(3)},
        {symbolic::integer(10), symbolic::integer(1)},
        symbolic::integer(0)
    );
    plan.dst = tiles::Layout(
        {symbolic::integer(2), symbolic::integer(3)}, {symbolic::integer(3), symbolic::integer(1)}, symbolic::integer(0)
    );
    plan.atom = tiles::CopyAtom::ScalarSync;
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 4
    ));
    builder.add_computational_memlet(block, buf, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, g, node, "_src", {}, ptr);

    codegen::CLanguageExtension le(builder.subject());
    tiles::TileCopyNodeDispatcher dispatcher(le, builder.subject(), block.dataflow(), node);
    codegen::PrettyPrinter stream, globals;
    codegen::CodeSnippetFactory snippets;
    dispatcher.dispatch(stream, globals, snippets);

    const std::string code = stream.str();
    // Nested per-mode loops with clean per-axis indices (no idiv/imod).
    EXPECT_NE(code.find("for (long long __tc_i0 = 0; __tc_i0 < 2; ++__tc_i0)"), std::string::npos) << code;
    EXPECT_NE(code.find("for (long long __tc_i1 = 0; __tc_i1 < 3; ++__tc_i1)"), std::string::npos) << code;
    // Row-major: the outer coord (the row of the 2x3 tile) is scaled by the src row
    // stride 10; the source address is affine in the loop indices.
    EXPECT_NE(code.find("10*"), std::string::npos) << code;
}

TEST(TileCopyNodeTest, ReferenceDispatcherEmitsBoundaryGuard) {
    auto builder = make_builder();
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("g", ptr);
    builder.add_container("buf", types::Array(elem, symbolic::integer(4)));

    auto& block = builder.add_block(builder.subject().root());
    auto& g = builder.add_access(block, "g");
    auto& buf = builder.add_access(block, "buf");
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::ScalarSync;
    // Skip elements whose tile coordinate exceeds a ragged bound (coord <= 2).
    tiles::TileGuard guard;
    guard.tile_sizes = {symbolic::integer(4)};
    guard.dims.push_back({0, symbolic::integer(0), symbolic::integer(2)});
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 4, guard
    ));
    builder.add_computational_memlet(block, buf, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, g, node, "_src", {}, ptr);

    codegen::CLanguageExtension le(builder.subject());
    tiles::TileCopyNodeDispatcher dispatcher(le, builder.subject(), block.dataflow(), node);
    codegen::PrettyPrinter stream, globals;
    codegen::CodeSnippetFactory snippets;
    dispatcher.dispatch(stream, globals, snippets);

    const std::string code = stream.str();
    EXPECT_NE(code.find("if ("), std::string::npos) << code;
    EXPECT_NE(code.find("__tc_i0 <= 2"), std::string::npos) << code;
}

TEST(TileCopyNodeTest, GuardSurvivesReplaceAndRoundTrip) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    tiles::TileGuard guard;
    guard.tile_sizes = {symbolic::integer(4)};
    guard.dims.push_back({0, symbolic::symbol("i"), symbolic::integer(7)}); // base carries `i`
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), ImplementationType_DUMMY, make_plan(), tiles::CopyDirection::In, 16, guard
    ));
    // `i` appears in both the plan offset and the guard.
    EXPECT_TRUE(node.symbols().count(symbolic::symbol("i")) == 1);
    node.replace(symbolic::symbol("i"), symbolic::symbol("j"));
    ASSERT_EQ(node.guard().dims.size(), 1u);
    EXPECT_TRUE(symbolic::eq(node.guard().dims[0].base, symbolic::symbol("j")));
    EXPECT_TRUE(symbolic::eq(node.guard().dims[0].max, symbolic::integer(7)));

    serializer::JSONSerializer serializer;
    auto j = serializer.serialize(builder.subject());
    auto restored = serializer.deserialize(j);
    auto& rblock = static_cast<structured_control_flow::Block&>(restored->root().at(0));
    for (auto& n_ : rblock.dataflow().nodes()) {
        if (auto* c = dynamic_cast<tiles::TileCopyNode*>(&n_)) {
            ASSERT_EQ(c->guard().dims.size(), 1u);
            EXPECT_TRUE(symbolic::eq(c->guard().dims[0].base, symbolic::symbol("j")));
            EXPECT_TRUE(symbolic::eq(c->guard().dims[0].max, symbolic::integer(7)));
        }
    }
}

// A ragged boundary dim discharges once assumptions bound its worst case in range.
TEST(TileCopyNodeTest, NormalizeGuardDischargesProvenFullDim) {
    auto builder = make_builder();
    auto& block = builder.add_block(builder.subject().root());
    auto b = symbolic::symbol("b");
    auto M = symbolic::symbol("M");
    tiles::TileGuard guard;
    guard.tile_sizes = {symbolic::integer(4)};
    guard.dims.push_back({0, b, M}); // b + coord <= M, worst case b + 3
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), ImplementationType_DUMMY, make_plan(), tiles::CopyDirection::In, 16, guard
    ));
    ASSERT_FALSE(node.guard().trivial());

    // Under b <= M - 3, the worst-case element b + 3 <= M, so the dim is redundant.
    symbolic::Assumptions assums;
    symbolic::Assumption ab(SymEngine::rcp_static_cast<const SymEngine::Symbol>(b));
    ab.add_lower_bound(symbolic::integer(0));
    ab.tight_lower_bound(symbolic::integer(0));
    ab.add_upper_bound(symbolic::sub(M, symbolic::integer(3)));
    ab.tight_upper_bound(symbolic::sub(M, symbolic::integer(3)));
    assums.insert_or_assign(SymEngine::rcp_static_cast<const SymEngine::Symbol>(b), ab);
    symbolic::SymbolSet params;
    params.insert(SymEngine::rcp_static_cast<const SymEngine::Symbol>(M));

    EXPECT_TRUE(node.normalize_guard(params, assums));
    EXPECT_TRUE(node.guard().trivial());
}

// The normalization pass discharges a node's guard once the enclosing loop bound
// proves the tile in range — the shape loop peeling leaves for a full-tile body.
TEST(TileCopyNodeTest, GuardNormalizationPassDischargesUnderLoopBound) {
    builder::StructuredSDFGBuilder builder("tgn_loop", FunctionType_CPU);
    types::Scalar i64(types::PrimitiveType::Int64);
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("N", i64, true);
    builder.add_container("i", i64);
    builder.add_container("g", ptr, true);
    builder.add_container("buf", types::Array(elem, symbolic::integer(4)));

    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    // for (i = 0; i < N - 3; ++i): the body carries i <= N - 4.
    auto& loop = builder.add_for(
        builder.subject().root(),
        i,
        symbolic::Lt(i, symbolic::sub(N, symbolic::integer(3))),
        symbolic::integer(0),
        symbolic::add(i, symbolic::integer(1))
    );
    auto& block = builder.add_block(loop.root());
    auto& g = builder.add_access(block, "g");
    auto& buf = builder.add_access(block, "buf");
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, i); // tile base i
    plan.dst = tiles::Layout({symbolic::integer(4)}, {symbolic::integer(1)}, symbolic::integer(0));
    tiles::TileGuard guard;
    guard.tile_sizes = {symbolic::integer(4)};
    guard.dims.push_back({0, i, symbolic::sub(N, symbolic::integer(1))}); // i + coord <= N - 1
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), ImplementationType_DUMMY, plan, tiles::CopyDirection::In, 4, guard
    ));
    builder.add_computational_memlet(block, buf, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, g, node, "_src", {}, ptr);
    ASSERT_FALSE(node.guard().trivial());

    analysis::AnalysisManager am(builder.subject());
    tiles::TileGuardNormalization pass;
    EXPECT_TRUE(pass.run(builder, am));
    EXPECT_TRUE(node.guard().trivial());
}

TEST(TileCopyNodeTest, CudaCooperativeDispatcherEmitsThreadStridedLoop) {
    auto builder = make_builder();
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Array buf_type(elem, symbolic::integer(64));
    builder.add_container("g", ptr);
    builder.add_container("buf", buf_type);

    auto& block = builder.add_block(builder.subject().root());
    auto& g = builder.add_access(block, "g");
    auto& buf = builder.add_access(block, "buf");
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(64)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.dst = tiles::Layout({symbolic::integer(64)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::ScalarSync;
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 4
    ));
    builder.add_computational_memlet(block, buf, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, g, node, "_src", {}, ptr);

    codegen::CUDALanguageExtension le(builder.subject());
    cuda::tiles::TileCopyNodeDispatcher dispatcher(le, builder.subject(), block.dataflow(), node);
    codegen::PrettyPrinter stream, globals;
    codegen::CodeSnippetFactory snippets;
    dispatcher.dispatch(stream, globals, snippets);

    const std::string code = stream.str();
    EXPECT_NE(code.find("threadIdx.x + threadIdx.y * (blockDim.x)"), std::string::npos) << code;
    EXPECT_NE(code.find("for (int __tc_c = __tc_tid; __tc_c < 64; __tc_c += __tc_n)"), std::string::npos) << code;
    EXPECT_NE(code.find("] = (reinterpret_cast<float *>"), std::string::npos) << code;
}

// With a known cooperating thread count, the loop gets a fixed (unrollable) trip
// count over `__tc_i` instead of the runtime thread-strided form.
TEST(TileCopyNodeTest, CudaCooperativeDispatcherEmitsFixedCountLoopWhenThreadsKnown) {
    auto builder = make_builder();
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    types::Array buf_type(elem, symbolic::integer(64));
    builder.add_container("g", ptr);
    builder.add_container("buf", buf_type);

    auto& block = builder.add_block(builder.subject().root());
    auto& g = builder.add_access(block, "g");
    auto& buf = builder.add_access(block, "buf");
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(64)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.dst = tiles::Layout({symbolic::integer(64)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::ScalarSync;
    // coop_threads = 32 (symbolic, folds) -> 64 elems / 32 threads = 2 iterations.
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block,
        DebugInfo(),
        data_flow::ImplementationType_NONE,
        plan,
        tiles::CopyDirection::In,
        4,
        tiles::TileGuard{},
        std::vector<int>{},
        symbolic::integer(32)
    ));
    builder.add_computational_memlet(block, buf, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, g, node, "_src", {}, ptr);

    codegen::CUDALanguageExtension le(builder.subject());
    cuda::tiles::TileCopyNodeDispatcher dispatcher(le, builder.subject(), block.dataflow(), node);
    codegen::PrettyPrinter stream, globals;
    codegen::CodeSnippetFactory snippets;
    dispatcher.dispatch(stream, globals, snippets);

    const std::string code = stream.str();
    EXPECT_NE(code.find("for (int __tc_i = 0; __tc_i < ((64) + (32) * 1 - 1) / ((32) * 1); __tc_i++)"), std::string::npos)
        << code;
    EXPECT_NE(code.find("__tc_c = 1 * (__tc_i * (32) + __tc_tid)"), std::string::npos) << code;
    EXPECT_NE(code.find("if (__tc_c < 64)"), std::string::npos) << code;
}

TEST(TileCopyNodeTest, CudaVectorAtomEmitsWidenedTransfer) {
    auto builder = make_builder();
    types::Scalar elem(types::PrimitiveType::Float);
    types::Pointer ptr(elem);
    builder.add_container("g", ptr);
    builder.add_container("buf", types::Array(elem, symbolic::integer(64)));

    auto& block = builder.add_block(builder.subject().root());
    auto& g = builder.add_access(block, "g");
    auto& buf = builder.add_access(block, "buf");
    tiles::TiledCopy plan;
    plan.src = tiles::Layout({symbolic::integer(64)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.dst = tiles::Layout({symbolic::integer(64)}, {symbolic::integer(1)}, symbolic::integer(0));
    plan.atom = tiles::CopyAtom::VectorSync;
    // float4: 16 bytes = 4 float elements per vector.
    auto& node = static_cast<tiles::TileCopyNode&>(builder.add_library_node<tiles::TileCopyNode>(
        block, DebugInfo(), data_flow::ImplementationType_NONE, plan, tiles::CopyDirection::In, 16
    ));
    builder.add_computational_memlet(block, buf, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, g, node, "_src", {}, ptr);

    codegen::CUDALanguageExtension le(builder.subject());
    cuda::tiles::TileCopyNodeDispatcher dispatcher(le, builder.subject(), block.dataflow(), node);
    codegen::PrettyPrinter stream, globals;
    codegen::CodeSnippetFactory snippets;
    dispatcher.dispatch(stream, globals, snippets);

    const std::string code = stream.str();
    EXPECT_NE(code.find("__tc_c = __tc_tid * 4"), std::string::npos) << code;
    EXPECT_NE(code.find("__tc_c += __tc_n * 4"), std::string::npos) << code;
    EXPECT_NE(code.find("reinterpret_cast<int4*>"), std::string::npos) << code;
}
