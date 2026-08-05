#include "sdfg/passes/loop_fusion/loop_fusion_pass.h"

#include <gtest/gtest.h>

#include "loop_info_debug_dump.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/passes/dataflow/tasklet_fusion.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/passes/redundant_load_elimination_pass.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;
using namespace sdfg::passes;
using namespace sdfg::passes::loop_fusion;

class MultiNestBuilder {
public:
    builder::StructuredSDFGBuilder& builder;
    MultiNestBuilder(builder::StructuredSDFGBuilder& builder) : builder(builder) {}

    ScheduleType sched_ = ScheduleType_Sequential::create();
    Sequence& root = builder.subject().root();

    Map& add_map(Sequence& parent, const std::string& iv, const std::string& end = "N") {
        builder.add_container(iv, types::Scalar(types::PrimitiveType::Int32));
        auto sym = symbolic::symbol(iv);
        return builder.add_map(
            parent,
            sym,
            symbolic::Lt(sym, symbolic::symbol(end)),
            symbolic::zero(),
            symbolic::add(sym, symbolic::one()),
            sched_
        );
    }
    For& add_for(Sequence& parent, const std::string& iv, const std::string& end = "N") {
        builder.add_container(iv, types::Scalar(types::PrimitiveType::Int32));
        auto sym = symbolic::symbol(iv);
        return builder.add_for(
            parent, sym, symbolic::Lt(sym, symbolic::symbol(end)), symbolic::zero(), symbolic::add(sym, symbolic::one())
        );
    }
};

TEST(LoopFusionByDomainTest, FuseMultipleStacks) {
    builder::StructuredSDFGBuilder builder("map_fuse_stacks", FunctionType_CPU);
    MultiNestBuilder m(builder);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr(scalar);
    types::Scalar itype(types::PrimitiveType::Int32);

    builder.add_container("N", scalar, true);
    builder.add_container("A", ptr, true);
    builder.add_container("_tmp_15", ptr, false);
    builder.add_container("B", ptr, true);
    builder.add_container("C", ptr, true);
    builder.add_container("ret", ptr, true);

    auto& malloc_tmp = builder.add_block(m.root);
    {
        auto& acc_tmp = builder.add_access(malloc_tmp, "_tmp_15");
        auto& malloc_node =
            builder.add_library_node<stdlib::MallocNode>(malloc_tmp, DebugInfo(), symbolic::parse("N*N*N*4"));
        builder.add_computational_memlet(malloc_tmp, malloc_node, "_ret", acc_tmp, {}, ptr);
    }

    auto& a0 = m.add_map(m.root, "a_i");
    auto& a1 = m.add_map(a0.root(), "a_j");
    auto& a2 = m.add_map(a1.root(), "a_k");
    std::vector<ElementId> a_org_ids{a0.element_id(), a1.element_id(), a2.element_id()};
    auto& a2_block = builder.add_block(a2.root());
    {
        auto& acc_B = builder.add_access(a2_block, "B");
        auto& acc_const = builder.add_constant(a2_block, "2.0", scalar);
        auto& acc_out = builder.add_access(a2_block, "_tmp_15");
        auto& tasklet = builder.add_tasklet(a2_block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(a2_block, acc_B, tasklet, "_in1", {symbolic::parse("a_k+a_j*N+a_i*N*N")}, ptr);
        builder.add_computational_memlet(a2_block, acc_const, tasklet, "_in2", {}, scalar);
        builder
            .add_computational_memlet(a2_block, tasklet, "_out", acc_out, {symbolic::parse("a_k+a_j*N+a_i*N*N")}, ptr);
    }

    auto& b0 = m.add_map(m.root, "b_i");
    auto& b1 = m.add_map(b0.root(), "b_j");
    auto& b2 = m.add_map(b1.root(), "b_k");
    auto& b2_block = builder.add_block(b2.root());
    {
        auto& acc_tmp = builder.add_access(b2_block, "_tmp_15");
        auto& acc_out = builder.add_access(b2_block, "A");
        auto& tasklet = builder.add_tasklet(b2_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(b2_block, acc_tmp, tasklet, "_in", {symbolic::parse("b_k+b_j*N+b_i*N*N")}, ptr);
        builder
            .add_computational_memlet(b2_block, tasklet, "_out", acc_out, {symbolic::parse("b_k+b_j*N+(b_i-5)*N*N")}, ptr);
    }

    auto& c0 = m.add_map(m.root, "c_i");
    auto& c1 = m.add_map(c0.root(), "c_j");
    auto& c2 = m.add_map(c1.root(), "c_k");
    std::vector<ElementId> c_org_ids{c0.element_id(), c1.element_id(), c2.element_id()};
    auto& c2_block = builder.add_block(c2.root());
    {
        auto& acc_A = builder.add_access(c2_block, "A");
        auto& acc_const = builder.add_constant(c2_block, "1.0", scalar);
        auto& acc_out = builder.add_access(c2_block, "ret");
        auto& tasklet = builder.add_tasklet(c2_block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder
            .add_computational_memlet(c2_block, acc_A, tasklet, "_in1", {symbolic::parse("c_k+c_j*N+(c_i-5)*N*N")}, ptr);
        builder.add_computational_memlet(c2_block, acc_const, tasklet, "_in2", {}, scalar);
        builder
            .add_computational_memlet(c2_block, tasklet, "_out", acc_out, {symbolic::parse("c_k+c_j*N+c_i*N*N")}, ptr);
    }

    // 4th stack, to have conflicts with the fused result of stack a-c on _tmp
    auto& d0 = m.add_map(m.root, "d_i");
    auto& d1 = m.add_map(d0.root(), "d_j");
    auto& d2 = m.add_map(d1.root(), "d_k");
    std::vector<ElementId> d_org_ids{d0.element_id(), d1.element_id(), d2.element_id()};
    auto& d2_block = builder.add_block(d2.root());
    {
        auto& acc_B = builder.add_access(d2_block, "B");
        auto& acc_const = builder.add_access(d2_block, "_tmp_15");
        auto& acc_out = builder.add_access(d2_block, "C");
        auto& tasklet = builder.add_tasklet(d2_block, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder
            .add_computational_memlet(d2_block, acc_B, tasklet, "_in1", {symbolic::parse("10+a_k+a_j*N+a_i*N*N")}, ptr);
        builder
            .add_computational_memlet(d2_block, acc_const, tasklet, "_in2", {symbolic::parse("-5+a_k+a_j*N+a_i*N*N")}, ptr);
        builder
            .add_computational_memlet(d2_block, tasklet, "_out", acc_out, {symbolic::parse("c_k+c_j*N+c_i*N*N")}, ptr);
    }

    analysis::AnalysisManager analysis_manager(builder.subject());
    auto& loops = analysis_manager.get<analysis::LoopAnalysis>();

    builder.subject().add_metadata("output_dir", "test_outputs/LoopFusionByDomainTest/FuseMultipleStacks");

    dump_loop_info(loops, "0.init");
    dump_sdfg(builder.subject(), "0.init");

    builder.subject().validate();
    analysis_manager.invalidate_all();

    LoopFusionPass pass({.map_fusion_by_domain = true, .map_fusion_by_access = false});
    pass.run_pass(builder, analysis_manager);

    auto& loops2 = analysis_manager.get<analysis::LoopAnalysis>();
    dump_loop_info(loops2, "1.fused");
    dump_sdfg(builder.subject(), "1.fused");

    auto& root = builder.subject().root();
    EXPECT_EQ(root.size(), 3);
    EXPECT_EQ(&root.at(0), &malloc_tmp);
    EXPECT_EQ(loops2.children(nullptr).size(), 2); // root loops

    EXPECT_FALSE(builder.find_element_by_id(a_org_ids.at(0))); // a0 no longer in the SDFG

    EXPECT_EQ(root.at(1).element_id(), c_org_ids.at(0));
    auto& c_outer = dynamic_cast<Map&>(root.at(1));
    EXPECT_EQ(c_outer.root().size(), 1);
    EXPECT_EQ(loops2.children(&c_outer).size(), 1);
    EXPECT_TRUE(loops2.loop_info(&c_outer).is_perfectly_nested);
    EXPECT_TRUE(loops2.loop_info(&c_outer).is_perfectly_parallel);
    EXPECT_EQ(c_outer.root().at(0).element_id(), c_org_ids.at(1));
    auto& c_middle = dynamic_cast<Map&>(c_outer.root().at(0));
    EXPECT_EQ(c_middle.root().size(), 1);
    EXPECT_EQ(c_middle.root().at(0).element_id(), c_org_ids.at(2));

    EXPECT_EQ(root.at(2).element_id(), d_org_ids.at(0));
    auto& d_outer = dynamic_cast<Map&>(root.at(2));
    EXPECT_EQ(d_outer.root().size(), 1);
    EXPECT_EQ(loops2.children(&d_outer).size(), 1);
    EXPECT_TRUE(loops2.loop_info(&d_outer).is_perfectly_nested);
    EXPECT_TRUE(loops2.loop_info(&d_outer).is_perfectly_parallel);
    EXPECT_EQ(d_outer.root().at(0).element_id(), d_org_ids.at(1));
    auto& d_middle = dynamic_cast<Map&>(d_outer.root().at(0));
    EXPECT_EQ(d_middle.root().size(), 1);
    EXPECT_EQ(d_middle.root().at(0).element_id(), d_org_ids.at(2));

    analysis_manager.invalidate_all();

    sdfg::passes::DeadDataElimination dde;
    dde.run(builder, analysis_manager);
    sdfg::passes::Pipeline dce = sdfg::passes::Pipeline::dead_code_elimination();
    dce.run(builder, analysis_manager);
    sdfg::passes::Pipeline block_fusion("BlockFusion");
    block_fusion.register_pass<sdfg::passes::BlockFusionPass>();
    block_fusion.run(builder, analysis_manager);

    dump_sdfg(builder.subject(), "2.cleanup");

    sdfg::passes::RedundantLoadEliminationPass rle;
    rle.run(builder, analysis_manager);
    dump_sdfg(builder.subject(), "3.rle");

    dde.run(builder, analysis_manager);
    sdfg::passes::TaskletFusionPass task_fuse_pass;
    task_fuse_pass.run(builder, analysis_manager);

    dump_sdfg(builder.subject(), "4.rle-cleanup");
}

TEST(LoopFusionByDomainTest, FindNonStackedNestedConflicts) {
    builder::StructuredSDFGBuilder builder("map_fuse_conflicted", FunctionType_CPU);
    MultiNestBuilder m(builder);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr(scalar);

    builder.add_container("N", scalar, true);
    builder.add_container("src", ptr, true);
    builder.add_container("dst", ptr, true);
    builder.add_container("_tmp", ptr, false);
    builder.add_container("extra", ptr, false);
    builder.add_container("extra2", ptr, false);

    auto& malloc_tmp = builder.add_block(m.root);
    {
        auto& acc_tmp = builder.add_access(malloc_tmp, "_tmp");
        auto& malloc_node =
            builder.add_library_node<stdlib::MallocNode>(malloc_tmp, DebugInfo(), symbolic::parse("N*N*4"));
        builder.add_computational_memlet(malloc_tmp, malloc_node, "_ret", acc_tmp, {}, ptr);
        auto& acc_extra = builder.add_access(malloc_tmp, "extra");
        auto& malloc_extra =
            builder.add_library_node<stdlib::MallocNode>(malloc_tmp, DebugInfo(), symbolic::parse("N*N*4"));
        builder.add_computational_memlet(malloc_tmp, malloc_extra, "_ret", acc_extra, {}, ptr);
        auto& acc_extra2 = builder.add_access(malloc_tmp, "extra2");
        auto& malloc_extra2 =
            builder.add_library_node<stdlib::MallocNode>(malloc_tmp, DebugInfo(), symbolic::parse("N*N*4"));
        builder.add_computational_memlet(malloc_tmp, malloc_extra2, "_ret", acc_extra2, {}, ptr);
    }

    // Producer: Map(a_i) { Map(a_j) { _tmp[a_i*N + a_j] = src[a_i*N + a_j] } }  (row-major write)
    auto& a0 = m.add_map(m.root, "a_i");
    auto& a1 = m.add_map(a0.root(), "a_j");
    auto& a_block = builder.add_block(a1.root());
    {
        auto& acc_src = builder.add_access(a_block, "src");
        auto& acc_tmp = builder.add_access(a_block, "_tmp");
        auto& tasklet = builder.add_tasklet(a_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(a_block, acc_src, tasklet, "_in", {symbolic::parse("a_i*N + a_j")}, ptr);
        builder.add_computational_memlet(a_block, tasklet, "_out", acc_tmp, {symbolic::parse("a_i*N + a_j")}, ptr);
    }
    auto& a_n_2 = m.add_map(a1.root(), "a_k");
    auto& a_hidden_block = builder.add_block(a_n_2.root());
    {
        auto& acc_tmp = builder.add_access(a_hidden_block, "_tmp");
        auto& acc_extra = builder.add_access(a_hidden_block, "extra");
        auto& tasklet = builder.add_tasklet(a_hidden_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(a_hidden_block, acc_tmp, tasklet, "_in", {symbolic::parse("a_i*N + a_k")}, ptr);
        builder
            .add_computational_memlet(a_hidden_block, tasklet, "_out", acc_extra, {symbolic::parse("a_k*N + a_i")}, ptr);
    }

    // Consumer: Map(b_i) { Map(b_j) { dst[b_i*N + b_j] = _tmp[b_i*N + b_j] } }  (perfectly matching read)
    auto& b0 = m.add_map(m.root, "b_i");
    auto& b1 = m.add_map(b0.root(), "b_j");
    auto& b_block = builder.add_block(b1.root());
    {
        auto& acc_tmp = builder.add_access(b_block, "_tmp");
        auto& acc_dst = builder.add_access(b_block, "dst");
        auto& tasklet = builder.add_tasklet(b_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(b_block, acc_tmp, tasklet, "_in", {symbolic::parse("b_i*N + b_j")}, ptr);
        builder.add_computational_memlet(b_block, tasklet, "_out", acc_dst, {symbolic::parse("b_i*N + b_j")}, ptr);
    }

    //  inner loop that is not part of the stack (that can be fused as one), but itself a candidate for fusing
    auto& b_n_2 = m.add_map(b1.root(), "b_k");
    auto& b_hidden_block = builder.add_block(b_n_2.root());
    {
        auto& acc_extra = builder.add_access(b_hidden_block, "extra");
        auto& acc_extra2 = builder.add_access(b_hidden_block, "extra2");
        auto& tasklet = builder.add_tasklet(b_hidden_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder
            .add_computational_memlet(b_hidden_block, acc_extra, tasklet, "_in", {symbolic::parse("b_i*N + b_k")}, ptr);
        builder
            .add_computational_memlet(b_hidden_block, tasklet, "_out", acc_extra2, {symbolic::parse("b_k*N + b_i")}, ptr);
    }

    dump_sdfg(builder.subject(), "0.init");

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis_manager.get<analysis::LoopAnalysis>();
    builder.subject().validate();
    analysis_manager.invalidate_all();

    LoopFusionPass pass({.map_fusion_by_domain = true, .map_fusion_by_access = false});
    pass.run_pass(builder, analysis_manager);

    dump_sdfg(builder.subject(), "1.after");

    // Not fused: malloc + producer stack + consumer stack all remain at root.
    EXPECT_EQ(builder.subject().root().size(), 3u)
        << "Transposed producer/consumer must NOT be fused (would create a read-before-write "
           "on the materialized intermediate _tmp).";
}


TEST(LoopFusionByDomainTest, DoNotFuseTransposedProducerConsumer) {
    // Reproducer for the segformer layernorm->reshape read-before-write.
    // Producer stack writes the malloc'd intermediate _tmp ROW-major: _tmp[i*N + j].
    // Consumer stack reads it TRANSPOSED: _tmp[j*N + i].
    //
    // Fusing these on the shared outer domain is ILLEGAL: within a single fused outer
    // iteration i the consumer reads _tmp[j*N + i] for all j (a whole column), but the
    // producer at that iteration wrote only row i (_tmp[i*N + 0:N]). For j != i the
    // consumer reads a cell the producer writes in a LATER iteration -> read-before-write
    // on the materialized intermediate. The pass must NOT fuse (both stacks stay at root).

    builder::StructuredSDFGBuilder builder("map_fuse_transpose", FunctionType_CPU);
    MultiNestBuilder m(builder);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr(scalar);

    builder.add_container("N", scalar, true);
    builder.add_container("src", ptr, true);
    builder.add_container("dst", ptr, true);
    builder.add_container("_tmp", ptr, false);

    auto& malloc_tmp = builder.add_block(m.root);
    {
        auto& acc_tmp = builder.add_access(malloc_tmp, "_tmp");
        auto& malloc_node =
            builder.add_library_node<stdlib::MallocNode>(malloc_tmp, DebugInfo(), symbolic::parse("N*N*4"));
        builder.add_computational_memlet(malloc_tmp, malloc_node, "_ret", acc_tmp, {}, ptr);
    }

    // Producer: Map(a_i) { Map(a_j) { _tmp[a_i*N + a_j] = src[a_i*N + a_j] } }  (row-major write)
    auto& a0 = m.add_map(m.root, "a_i");
    auto& a1 = m.add_map(a0.root(), "a_j");
    auto& a_block = builder.add_block(a1.root());
    {
        auto& acc_src = builder.add_access(a_block, "src");
        auto& acc_tmp = builder.add_access(a_block, "_tmp");
        auto& tasklet = builder.add_tasklet(a_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(a_block, acc_src, tasklet, "_in", {symbolic::parse("a_i*N + a_j")}, ptr);
        builder.add_computational_memlet(a_block, tasklet, "_out", acc_tmp, {symbolic::parse("a_i*N + a_j")}, ptr);
    }

    // Consumer: Map(b_i) { Map(b_j) { dst[b_i*N + b_j] = _tmp[b_j*N + b_i] } }  (TRANSPOSED read)
    auto& b0 = m.add_map(m.root, "b_i");
    auto& b1 = m.add_map(b0.root(), "b_j");
    auto& b_block = builder.add_block(b1.root());
    {
        auto& acc_tmp = builder.add_access(b_block, "_tmp");
        auto& acc_dst = builder.add_access(b_block, "dst");
        auto& tasklet = builder.add_tasklet(b_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(b_block, acc_tmp, tasklet, "_in", {symbolic::parse("b_j*N + b_i")}, ptr);
        builder.add_computational_memlet(b_block, tasklet, "_out", acc_dst, {symbolic::parse("b_i*N + b_j")}, ptr);
    }

    dump_sdfg(builder.subject(), "0.init");

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis_manager.get<analysis::LoopAnalysis>();
    builder.subject().validate();
    analysis_manager.invalidate_all();

    LoopFusionPass pass({.map_fusion_by_domain = true, .map_fusion_by_access = false});
    pass.run_pass(builder, analysis_manager);

    dump_sdfg(builder.subject(), "1.after");

    // Not fused: malloc + producer stack + consumer stack all remain at root.
    EXPECT_EQ(builder.subject().root().size(), 3u)
        << "Transposed producer/consumer must NOT be fused (would create a read-before-write "
           "on the materialized intermediate _tmp).";
}

TEST(LoopFusionByDomainTest, DoNotFuseTransposedSharedBand_Segformer) {
    // Faithful reproducer of the segformer layernorm->reshape->transpose kernel (rid 764639257).
    // Ground-truth structure from the emitted cutout:
    //
    //   Map(X<16) { Map(r<256) {                       // shared outer band
    //     Map(c<256)        { _tmp[65536*X + 256*r + c]          = ... }   // producer: writes ROW r
    //     Map(u<16){Map(v<16){ dst[...] = _tmp[65536*X + r + 4096*u + 256*v] }}}}  // consumer: reads COLUMN r
    //
    // _tmp is a [16][256][256] transient. The producer writes it row-major ([X][r][c]);
    // the consumer reads it transposed ([X][w][r], w = 16u+v). Fusing on the r loop is ILLEGAL:
    // at fused iteration (X, r) the consumer reads _tmp[65536*X + 256*w + r] for all w in [0,256),
    // i.e. a whole COLUMN spanning every row, but the producer has only written ROW r so far.
    // For w != r those cells are written in LATER r-iterations -> read-before-write.
    //
    // Note the fused dim r has DIFFERENT strides in _tmp: 256 in the producer write, 1 in the
    // consumer read. The pass must detect this subset mismatch and NOT fuse.
    //
    // This differs from DoNotFuseTransposedProducerConsumer by faithfully modelling the
    // uneven inner depth (producer 1 inner loop, consumer 2) that the real kernel has.

    builder::StructuredSDFGBuilder builder("map_fuse_transpose_band", FunctionType_CPU);
    MultiNestBuilder m(builder);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr(scalar);

    builder.add_container("BX", scalar, true); // 16
    builder.add_container("BR", scalar, true); // 256
    builder.add_container("BC", scalar, true); // 256
    builder.add_container("BU", scalar, true); // 16
    builder.add_container("BV", scalar, true); // 16
    builder.add_container("src", ptr, true);
    builder.add_container("dst", ptr, true);
    builder.add_container("_tmp", ptr, false);

    auto& malloc_tmp = builder.add_block(m.root);
    {
        auto& acc_tmp = builder.add_access(malloc_tmp, "_tmp");
        auto& malloc_node =
            builder.add_library_node<stdlib::MallocNode>(malloc_tmp, DebugInfo(), symbolic::parse("BX*BR*BC*4"));
        builder.add_computational_memlet(malloc_tmp, malloc_node, "_ret", acc_tmp, {}, ptr);
    }

    // Producer: Map(a_x) { Map(a_r) { Map(a_c) { _tmp[65536*a_x + 256*a_r + a_c] = src[...] } } }
    auto& a0 = m.add_map(m.root, "a_x", "BX");
    auto& a1 = m.add_map(a0.root(), "a_r", "BR");
    auto& a2 = m.add_map(a1.root(), "a_c", "BC");
    auto& a_block = builder.add_block(a2.root());
    {
        auto& acc_src = builder.add_access(a_block, "src");
        auto& acc_tmp = builder.add_access(a_block, "_tmp");
        auto& tasklet = builder.add_tasklet(a_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(
            a_block, acc_src, tasklet, "_in", {symbolic::parse("65536*a_x + 256*a_r + a_c")}, ptr
        );
        builder.add_computational_memlet(
            a_block, tasklet, "_out", acc_tmp, {symbolic::parse("65536*a_x + 256*a_r + a_c")}, ptr
        );
    }

    // Consumer: Map(b_x) { Map(b_r) { Map(b_u) { Map(b_v) {
    //     dst[65536*b_x + 256*b_r + 16*b_u + b_v] = _tmp[65536*b_x + b_r + 4096*b_u + 256*b_v]  (TRANSPOSED read)
    // } } } }
    auto& b0 = m.add_map(m.root, "b_x", "BX");
    auto& b1 = m.add_map(b0.root(), "b_r", "BR");
    auto& b2 = m.add_map(b1.root(), "b_u", "BU");
    auto& b3 = m.add_map(b2.root(), "b_v", "BV");
    auto& b_block = builder.add_block(b3.root());
    {
        auto& acc_tmp = builder.add_access(b_block, "_tmp");
        auto& acc_dst = builder.add_access(b_block, "dst");
        auto& tasklet = builder.add_tasklet(b_block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(
            b_block, acc_tmp, tasklet, "_in", {symbolic::parse("65536*b_x + b_r + 4096*b_u + 256*b_v")}, ptr
        );
        builder.add_computational_memlet(
            b_block, tasklet, "_out", acc_dst, {symbolic::parse("65536*b_x + 256*b_r + 16*b_u + b_v")}, ptr
        );
    }

    dump_sdfg(builder.subject(), "0.init");

    analysis::AnalysisManager analysis_manager(builder.subject());
    analysis_manager.get<analysis::LoopAnalysis>();
    builder.subject().validate();
    analysis_manager.invalidate_all();

    LoopFusionPass pass({.map_fusion_by_domain = true, .map_fusion_by_access = false});
    pass.run_pass(builder, analysis_manager);

    dump_sdfg(builder.subject(), "1.after");

    // Not fused: malloc + producer stack + consumer stack all remain at root.
    EXPECT_EQ(builder.subject().root().size(), 3u)
        << "Segformer transposed producer/consumer must NOT be fused on the r loop (would create a "
           "read-before-write on the materialized transient _tmp).";
}

TEST(LoopFusionByDomainTest, DoNotCauseIndvarReuse) {
    builder::StructuredSDFGBuilder builder("map_fuse_stacks", FunctionType_CPU);
    MultiNestBuilder m(builder);

    auto scalar = types::Scalar(types::PrimitiveType::Float);
    auto int_scalar = types::Scalar(types::PrimitiveType::Int32);
    auto ptr = types::Pointer(scalar);

    builder.add_container("copyA_src", ptr, true);
    builder.add_container("copyA_dst", ptr, true);
    builder.add_container("copyB_src", ptr, true);
    builder.add_container("copyB_dst", ptr, true);
    builder.add_container("N", int_scalar, true);
    builder.add_container("N1", int_scalar, true);
    builder.add_container("N2", int_scalar, true);
    builder.add_container("N3", int_scalar, true);

    auto& la_0 = m.add_map(m.root, "a_i");
    auto& la_1 = m.add_map(la_0.root(), "a_j", "N1");
    auto& la_2 = m.add_map(la_1.root(), "a_k", "N2");

    auto& la_block = builder.add_block(la_2.root());

    auto& a_src = builder.add_access(la_block, "copyA_src");
    auto& a_dst = builder.add_access(la_block, "copyA_dst");
    auto& a_assign = builder.add_tasklet(la_block, data_flow::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(la_block, a_src, a_assign, "_in", {symbolic::parse("N2*N1*a_i+N2*a_j+a_k")}, ptr);
    builder.add_computational_memlet(la_block, a_assign, "_out", a_dst, {symbolic::parse("N2*N1*a_i+N2*a_j+a_k")}, ptr);


    auto& lb_0 = m.add_map(m.root, "b_i");
    auto& lb_1 = m.add_map(lb_0.root(), "b_j", "N2");
    auto& lb_2 = m.add_map(lb_1.root(), "b_k", "N3");

    auto& lb_block = builder.add_block(lb_2.root());

    auto& b_src = builder.add_access(lb_block, "copyB_src");
    auto& b_dst = builder.add_access(lb_block, "copyB_dst");
    auto& b_assign = builder.add_tasklet(lb_block, data_flow::assign, {"_out"}, {"_in"});
    builder.add_computational_memlet(lb_block, b_src, b_assign, "_in", {symbolic::parse("N3*N2*b_i+N3*b_j+b_k")}, ptr);
    builder.add_computational_memlet(lb_block, b_assign, "_out", b_dst, {symbolic::parse("N3*N2*b_i+N3*b_j+b_k")}, ptr);

    dump_sdfg(builder.subject(), "0.init");

    loop_fusion::LoopFusionPass pass;
    analysis::AnalysisManager ana(builder.subject());
    pass.run_pass(builder, ana);

    dump_sdfg(builder.subject(), "1.fused");

    EXPECT_EQ(la_1.indvar()->get_name(), "a_j");
    EXPECT_EQ(la_2.indvar()->get_name(), "a_k");

    EXPECT_EQ(lb_1.indvar()->get_name(), "b_j");
    EXPECT_EQ(lb_2.indvar()->get_name(), "b_k");
}

// Minimal reproducer of the MLP softmax fusion gap.
// Nest 1 (row-max reduction):  Map i<N { tmp14[i]=init; For k<M { tmp14[i] = tmp14[i] + tmp12[i*M+k] } }
// Nest 2 (exp broadcast):      Map i<N { Map j<M { tmp16[i*M+j] = tmp12[i*M+j] - tmp14[i] } }
// Both iterate the outer row domain (N). tmp14[i] is produced per-row by nest 1 and read
// per-row (broadcast over j) by nest 2, so fusing the two outer i-maps into a single row loop
// is legal. The asymmetry is the nested structure: nest 1's inner is a sequential For (reduce,
// map_stack_depth 1) while nest 2's inner is a Map (map_stack_depth 2).
TEST(LoopFusionByDomainTest, FuseReduceThenBroadcast_Softmax) {
    builder::StructuredSDFGBuilder builder("softmax_reduce_bcast", FunctionType_CPU);
    MultiNestBuilder m(builder);

    types::Scalar scalar(types::PrimitiveType::Float);
    types::Pointer ptr(scalar);

    builder.add_container("N", scalar, true);
    builder.add_container("M", scalar, true);
    builder.add_container("tmp12", ptr, true); // input [N*M]
    builder.add_container("tmp14", ptr, false); // row reduction result [N]
    builder.add_container("tmp16", ptr, true); // output [N*M]

    // Nest 1: row reduction -> tmp14[i]
    auto& max_i = m.add_map(m.root, "mi", "N");
    {
        auto& init = builder.add_block(max_i.root());
        auto& c = builder.add_constant(init, "0.0", scalar);
        auto& t14 = builder.add_access(init, "tmp14");
        auto& tsk = builder.add_tasklet(init, data_flow::TaskletCode::assign, {"_out"}, {"_in"});
        builder.add_computational_memlet(init, c, tsk, "_in", {}, scalar);
        builder.add_computational_memlet(init, tsk, "_out", t14, {symbolic::parse("mi")}, ptr);

        auto& red = m.add_for(max_i.root(), "mk", "M");
        auto& blk = builder.add_block(red.root());
        auto& in14 = builder.add_access(blk, "tmp14");
        auto& in12 = builder.add_access(blk, "tmp12");
        auto& out14 = builder.add_access(blk, "tmp14");
        auto& tsk2 = builder.add_tasklet(blk, data_flow::TaskletCode::fp_add, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(blk, in14, tsk2, "_in1", {symbolic::parse("mi")}, ptr);
        builder.add_computational_memlet(blk, in12, tsk2, "_in2", {symbolic::parse("mi*M+mk")}, ptr);
        builder.add_computational_memlet(blk, tsk2, "_out", out14, {symbolic::parse("mi")}, ptr);
    }

    // Nest 2: exp broadcast -> tmp16[i*M+j] using tmp14[i]
    auto& exp_i = m.add_map(m.root, "ei", "N");
    auto& exp_j = m.add_map(exp_i.root(), "ej", "M");
    {
        auto& blk = builder.add_block(exp_j.root());
        auto& in12 = builder.add_access(blk, "tmp12");
        auto& in14 = builder.add_access(blk, "tmp14");
        auto& out16 = builder.add_access(blk, "tmp16");
        auto& tsk = builder.add_tasklet(blk, data_flow::TaskletCode::fp_sub, {"_out"}, {"_in1", "_in2"});
        builder.add_computational_memlet(blk, in12, tsk, "_in1", {symbolic::parse("ei*M+ej")}, ptr);
        builder.add_computational_memlet(blk, in14, tsk, "_in2", {symbolic::parse("ei")}, ptr);
        builder.add_computational_memlet(blk, tsk, "_out", out16, {symbolic::parse("ei*M+ej")}, ptr);
    }

    builder.subject().validate();

    analysis::AnalysisManager analysis_manager(builder.subject());
    LoopFusionPass pass({.map_fusion_by_domain = true, .map_fusion_by_access = false});
    pass.run_pass(builder, analysis_manager);

    auto& root = builder.subject().root();
    // The two outer i-maps should fuse into a single row loop over N.
    EXPECT_EQ(root.size(), 1) << "reduce nest and broadcast nest should fuse into one row loop";
    auto* fused = dyn_cast<structured_control_flow::Map*>(&root.at(0));
    ASSERT_TRUE(fused != nullptr);
    // Fused body: init block + reduction For + exp Map = 3 children.
    EXPECT_EQ(fused->root().size(), 3);
}
