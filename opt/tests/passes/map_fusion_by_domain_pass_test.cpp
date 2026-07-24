#include "sdfg/passes/map_fusion_by_domain_pass.h"

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

TEST(MapFusionByDomainTest, FuseMultipleStacks) {
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

    builder.subject().add_metadata("output_dir", "test_outputs/MapFusionByDomainTest/FuseMultipleStacks");

    dump_loop_info(loops, "0.init");
    dump_sdfg(builder.subject(), "0.init");

    builder.subject().validate();
    analysis_manager.invalidate_all();

    MapFusionByDomainPass pass;
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

TEST(MapFusionByDomainTest, DoNotCauseIndvarReuse) {
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

    MapFusionByDomainPass pass;
    analysis::AnalysisManager ana(builder.subject());
    pass.run_pass(builder, ana);

    dump_sdfg(builder.subject(), "1.fused");

    EXPECT_EQ(la_1.indvar()->get_name(), "a_j");
    EXPECT_EQ(la_2.indvar()->get_name(), "a_k");

    EXPECT_EQ(lb_1.indvar()->get_name(), "b_j");
    EXPECT_EQ(lb_2.indvar()->get_name(), "b_k");
}
