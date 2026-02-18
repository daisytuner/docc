#include <gtest/gtest.h>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/mean_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/std_node.h"
#include "sdfg/data_flow/library_nodes/math/tensor/reduce_ops/sum_node.h"

using namespace sdfg;

TEST(ReduceTest, SumNode_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_sum", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32)};
    std::vector<int64_t> axes = {0};
    bool keepdims = false;
    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), {});

    auto& sum_node =
        static_cast<math::tensor::SumNode&>(builder.add_library_node<
                                            math::tensor::SumNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, sum_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, sum_node, "Y", b_node, {}, output_tensor, block.debug_info());

    // Check inputs and outputs
    EXPECT_EQ(sum_node.inputs().size(), 1);
    EXPECT_EQ(sum_node.inputs()[0], "X");
    EXPECT_EQ(sum_node.outputs().size(), 1);
    EXPECT_EQ(sum_node.outputs()[0], "Y");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(sum_node.expand(builder, analysis_manager));

    EXPECT_EQ(sdfg.root().size(), 1);
    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);
    EXPECT_EQ(new_sequence.size(), 2);

    // Init block
    {
        auto init_block = dynamic_cast<structured_control_flow::Block*>(&new_sequence.at(0).first);
        EXPECT_NE(init_block, nullptr);
        EXPECT_EQ(init_block->dataflow().nodes().size(), 3);
        EXPECT_EQ(init_block->dataflow().edges().size(), 2);
        EXPECT_EQ(init_block->dataflow().tasklets().size(), 1);

        auto init_tasklet = *init_block->dataflow().tasklets().begin();
        EXPECT_EQ(init_tasklet->code(), data_flow::TaskletCode::assign);
        auto& dataflow = init_tasklet->get_parent();

        auto& iedge = *dataflow.in_edges(*init_tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), output_tensor);

        auto src = dynamic_cast<data_flow::ConstantNode*>(&iedge.src());
        EXPECT_NE(src, nullptr);
        EXPECT_EQ(src->data(), "0.0");

        auto& oedge = *dataflow.out_edges(*init_tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), output_tensor);

        auto dst = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        EXPECT_NE(dst, nullptr);
        EXPECT_EQ(dst->data(), "b");
    }

    // Reduction loops
    {
        auto sum_loop = dynamic_cast<structured_control_flow::For*>(&new_sequence.at(1).first);
        EXPECT_NE(sum_loop, nullptr);
        EXPECT_EQ(sum_loop->root().size(), 1);

        auto reduce_block = dynamic_cast<structured_control_flow::Block*>(&sum_loop->root().at(0).first);
        EXPECT_EQ(reduce_block->dataflow().nodes().size(), 4);
        EXPECT_EQ(reduce_block->dataflow().edges().size(), 3);
        EXPECT_EQ(reduce_block->dataflow().tasklets().size(), 1);

        auto reduce_tasklet = *reduce_block->dataflow().tasklets().begin();
        EXPECT_EQ(reduce_tasklet->code(), data_flow::TaskletCode::fp_add);
        auto& dataflow = reduce_tasklet->get_parent();

        auto iedge1 = &(*dataflow.in_edges(*reduce_tasklet).begin());
        auto iedge2 = &(*(++dataflow.in_edges(*reduce_tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }
        EXPECT_EQ(iedge1->dst_conn(), "_in1");

        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(0), symbolic::symbol("_k0")));
        EXPECT_EQ(iedge1->base_type(), input_tensor);

        auto src1 = dynamic_cast<data_flow::AccessNode*>(&iedge1->src());
        EXPECT_NE(src1, nullptr);
        EXPECT_EQ(src1->data(), "a");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), output_tensor);

        auto src2 = dynamic_cast<data_flow::AccessNode*>(&iedge2->src());
        EXPECT_NE(src2, nullptr);
        EXPECT_EQ(src2->data(), "b");

        auto& oedge = *dataflow.out_edges(*reduce_tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), output_tensor);

        auto dst = dynamic_cast<data_flow::AccessNode*>(&oedge.dst());
        EXPECT_NE(dst, nullptr);
        EXPECT_EQ(dst->data(), "b");
    }
}

TEST(ReduceTest, SumNode_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_sum_2d", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32), symbolic::integer(32)};
    std::vector<int64_t> axes = {1};
    bool keepdims = false;
    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), {symbolic::integer(32)});

    auto& sum_node =
        static_cast<math::tensor::SumNode&>(builder.add_library_node<
                                            math::tensor::SumNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, sum_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, sum_node, "Y", b_node, {}, output_tensor, block.debug_info());

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(sum_node.expand(builder, analysis_manager));

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Reduction loops
    // Outer loop: dim 0 (Map)
    // Inner loop: dim 1 (For)
    {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&new_sequence.at(1).first);
        EXPECT_NE(map_loop, nullptr);

        auto for_loop = dynamic_cast<structured_control_flow::For*>(&map_loop->root().at(0).first);
        EXPECT_NE(for_loop, nullptr);

        auto reduce_block = dynamic_cast<structured_control_flow::Block*>(&for_loop->root().at(0).first);
        auto reduce_tasklet = *reduce_block->dataflow().tasklets().begin();
        auto& dataflow = reduce_tasklet->get_parent();

        auto iedge1 = &(*dataflow.in_edges(*reduce_tasklet).begin());
        auto iedge2 = &(*(++dataflow.in_edges(*reduce_tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 2);
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(0), symbolic::symbol("_i0")));
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(1), symbolic::symbol("_k0")));

        // Output subset: i0
        auto& oedge = *dataflow.out_edges(*reduce_tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), symbolic::symbol("_i0")));
    }
}

TEST(ReduceTest, SumNode_2D_KeepDims) {
    builder::StructuredSDFGBuilder builder("sdfg_sum_2d_keepdims", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32), symbolic::integer(32)};
    std::vector<int64_t> axes = {1};
    bool keepdims = true;
    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), {symbolic::integer(32), symbolic::one()});

    auto& sum_node =
        static_cast<math::tensor::SumNode&>(builder.add_library_node<
                                            math::tensor::SumNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, sum_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, sum_node, "Y", b_node, {}, output_tensor, block.debug_info());

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(sum_node.expand(builder, analysis_manager));

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Reduction loops
    // Outer loop: dim 0 (Map)
    // Inner loop: dim 1 (For)
    {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&new_sequence.at(1).first);
        EXPECT_NE(map_loop, nullptr);

        auto for_loop = dynamic_cast<structured_control_flow::For*>(&map_loop->root().at(0).first);
        EXPECT_NE(for_loop, nullptr);

        auto reduce_block = dynamic_cast<structured_control_flow::Block*>(&for_loop->root().at(0).first);
        auto reduce_tasklet = *reduce_block->dataflow().tasklets().begin();
        auto& dataflow = reduce_tasklet->get_parent();

        auto iedge1 = &(*dataflow.in_edges(*reduce_tasklet).begin());
        auto iedge2 = &(*(++dataflow.in_edges(*reduce_tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        // Input subset: i0, k0
        EXPECT_EQ(iedge1->subset().size(), 2);
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(0), symbolic::symbol("_i0")));
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(1), symbolic::symbol("_k0")));

        // Output subset: i0, 0
        auto& oedge = *dataflow.out_edges(*reduce_tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 2);
        EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), symbolic::symbol("_i0")));
        EXPECT_TRUE(symbolic::eq(oedge.subset().at(1), symbolic::zero()));
    }
}

TEST(ReduceTest, SumNode_3D_Reduce_0_2) {
    builder::StructuredSDFGBuilder builder("sdfg_sum_3d_reduce_0_2", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    builder.add_container("a", desc_ptr);
    builder.add_container("b", desc_ptr);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(10), symbolic::integer(20), symbolic::integer(30)};
    std::vector<int64_t> axes = {0, 2};
    bool keepdims = false;
    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), {symbolic::integer(20)});

    auto& sum_node =
        static_cast<math::tensor::SumNode&>(builder.add_library_node<
                                            math::tensor::SumNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, sum_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, sum_node, "Y", b_node, {}, output_tensor, block.debug_info());

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(sum_node.expand(builder, analysis_manager));

    auto& new_sequence = dynamic_cast<structured_control_flow::Sequence&>(sdfg.root().at(0).first);

    // Reduction loops
    // Outer loop: dim 1 (Map)
    // Inner loops: dim 0 (For), dim 2 (For)
    {
        auto map_loop = dynamic_cast<structured_control_flow::Map*>(&new_sequence.at(1).first);
        EXPECT_NE(map_loop, nullptr);

        auto for_loop_0 = dynamic_cast<structured_control_flow::For*>(&map_loop->root().at(0).first);
        EXPECT_NE(for_loop_0, nullptr);

        auto for_loop_2 = dynamic_cast<structured_control_flow::For*>(&for_loop_0->root().at(0).first);
        EXPECT_NE(for_loop_2, nullptr);

        auto reduce_block = dynamic_cast<structured_control_flow::Block*>(&for_loop_2->root().at(0).first);
        auto reduce_tasklet = *reduce_block->dataflow().tasklets().begin();
        auto& dataflow = reduce_tasklet->get_parent();

        auto iedge1 = &(*dataflow.in_edges(*reduce_tasklet).begin());
        auto iedge2 = &(*(++dataflow.in_edges(*reduce_tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 3);
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(0), symbolic::symbol("_k0")));
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(1), symbolic::symbol("_i0")));
        EXPECT_TRUE(symbolic::eq(iedge1->subset().at(2), symbolic::symbol("_k1")));

        // Output subset: i1
        auto& oedge = *dataflow.out_edges(*reduce_tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(symbolic::eq(oedge.subset().at(0), symbolic::symbol("_i0")));
    }
}

TEST(ReduceTest, MeanNode_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_sum", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32)};
    std::vector<int64_t> axes = {0};
    bool keepdims = false;
    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), {});

    auto& mean_node =
        static_cast<math::tensor::MeanNode&>(builder.add_library_node<
                                             math::tensor::MeanNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, mean_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, mean_node, "Y", b_node, {}, output_tensor, block.debug_info());

    // Check inputs and outputs
    EXPECT_EQ(mean_node.inputs().size(), 1);
    EXPECT_EQ(mean_node.inputs()[0], "X");
    EXPECT_EQ(mean_node.outputs().size(), 1);
    EXPECT_EQ(mean_node.outputs()[0], "Y");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(mean_node.expand(builder, analysis_manager));

    EXPECT_EQ(sdfg.root().size(), 3);
    auto& sum_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(0).first);
    EXPECT_EQ(sum_block.dataflow().nodes().size(), 3);
    EXPECT_EQ(sum_block.dataflow().edges().size(), 2);
    EXPECT_EQ(sum_block.dataflow().library_nodes().size(), 1);

    auto& count_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(1).first);
    EXPECT_EQ(count_block.dataflow().nodes().size(), 0);
    auto& count_transition = sdfg.root().at(1).second;
    EXPECT_EQ(count_transition.assignments().size(), 1);
    auto count_var = count_transition.assignments().begin()->first;
    auto count_expr = count_transition.assignments().begin()->second;
    EXPECT_TRUE(symbolic::eq(count_expr, symbolic::integer(32)));

    auto& div_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(2).first);
    EXPECT_EQ(div_block.dataflow().nodes().size(), 4);
    EXPECT_EQ(div_block.dataflow().edges().size(), 3);
    EXPECT_EQ(div_block.dataflow().library_nodes().size(), 1);
}

TEST(ReduceTest, MeanNode_2D) {
    builder::StructuredSDFGBuilder builder("sdfg_sum", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32), symbolic::integer(16)};
    std::vector<int64_t> axes = {-1};
    bool keepdims = false;
    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), {symbolic::integer(32)});

    auto& mean_node =
        static_cast<math::tensor::MeanNode&>(builder.add_library_node<
                                             math::tensor::MeanNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, mean_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, mean_node, "Y", b_node, {}, output_tensor, block.debug_info());

    // Check inputs and outputs
    EXPECT_EQ(mean_node.inputs().size(), 1);
    EXPECT_EQ(mean_node.inputs()[0], "X");
    EXPECT_EQ(mean_node.outputs().size(), 1);
    EXPECT_EQ(mean_node.outputs()[0], "Y");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(mean_node.expand(builder, analysis_manager));

    EXPECT_EQ(sdfg.root().size(), 3);
    auto& sum_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(0).first);
    EXPECT_EQ(sum_block.dataflow().nodes().size(), 3);
    EXPECT_EQ(sum_block.dataflow().edges().size(), 2);
    EXPECT_EQ(sum_block.dataflow().library_nodes().size(), 1);

    auto& count_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(1).first);
    EXPECT_EQ(count_block.dataflow().nodes().size(), 0);
    auto& count_transition = sdfg.root().at(1).second;
    EXPECT_EQ(count_transition.assignments().size(), 1);
    auto count_var = count_transition.assignments().begin()->first;
    auto count_expr = count_transition.assignments().begin()->second;
    EXPECT_TRUE(symbolic::eq(count_expr, symbolic::integer(16)));

    auto& div_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(2).first);
    EXPECT_EQ(div_block.dataflow().nodes().size(), 4);
    EXPECT_EQ(div_block.dataflow().edges().size(), 3);
    EXPECT_EQ(div_block.dataflow().library_nodes().size(), 1);
}

TEST(ReduceTest, StdNode_1D) {
    builder::StructuredSDFGBuilder builder("sdfg_std", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    std::vector<symbolic::Expression> shape = {symbolic::integer(32)};
    std::vector<int64_t> axes = {0};
    bool keepdims = false;
    types::Tensor input_tensor(desc.primitive_type(), shape);
    types::Tensor output_tensor(desc.primitive_type(), {});

    auto& std_node =
        static_cast<math::tensor::StdNode&>(builder.add_library_node<
                                            math::tensor::StdNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, std_node, "X", {}, input_tensor, block.debug_info());
    builder.add_computational_memlet(block, std_node, "Y", b_node, {}, output_tensor, block.debug_info());

    // Check inputs and outputs
    EXPECT_EQ(std_node.inputs().size(), 1);
    EXPECT_EQ(std_node.inputs()[0], "X");
    EXPECT_EQ(std_node.outputs().size(), 1);
    EXPECT_EQ(std_node.outputs()[0], "Y");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    sdfg.validate();
    analysis::AnalysisManager analysis_manager(sdfg);
    EXPECT_TRUE(std_node.expand(builder, analysis_manager));

    EXPECT_EQ(sdfg.root().size(), 7);

    // Check first block (Pow X^2)
    auto& pow_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(1).first);
    EXPECT_EQ(pow_block.dataflow().library_nodes().size(), 1);

    // Check second block (Mean X^2)
    auto& mean_x2_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(2).first);
    EXPECT_EQ(mean_x2_block.dataflow().library_nodes().size(), 1);

    // Check last block (Sqrt)
    auto& sqrt_block = dynamic_cast<structured_control_flow::Block&>(sdfg.root().at(6).first);
    EXPECT_EQ(sqrt_block.dataflow().library_nodes().size(), 1);
}
