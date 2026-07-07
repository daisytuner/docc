#include "sdfg/passes/schedules/expansion_pass.h"

#include <gtest/gtest.h>

#include "sdfg/data_flow/library_nodes/math/math.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(ExpansionPassTest, MeanNode_2D) {
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
    types::Tensor tensor_desc_input(types::PrimitiveType::Double, shape);
    types::Tensor tensor_desc_output(types::PrimitiveType::Double, {symbolic::integer(32)});

    auto& mean_node =
        static_cast<math::tensor::MeanNode&>(builder.add_library_node<
                                             math::tensor::MeanNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, mean_node, "X", {}, tensor_desc_input, block.debug_info());
    builder.add_computational_memlet(block, b_node, mean_node, "Y", {}, tensor_desc_output, block.debug_info());

    dump_sdfg(builder.subject(), "0.init");

    // Check inputs and outputs
    EXPECT_EQ(mean_node.inputs().size(), 2);
    EXPECT_EQ(mean_node.input(0), "Y");
    EXPECT_EQ(mean_node.input(1), "X");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::LibraryNodeExpansionPass expansion_pass;
    EXPECT_TRUE(expansion_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.expanded");
}

TEST(ExpansionPassTest, StdNode_1D) {
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
    types::Tensor tensor_desc_input(types::PrimitiveType::Double, shape);
    types::Tensor tensor_desc_output(types::PrimitiveType::Double, {});

    auto& std_node =
        static_cast<math::tensor::StdNode&>(builder.add_library_node<
                                            math::tensor::StdNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, std_node, "X", {}, tensor_desc_input, block.debug_info());
    builder.add_computational_memlet(block, b_node, std_node, "Y", {}, tensor_desc_output, block.debug_info());

    dump_sdfg(builder.subject(), "0.init");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::LibraryNodeExpansionPass expansion_pass;
    EXPECT_TRUE(expansion_pass.run(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.expanded");
}

TEST(ExpansionPassTest, Does_Not_Break_Non_Standalone) {
    builder::StructuredSDFGBuilder builder("sdfg_std", FunctionType_CPU);

    auto& sdfg = builder.subject();

    types::Scalar desc(types::PrimitiveType::Double);
    types::Pointer desc_ptr(desc);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(sdfg.root());

    auto& imm_node = builder.add_constant(block, "1.0", desc);

    auto& a_node = builder.add_access(block, "a");
    auto& b_node = builder.add_access(block, "b");

    auto& assign_tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, {"_out"}, {"_in"});

    builder.add_computational_memlet(block, imm_node, assign_tasklet, "_in", {}, block.debug_info());
    builder.add_computational_memlet(
        block, assign_tasklet, "_out", a_node, {symbolic::integer(0)}, desc_ptr, block.debug_info()
    );

    std::vector<symbolic::Expression> shape = {symbolic::integer(32)};
    std::vector<int64_t> axes = {0};
    bool keepdims = false;
    types::Tensor tensor_desc_input(types::PrimitiveType::Double, shape);
    types::Tensor tensor_desc_output(types::PrimitiveType::Double, {});

    auto& sum_node =
        static_cast<math::tensor::SumNode&>(builder.add_library_node<
                                            math::tensor::SumNode>(block, DebugInfo(), shape, axes, keepdims));

    builder.add_computational_memlet(block, a_node, sum_node, "X", {}, tensor_desc_input, block.debug_info());
    builder.add_computational_memlet(block, b_node, sum_node, "Y", {}, tensor_desc_output, block.debug_info());

    dump_sdfg(builder.subject(), "0.init");

    auto outcome = passes::expansion::expand_single_math_node(builder, block, sum_node);
    EXPECT_FALSE(outcome.expanded);
    EXPECT_FALSE(outcome.block_removed);

    dump_sdfg(builder.subject(), "1.expanded");
}

class LibNodeCollector : public visitor::ActualStructuredSDFGVisitor {
public:
    std::vector<data_flow::LibraryNode*> lib_nodes_;
    bool visit(sdfg::structured_control_flow::Block& node) override {
        auto libnodes = node.dataflow().library_nodes();
        lib_nodes_.insert(lib_nodes_.end(), libnodes.begin(), libnodes.end());
        return true;
    }
};

TEST(ExpansionPassTest, StdNode_1D_Recursive_Expansion) {
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
    builder.add_computational_memlet(block, b_node, std_node, "Y", {}, output_tensor, block.debug_info());

    dump_sdfg(builder.subject(), "0.init");

    // Check inputs and outputs
    EXPECT_EQ(std_node.inputs().size(), 2);
    EXPECT_EQ(std_node.input(1), "X");
    EXPECT_EQ(std_node.input(0), "Y");

    EXPECT_EQ(block.dataflow().nodes().size(), 3);

    sdfg.validate();
    passes::LibraryNodeExpansionPass pass;
    analysis::AnalysisManager analysis_manager(builder.subject());
    EXPECT_TRUE(pass.run_pass(builder, analysis_manager));

    dump_sdfg(builder.subject(), "1.expanded");

    LibNodeCollector collector;
    collector.dispatch(builder.subject().root());
    EXPECT_EQ(collector.lib_nodes_.size(), 2);
    EXPECT_EQ(collector.lib_nodes_.at(0)->code(), stdlib::LibraryNodeType_Malloc);
    EXPECT_EQ(collector.lib_nodes_.at(1)->code(), math::cmath::LibraryNodeType_CMath);
}
