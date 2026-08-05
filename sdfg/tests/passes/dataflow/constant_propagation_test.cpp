#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/passes/dataflow/constant_propagation.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"

using namespace sdfg;

namespace {

// Builds a `ConstantNode(value) -> assign -> AccessNode(name)` definition in `block`.
void build_constant_definition(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& name,
    const std::string& value,
    const types::Scalar& type
) {
    auto& constant = builder.add_constant(block, value, type);
    auto& out_node = builder.add_access(block, name);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, constant, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {});
}

// Builds `AccessNode(src) -> assign -> AccessNode(dst)`, i.e. a read of `src`.
void build_scalar_use(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Block& block,
    const std::string& src,
    const std::string& dst
) {
    auto& in_node = builder.add_access(block, src);
    auto& out_node = builder.add_access(block, dst);
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in_node, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", out_node, {});
}

// True if any tasklet in `block` reads a ConstantNode with the given literal.
bool reads_constant(structured_control_flow::Block& block, const std::string& value) {
    for (auto* tasklet : block.dataflow().tasklets()) {
        for (auto& edge : block.dataflow().in_edges(*tasklet)) {
            if (auto* constant = dynamic_cast<data_flow::ConstantNode*>(&edge.src())) {
                if (constant->data() == value) {
                    return true;
                }
            }
        }
    }
    return false;
}

// True if any tasklet in `block` still reads the (non-constant) container `name`.
bool reads_container(structured_control_flow::Block& block, const std::string& name) {
    for (auto* tasklet : block.dataflow().tasklets()) {
        for (auto& edge : block.dataflow().in_edges(*tasklet)) {
            auto* access = dynamic_cast<data_flow::AccessNode*>(&edge.src());
            if (access == nullptr || dynamic_cast<data_flow::ConstantNode*>(access) != nullptr) {
                continue;
            }
            if (access->data() == name) {
                return true;
            }
        }
    }
    return false;
}

symbolic::Symbol add_index(builder::StructuredSDFGBuilder& builder, const std::string& name) {
    builder.add_container(name, types::Scalar(types::PrimitiveType::UInt64));
    return symbolic::symbol(name);
}

} // namespace

TEST(ConstantPropagationTest, Sequence_StraightLine) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c", type);
    builder.add_container("out", type);

    auto& root = builder.subject().root();
    auto& def = builder.add_block(root);
    build_constant_definition(builder, def, "c", "1.0", type);
    auto& use = builder.add_block(root);
    build_scalar_use(builder, use, "c", "out");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_TRUE(reads_constant(use, "1.0"));
    EXPECT_FALSE(reads_container(use, "c"));
}

TEST(ConstantPropagationTest, IfElse_PropagatesIntoBranch) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c", type);
    builder.add_container("out", type);
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));

    auto& root = builder.subject().root();
    auto& def = builder.add_block(root);
    build_constant_definition(builder, def, "c", "2.0", type);

    auto& if_else = builder.add_if_else(root);
    auto& branch = builder.add_case(if_else, symbolic::Eq(symbolic::symbol("i"), symbolic::zero()));
    auto& use = builder.add_block(branch);
    build_scalar_use(builder, use, "c", "out");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_TRUE(reads_constant(use, "2.0"));
    EXPECT_FALSE(reads_container(use, "c"));
}

TEST(ConstantPropagationTest, For_PropagatesIntoBody) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c", type);
    builder.add_container("out", type);

    auto& root = builder.subject().root();
    auto& def = builder.add_block(root);
    build_constant_definition(builder, def, "c", "3.0", type);

    auto indvar = add_index(builder, "k");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one())
    );
    auto& use = builder.add_block(loop.root());
    build_scalar_use(builder, use, "c", "out");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_TRUE(reads_constant(use, "3.0"));
    EXPECT_FALSE(reads_container(use, "c"));
}

TEST(ConstantPropagationTest, While_PropagatesIntoBody) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c", type);
    builder.add_container("out", type);

    auto& root = builder.subject().root();
    auto& def = builder.add_block(root);
    build_constant_definition(builder, def, "c", "4.0", type);

    auto& loop = builder.add_while(root);
    auto& use = builder.add_block(loop.root());
    build_scalar_use(builder, use, "c", "out");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_TRUE(reads_constant(use, "4.0"));
    EXPECT_FALSE(reads_container(use, "c"));
}

TEST(ConstantPropagationTest, Map_PropagatesIntoBody) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c", type);
    builder.add_container("out", type);

    auto& root = builder.subject().root();
    auto& def = builder.add_block(root);
    build_constant_definition(builder, def, "c", "5.0", type);

    auto indvar = add_index(builder, "k");
    auto& loop = builder.add_map(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create()
    );
    auto& use = builder.add_block(loop.root());
    build_scalar_use(builder, use, "c", "out");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_TRUE(reads_constant(use, "5.0"));
    EXPECT_FALSE(reads_container(use, "c"));
}

TEST(ConstantPropagationTest, LoopBodyDefinitionKilledAfter) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c", type);
    builder.add_container("out", type);

    auto& root = builder.subject().root();

    // Definition sits INSIDE the loop body; it must not be assumed after the loop.
    auto indvar = add_index(builder, "k");
    auto& loop = builder.add_for(
        root,
        indvar,
        symbolic::Lt(indvar, symbolic::integer(10)),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one())
    );
    auto& def = builder.add_block(loop.root());
    build_constant_definition(builder, def, "c", "6.0", type);

    auto& use = builder.add_block(root);
    build_scalar_use(builder, use, "c", "out");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    // Propagation happens inside the body would need a use there; here there is none, so nothing applies.
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    EXPECT_FALSE(reads_constant(use, "6.0"));
    EXPECT_TRUE(reads_container(use, "c"));
}

TEST(ConstantPropagationTest, OverwriteKillsConstant) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c", type);
    builder.add_container("d", type, true); // argument feeding the non-constant overwrite
    builder.add_container("out", type);

    auto& root = builder.subject().root();
    auto& def = builder.add_block(root);
    build_constant_definition(builder, def, "c", "7.0", type);

    // Overwrite c with a non-constant value: d -> assign -> c
    auto& overwrite = builder.add_block(root);
    build_scalar_use(builder, overwrite, "d", "c");

    auto& use = builder.add_block(root);
    build_scalar_use(builder, use, "c", "out");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_FALSE(pass.run(builder, analysis_manager));

    EXPECT_FALSE(reads_constant(use, "7.0"));
    EXPECT_TRUE(reads_container(use, "c"));
}

TEST(ConstantPropagationTest, WeaklyConnectedComponents) {
    builder::StructuredSDFGBuilder builder("sdfg", FunctionType_CPU);
    types::Scalar type(types::PrimitiveType::Float);
    builder.add_container("c1", type);
    builder.add_container("c2", type);
    builder.add_container("o1", type);
    builder.add_container("o2", type);

    auto& root = builder.subject().root();

    // One block with two independent weakly-connected components, each a constant definition.
    auto& def = builder.add_block(root);
    build_constant_definition(builder, def, "c1", "1.0", type);
    build_constant_definition(builder, def, "c2", "2.0", type);
    EXPECT_EQ(def.dataflow().weakly_connected_components().first, 2);

    auto& use = builder.add_block(root);
    build_scalar_use(builder, use, "c1", "o1");
    build_scalar_use(builder, use, "c2", "o2");

    analysis::AnalysisManager analysis_manager(builder.subject());
    passes::ConstantPropagation pass;
    EXPECT_TRUE(pass.run(builder, analysis_manager));

    EXPECT_TRUE(reads_constant(use, "1.0"));
    EXPECT_TRUE(reads_constant(use, "2.0"));
    EXPECT_FALSE(reads_container(use, "c1"));
    EXPECT_FALSE(reads_container(use, "c2"));
}
