#include "sdfg/passes/structured_control_flow/common_assignment_elimination.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg_debug_dump.h"

using namespace sdfg;

TEST(CommonAssignmentEliminationTest, RodiniaBfsAssignmentBlockReproducer) {
    builder::StructuredSDFGBuilder builder("sdfg_test", FunctionType_CPU);

    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    // Add containers
    types::Scalar sym_desc(types::PrimitiveType::UInt64);
    types::Scalar fp_desc(types::PrimitiveType::Float);
    types::Pointer ptr_type(fp_desc);
    builder.add_container("ptr", ptr_type, true);
    builder.add_container("N", sym_desc, true);
    builder.add_container("i", sym_desc);
    builder.add_container("x", sym_desc);
    builder.add_container("y", sym_desc);

    auto i = symbolic::symbol("i");
    auto x = symbolic::symbol("x");
    auto y = symbolic::symbol("y");
    auto N = symbolic::symbol("N");

    // Assignments that are identical in both branches
    control_flow::Assignments common_assignments = {
        {x, symbolic::add(i, symbolic::one())},
        {y, symbolic::add(i, symbolic::integer(2))},
    };

    // Define if-else with two branches
    auto& ifelse = builder.add_if_else(root);

    // If branch: condition N > 0
    auto& if_body = builder.add_case(ifelse, symbolic::Gt(N, symbolic::zero()));
    builder.add_assignments(if_body, common_assignments);

    // Else branch: condition N <= 0
    auto& else_body = builder.add_case(ifelse, symbolic::Le(N, symbolic::zero()));
    auto& side_effect_block = builder.add_block(else_body);
    auto& ptr_write = builder.add_access(side_effect_block, "ptr");
    auto& fp_const = builder.add_constant(side_effect_block, "1.3f", fp_desc);
    auto& tasklet = builder.add_tasklet(side_effect_block, data_flow::assign, "_out", {"_in"});
    builder.add_computational_memlet(side_effect_block, fp_const, tasklet, "_in", {}, fp_desc);
    builder.add_computational_memlet(side_effect_block, tasklet, "_out", ptr_write, {i}, ptr_type);

    builder.add_assignments(else_body, common_assignments);

    // use of assignments directly after in such a way as reusing this block will cause issues, as assignments in the
    // same block are not ordered and canont reuse other assignments results
    auto& existing_shared_assignment = builder.add_assignments(root, {{i, symbolic::mul(y, x)}});

    builder.add_return(root, "i");

    dump_sdfg(builder.subject(), "0.init");

    // Run pass
    auto sdfg_opt = builder.move();
    builder::StructuredSDFGBuilder builder_opt(sdfg_opt);
    analysis::AnalysisManager analysis_manager(builder_opt.subject());
    passes::CommonAssignmentElimination common_assignment_elimination;
    common_assignment_elimination.run(builder_opt, analysis_manager);

    dump_sdfg(builder_opt.subject(), "1.after");

    // must have hoisted the common assignments to after if_else but before existing. Disallowed to put it into the
    // existing block because it reuses them.
    ASSERT_EQ(root.size(), 4);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Return*>(&root.at(3)));
    EXPECT_EQ(&root.at(2), &existing_shared_assignment);
    auto* hoisted = dyn_cast<AssignmentBlock*>(&root.at(1));
    EXPECT_TRUE(hoisted);
    EXPECT_TRUE(hoisted->assignments().contains(x));
    EXPECT_TRUE(hoisted->assignments().contains(y));

    EXPECT_FALSE(common_assignment_elimination.run(builder_opt, analysis_manager));

    // Must not have hoisted the same code again, as that would lead to infinite looping in current code
    ASSERT_EQ(root.size(), 4);
    EXPECT_TRUE(dyn_cast<structured_control_flow::Return*>(&root.at(3)));
    EXPECT_EQ(&root.at(2), &existing_shared_assignment);
    auto* hoisted_still = dyn_cast<AssignmentBlock*>(&root.at(1));
    EXPECT_TRUE(hoisted_still);
    EXPECT_TRUE(hoisted_still->assignments().contains(x));
    EXPECT_TRUE(hoisted_still->assignments().contains(y));
}
