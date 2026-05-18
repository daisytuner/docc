#include <gtest/gtest.h>

#include "docc/lifting/lifting.h"
#include "lifting/test_utils.h"

#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/library_nodes/stdlib/stdlib.h>

using namespace docc;

TEST(LibFuncLiftingTest, VisitLibFunc_Free) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @free(ptr noundef)

define void @foo(ptr %a) {
entry:
  tail call void @free(ptr noundef %a)
  ret void
}
)";

    llvm::LLVMContext context;
    auto module = loadModuleFromIR(ir, context);
    ASSERT_NE(module, nullptr);

    // Construct the TargetLibraryInfo required by the lifting pass
    llvm::TargetLibraryInfoImpl TLIImpl(llvm::Triple(module->getTargetTriple()));
    llvm::TargetLibraryInfo TLI(TLIImpl);

    llvm::Function* function = module->getFunction("foo");
    ASSERT_NE(function, nullptr);

    // Run the lifting pass
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("a"));

    EXPECT_TRUE(sdfg->exists("free"));
    EXPECT_TRUE(sdfg->is_external("free"));

    bool found_free = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_free);

        const sdfg::stdlib::FreeNode* free_node = nullptr;
        for (auto& node : state.dataflow().nodes()) {
            if (auto lib_node = dynamic_cast<const sdfg::stdlib::FreeNode*>(&node)) {
                free_node = lib_node;
                found_free = true;
                break;
            }
        }
        EXPECT_NE(free_node, nullptr);

        EXPECT_EQ(state.dataflow().in_degree(*free_node), 1);
        EXPECT_EQ(state.dataflow().out_degree(*free_node), 1);

        auto& iedge = *state.dataflow().in_edges(*free_node).begin();
        EXPECT_EQ(iedge.dst_conn(), "_ptr");
        EXPECT_EQ(iedge.src_conn(), "void");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), sdfg::types::Pointer());

        auto& src = static_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src.data(), "a");

        auto& oedge = *state.dataflow().out_edges(*free_node).begin();
        EXPECT_EQ(oedge.src_conn(), "_ptr");
        EXPECT_EQ(oedge.dst_conn(), "void");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), sdfg::types::Pointer());

        auto& dst = static_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "a");
    }
    EXPECT_TRUE(found_free);
}

TEST(LibFuncLiftingTest, VisitLibFunc_Malloc) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @malloc(i64 noundef)

define void @foo(i64 noundef %a) {
entry:
  %0 = call ptr @malloc(i64 noundef %a)
  ret void
}
)";

    llvm::LLVMContext context;
    auto module = loadModuleFromIR(ir, context);
    ASSERT_NE(module, nullptr);

    // Construct the TargetLibraryInfo required by the lifting pass
    llvm::TargetLibraryInfoImpl TLIImpl(llvm::Triple(module->getTargetTriple()));
    llvm::TargetLibraryInfo TLI(TLIImpl);

    llvm::Function* function = module->getFunction("foo");
    ASSERT_NE(function, nullptr);

    // Run the lifting pass
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->containers().size(), 3);
    EXPECT_TRUE(sdfg->exists("a"));
    EXPECT_TRUE(sdfg->exists("_0"));

    EXPECT_TRUE(sdfg->exists("malloc"));
    EXPECT_TRUE(sdfg->is_external("malloc"));

    bool found_malloc = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_malloc);

        const sdfg::stdlib::MallocNode* target_node = nullptr;
        for (auto& node : state.dataflow().nodes()) {
            if (auto lib_node = dynamic_cast<const sdfg::stdlib::MallocNode*>(&node)) {
                target_node = lib_node;
                found_malloc = true;
                break;
            }
        }
        EXPECT_NE(target_node, nullptr);

        EXPECT_TRUE(sdfg::symbolic::eq(target_node->size(), sdfg::symbolic::symbol("a")));

        EXPECT_EQ(state.dataflow().in_degree(*target_node), 0);
        EXPECT_EQ(state.dataflow().out_degree(*target_node), 1);

        auto& oedge = *state.dataflow().out_edges(*target_node).begin();
        EXPECT_EQ(oedge.src_conn(), "_ret");
        EXPECT_EQ(oedge.dst_conn(), "void");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), sdfg::types::Pointer());

        auto& dst = static_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");
    }
    EXPECT_TRUE(found_malloc);
}

TEST(LibFuncLiftingTest, VisitLibFunc_Calloc) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @calloc(i64 noundef, i64 noundef)

define void @foo(i64 noundef %a) {
entry:
  %0 = call ptr @calloc(i64 noundef %a, i64 noundef 1)
  ret void
}
)";

    llvm::LLVMContext context;
    auto module = loadModuleFromIR(ir, context);
    ASSERT_NE(module, nullptr);

    // Construct the TargetLibraryInfo required by the lifting pass
    llvm::TargetLibraryInfoImpl TLIImpl(llvm::Triple(module->getTargetTriple()));
    llvm::TargetLibraryInfo TLI(TLIImpl);

    llvm::Function* function = module->getFunction("foo");
    ASSERT_NE(function, nullptr);

    // Run the lifting pass
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->containers().size(), 3);
    EXPECT_TRUE(sdfg->exists("a"));
    EXPECT_TRUE(sdfg->exists("_0"));

    EXPECT_TRUE(sdfg->exists("calloc"));
    EXPECT_TRUE(sdfg->is_external("calloc"));

    bool found_calloc = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_calloc);

        const sdfg::stdlib::CallocNode* target_node = nullptr;
        for (auto& node : state.dataflow().nodes()) {
            if (auto lib_node = dynamic_cast<const sdfg::stdlib::CallocNode*>(&node)) {
                target_node = lib_node;
                found_calloc = true;
                break;
            }
        }
        EXPECT_NE(target_node, nullptr);

        EXPECT_TRUE(sdfg::symbolic::eq(target_node->num(), sdfg::symbolic::symbol("a")));
        EXPECT_TRUE(sdfg::symbolic::eq(target_node->size(), sdfg::symbolic::integer(1)));

        EXPECT_EQ(state.dataflow().in_degree(*target_node), 0);
        EXPECT_EQ(state.dataflow().out_degree(*target_node), 1);

        auto& oedge = *state.dataflow().out_edges(*target_node).begin();
        EXPECT_EQ(oedge.src_conn(), "_ret");
        EXPECT_EQ(oedge.dst_conn(), "void");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), sdfg::types::Pointer());

        auto& dst = static_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");
    }
    EXPECT_TRUE(found_calloc);
}

TEST(LibFuncLiftingTest, VisitLibFunc_Sqrtf_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare float @sqrtf(float)

define void @foo(float %i) {
entry:
  %0 = tail call float @sqrtf(float %i)
  ret void
}
)";

    llvm::LLVMContext context;
    auto module = loadModuleFromIR(ir, context);
    ASSERT_NE(module, nullptr);

    // Construct the TargetLibraryInfo required by the lifting pass
    llvm::TargetLibraryInfoImpl TLIImpl(llvm::Triple(module->getTargetTriple()));
    llvm::TargetLibraryInfo TLI(TLIImpl);

    llvm::Function* function = module->getFunction("foo");
    ASSERT_NE(function, nullptr);

    // Run the lifting pass
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->containers().size(), 3);
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("i"));

    EXPECT_TRUE(sdfg->exists("sqrtf"));
    EXPECT_TRUE(sdfg->is_external("sqrtf"));

    bool found_sqrt = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_sqrt);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto lib_node = dynamic_cast<const sdfg::math::cmath::CMathNode*>(*data_flow.library_nodes().begin());
        EXPECT_EQ(lib_node->name(), "sqrtf");
        EXPECT_EQ(lib_node->inputs().size(), 1);
        EXPECT_EQ(lib_node->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto& iedge = *data_flow.in_edges(*lib_node).begin();

        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);
        EXPECT_EQ(iedge.dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "i");

        auto& oedge = *data_flow.out_edges(*lib_node).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), lib_node->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_sqrt = true;
    }
    EXPECT_TRUE(found_sqrt);
}
