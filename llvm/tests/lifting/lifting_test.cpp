#include <gtest/gtest.h>

#include <llvm/Transforms/Utils/ModuleUtils.h>
#include "docc/lifting/function_to_sdfg.h"
#include "docc/lifting/lifting.h"
#include "lifting/test_utils.h"

#include <sdfg/data_flow/library_nodes/stdlib/stdlib.h>
#include <sdfg/structured_control_flow/return.h>

using namespace docc;

TEST(LiftingTest, VisitArguments) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a, float %b, i8* %c) {
entry:
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    // Check that all arguments were correctly lifted into the SDFG
    ASSERT_EQ(sdfg->arguments().size(), 3);
    EXPECT_EQ(sdfg->arguments()[0], "a");
    EXPECT_EQ(sdfg->arguments()[1], "b");
    EXPECT_EQ(sdfg->arguments()[2], "c");

    sdfg::types::Scalar type_i32(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Scalar type_f32(sdfg::types::PrimitiveType::Float);
    sdfg::types::Pointer ptr_type;

    EXPECT_EQ(sdfg->type("a"), type_i32);
    EXPECT_EQ(sdfg->type("b"), type_f32);
    EXPECT_EQ(sdfg->type("c"), ptr_type);
}

TEST(LiftingTest, VisitArguments_VarArg) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a, ...) {
entry:
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

    // Create the lifting object and run it
    EXPECT_TRUE(lifting::FunctionToSDFG::is_blacklisted(*function, false));
    EXPECT_THROW({ lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU); }, lifting::NotImplementedException);
}

TEST(LiftingTest, VisitGlobals_External) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = global i32 0
@h = global i32 0

define void @foo() {
entry:
  %0 = load i32, ptr @g
  %1 = load i32, ptr @h
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    ASSERT_EQ(sdfg->externals().size(), 2);

    auto& externals = sdfg->externals();
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "g") != externals.end());
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "h") != externals.end());

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Pointer ptr_type(base_type);
    EXPECT_EQ(sdfg->type("g"), ptr_type);
    EXPECT_EQ(sdfg->type("h"), ptr_type);
    EXPECT_EQ(sdfg->linkage_type("g"), sdfg::LinkageType_External);
    EXPECT_EQ(sdfg->linkage_type("h"), sdfg::LinkageType_External);

    // Check thinlto-preserve attribute and llvm.used
    auto g_global = function->getParent()->getGlobalVariable("g");
    auto h_global = function->getParent()->getGlobalVariable("h");

    auto used_global = function->getParent()->getGlobalVariable("llvm.used");
    EXPECT_EQ(used_global->getLinkage(), llvm::GlobalValue::AppendingLinkage);

    {
        llvm::SmallVector<llvm::GlobalValue*, 8> used_globals;
        llvm::collectUsedGlobalVariables(*function->getParent(), used_globals, false);
        EXPECT_TRUE(std::find(used_globals.begin(), used_globals.end(), g_global) != used_globals.end());
        EXPECT_TRUE(std::find(used_globals.begin(), used_globals.end(), h_global) != used_globals.end());
    }
}

TEST(LiftingTest, VisitGlobals_Internal_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = internal global i32 0
@h = internal global i32 1

define void @foo() {
entry:
  %0 = load i32, ptr @g
  %1 = load i32, ptr @h
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    ASSERT_EQ(sdfg->externals().size(), 2);

    auto& externals = sdfg->externals();
    for (const auto& ext : externals) {
        std::cout << "External: " << ext << std::endl;
    }
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "__daisy_int_in_memoryIR_g") != externals.end());
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "__daisy_int_in_memoryIR_h") != externals.end());

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Pointer ptr_type(base_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_g"), ptr_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_h"), ptr_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_g").initializer(), "");
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_h").initializer(), "");
    EXPECT_EQ(sdfg->linkage_type("__daisy_int_in_memoryIR_g"), sdfg::LinkageType_External);
    EXPECT_EQ(sdfg->linkage_type("__daisy_int_in_memoryIR_h"), sdfg::LinkageType_External);
}

TEST(LiftingTest, VisitGlobals_Internal_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = internal global <4 x i32> zeroinitializer
@h = internal global <4 x i32> <i32 -1, i32 0, i32 1, i32 2>

define void @foo() {
entry:
  %0 = load <4 x i32>, ptr @g
  %1 = load <4 x i32>, ptr @h
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    ASSERT_EQ(sdfg->externals().size(), 2);

    auto& externals = sdfg->externals();
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "__daisy_int_in_memoryIR_g") != externals.end());
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "__daisy_int_in_memoryIR_h") != externals.end());

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Structure struct_type("__daisy_vec_4_4");
    sdfg::types::Pointer ptr_type(struct_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_g"), ptr_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_h"), ptr_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_g").initializer(), "");
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_h").initializer(), "");
    EXPECT_EQ(sdfg->linkage_type("__daisy_int_in_memoryIR_g"), sdfg::LinkageType_External);
    EXPECT_EQ(sdfg->linkage_type("__daisy_int_in_memoryIR_h"), sdfg::LinkageType_External);
}

TEST(LiftingTest, VisitGlobals_Internal_Floats) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = internal global float 0.0
@h = internal global <4 x float> <float -1.0, float 0.0, float 1.0, float 2.0>

define void @foo() {
entry:
  %0 = load float, ptr @g
  %1 = load <4 x float>, ptr @h
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    ASSERT_EQ(sdfg->externals().size(), 2);

    auto& externals = sdfg->externals();
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "__daisy_int_in_memoryIR_g") != externals.end());
    EXPECT_TRUE(std::find(externals.begin(), externals.end(), "__daisy_int_in_memoryIR_h") != externals.end());

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
    sdfg::types::Structure struct_type("__daisy_vec_14_4");
    sdfg::types::Pointer ptr_structure_type(struct_type);
    sdfg::types::Pointer ptr_type(base_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_g"), ptr_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_h"), ptr_structure_type);
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_g").initializer(), "");
    EXPECT_EQ(sdfg->type("__daisy_int_in_memoryIR_h").initializer(), "");
    EXPECT_EQ(sdfg->linkage_type("__daisy_int_in_memoryIR_g"), sdfg::LinkageType_External);
    EXPECT_EQ(sdfg->linkage_type("__daisy_int_in_memoryIR_h"), sdfg::LinkageType_External);
}

TEST(LiftingTest, VisitCFG_Empty) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->states().size(), 3);
    EXPECT_EQ(sdfg->edges().size(), 2);
}

TEST(LiftingTest, VisitCFG_Branch) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  br label %exit
exit:
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->states().size(), 4);
    EXPECT_EQ(sdfg->edges().size(), 3);
}

TEST(LiftingTest, VisitCFG_ConditionalBranch) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i1 %cond) {
entry:
  br i1 %cond, label %bb_true, label %bb_false
bb_true:
  br label %exit
bb_false:
  br label %exit
exit:
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

    // Create the lifting object and run it
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->states().size(), 6);
    EXPECT_EQ(sdfg->edges().size(), 6);
}

TEST(LiftingTest, VisitCFG_Phi) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i1 %a) {
entry:
  br label %bb1
bb1:
  %0 = phi i1 [ %a, %entry ], [ %0, %bb1 ]
  br i1 %0, label %exit, label %bb1
exit:
  ret void
}
)";

    llvm::LLVMContext context;
    auto module = loadModuleFromIR(ir, context);
    ASSERT_NE(module, nullptr);

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
    EXPECT_TRUE(sdfg->exists("_0_bb1_bb1"));
}


TEST(LiftingTest, VisitCFG_Phi_ConstantInt) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i1 %a) {
entry:
  br label %bb1
bb1:
  %0 = phi i1 [ 0, %entry ], [ %a, %bb1 ]
  br i1 %0, label %exit, label %bb1
exit:
  ret void
}
)";

    llvm::LLVMContext context;
    auto module = loadModuleFromIR(ir, context);
    ASSERT_NE(module, nullptr);

    llvm::TargetLibraryInfoImpl TLIImpl(llvm::Triple(module->getTargetTriple()));
    llvm::TargetLibraryInfo TLI(TLIImpl);

    llvm::Function* function = module->getFunction("foo");
    ASSERT_NE(function, nullptr);

    // Run the lifting pass
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("a"));
    EXPECT_TRUE(sdfg->exists("_0"));
}

TEST(LiftingTest, VisitCFG_Phi_ConstantPointerNull) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %a, i1 %b) {
entry:
  br label %bb1
bb1:
  %0 = phi ptr [ null, %entry ], [ %a, %bb1 ]
  br i1 %b, label %exit, label %bb1
exit:
  ret void
}
)";

    llvm::LLVMContext context;
    auto module = loadModuleFromIR(ir, context);
    ASSERT_NE(module, nullptr);

    llvm::TargetLibraryInfoImpl TLIImpl(llvm::Triple(module->getTargetTriple()));
    llvm::TargetLibraryInfo TLI(TLIImpl);

    llvm::Function* function = module->getFunction("foo");
    ASSERT_NE(function, nullptr);

    // Run the lifting pass
    lifting::Lifting lifting(TLI, *function, sdfg::FunctionType_CPU);
    auto sdfg = lifting.run();

    EXPECT_EQ(sdfg->containers().size(), 3);
    EXPECT_TRUE(sdfg->exists("a"));
    EXPECT_TRUE(sdfg->exists("b"));
    EXPECT_TRUE(sdfg->exists("_0"));
}

TEST(LiftingTest, VisitStoreInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p, i32 %a) {
entry:
  store i32 %a, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("a"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Pointer ptr_type(base_type);

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(oedge.base_type(), ptr_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "p");


        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Scalar_ConstantInt) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store i32 0, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Pointer ptr_type(base_type);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(oedge.base_type(), ptr_type);

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);
        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge.src());
        EXPECT_EQ(src.data(), "0");
        EXPECT_EQ(src.type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "p");


        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Scalar_ConstantBool_True) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store i1 true, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Bool);
        sdfg::types::Pointer ptr_type(base_type);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(oedge.base_type(), ptr_type);

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);
        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge.src());
        EXPECT_EQ(src.data(), "1");
        EXPECT_EQ(src.type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "p");


        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Scalar_ConstantBool_False) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store i1 false, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Bool);
        sdfg::types::Pointer ptr_type(base_type);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(oedge.base_type(), ptr_type);

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);
        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge.src());
        EXPECT_EQ(src.data(), "0");
        EXPECT_EQ(src.type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "p");


        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Scalar_ConstantFP_32) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store float 0.0, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Pointer ptr_type(base_type);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(oedge.base_type(), ptr_type);

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);
        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge.src());
        EXPECT_EQ(src.data(), "0.0f");
        EXPECT_EQ(src.type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "p");


        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Scalar_ConstantFP_64) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store double 0.0, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Double);
        sdfg::types::Pointer ptr_type(base_type);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(oedge.base_type(), ptr_type);

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);
        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge.src());
        EXPECT_EQ(src.data(), "0.0");
        EXPECT_EQ(src.type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "p");


        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p, <4 x i32> %a) {
entry:
  store <4 x i32> %a, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("a"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure struct_type("__daisy_vec_4_4");
        sdfg::types::Pointer ptr_type(struct_type);

        auto& ref_edge = *data_flow.edges().begin();
        EXPECT_EQ(ref_edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(ref_edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(ref_edge.base_type(), ptr_type);
        EXPECT_EQ(ref_edge.type(), sdfg::data_flow::MemletType::Dereference_Dst);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.dst());
        EXPECT_EQ(dst.data(), "p");

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.src());
        EXPECT_EQ(src.data(), "a");

        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Vector_ConstantAggregateZero) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store <4 x i32> zeroinitializer, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure struct_type("__daisy_vec_4_4");
        sdfg::types::Pointer ptr_type(struct_type);

        auto& ref_edge = *data_flow.edges().begin();
        EXPECT_EQ(ref_edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(ref_edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(ref_edge.base_type(), ptr_type);
        EXPECT_EQ(ref_edge.type(), sdfg::data_flow::MemletType::Dereference_Dst);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.dst());
        EXPECT_EQ(dst.data(), "p");

        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(ref_edge.src());
        EXPECT_EQ(src.data(), "{0, 0, 0, 0}");
        EXPECT_EQ(src.type(), struct_type);

        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Vector_ConstantDataSequential) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    size_t store_state_count = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure struct_type("__daisy_vec_4_4");
        sdfg::types::Pointer ptr_type(struct_type);

        auto& ref_edge = *data_flow.edges().begin();
        EXPECT_EQ(ref_edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(ref_edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(ref_edge.base_type(), ptr_type);
        EXPECT_EQ(ref_edge.type(), sdfg::data_flow::MemletType::Dereference_Dst);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.dst());
        EXPECT_EQ(dst.data(), "p");

        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(ref_edge.src());
        EXPECT_EQ(src.data(), "{0, 1, 2, 3}");
        EXPECT_EQ(src.type(), struct_type);

        store_state_count++;
    }
    EXPECT_EQ(store_state_count, 1);
}

TEST(LiftingTest, VisitStoreInst_Pointer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p, ptr %q) {
entry:
  store ptr %q, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("q"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& edge = *data_flow.edges().begin();
        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.src());
        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.dst());

        EXPECT_EQ(src.data(), "q");
        EXPECT_EQ(dst.data(), "p");
        EXPECT_EQ(edge.type(), sdfg::data_flow::MemletType::Dereference_Dst);

        sdfg::types::Pointer base_type;
        sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_type));
        EXPECT_EQ(edge.base_type(), ptr_type);

        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Pointer_ConstantPointerNull) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store ptr null, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& edge = *data_flow.edges().begin();
        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.src());
        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.dst());

        EXPECT_EQ(src.data(), sdfg::symbolic::__nullptr__()->get_name());
        EXPECT_EQ(dst.data(), "p");
        EXPECT_EQ(edge.type(), sdfg::data_flow::MemletType::Dereference_Dst);

        sdfg::types::Pointer base_type;
        sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_type));
        EXPECT_EQ(edge.base_type(), ptr_type);

        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitStoreInst_Structure) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  store {i32, i32} {i32 0, i32 1} , ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("p"));

    bool found_store = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_store);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        sdfg::types::Structure struct_type("__daisy_anonymous_struct0");
        sdfg::types::Pointer ptr_type(struct_type);

        auto& ref_edge = *data_flow.edges().begin();
        EXPECT_EQ(ref_edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(ref_edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(ref_edge.base_type(), ptr_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.dst());
        EXPECT_EQ(dst.data(), "p");

        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(ref_edge.src());
        EXPECT_EQ(src.data(), "{0, 1}");
        EXPECT_EQ(src.type(), struct_type);

        found_store = true;
    }
    EXPECT_TRUE(found_store);
}

TEST(LiftingTest, VisitLoadInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = load i32, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));

    bool found_load = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_load);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge.subset().at(0), sdfg::symbolic::zero()));

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Pointer ptr_type(base_type);
        EXPECT_EQ(iedge.base_type(), ptr_type);

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src.data(), "p");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_load = true;
    }
}

TEST(LiftingTest, VisitLoadInst_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = load <4 x i32>, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));

    bool found_load = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_load);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure struct_type("__daisy_vec_4_4");
        sdfg::types::Pointer ptr_type(struct_type);

        auto& ref_edge = *data_flow.edges().begin();
        EXPECT_EQ(ref_edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(ref_edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(ref_edge.type(), sdfg::data_flow::MemletType::Dereference_Src);
        EXPECT_EQ(ref_edge.base_type(), ptr_type);

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.src());
        EXPECT_EQ(src.data(), "p");

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_load = true;
    }
}

TEST(LiftingTest, VisitLoadInst_Pointer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = load ptr, ptr %p
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));

    bool found_load = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_load);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& edge = *data_flow.edges().begin();
        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.src());
        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.dst());

        EXPECT_EQ(src.data(), "p");
        EXPECT_EQ(dst.data(), "_0");
        EXPECT_EQ(edge.type(), sdfg::data_flow::MemletType::Dereference_Src);

        sdfg::types::Pointer base_type;
        sdfg::types::Pointer ptr_type(static_cast<const sdfg::types::IType&>(base_type));
        EXPECT_EQ(edge.base_type(), ptr_type);

        found_load = true;
    }
}

TEST(LiftingTest, VisitLoadInst_Structure) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = load { i32, i32 }, ptr %p
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
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));

    size_t found_loads = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        sdfg::types::Structure base_type("__daisy_anonymous_struct0");
        sdfg::types::Pointer ptr_type(base_type);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& ref_edge = *data_flow.edges().begin();
        EXPECT_EQ(ref_edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(ref_edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(ref_edge.type(), sdfg::data_flow::MemletType::Dereference_Src);
        EXPECT_EQ(ref_edge.base_type(), ptr_type);

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.src());
        EXPECT_EQ(src.data(), "p");

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_loads++;
    }
    EXPECT_EQ(found_loads, 1);
}

TEST(LiftingTest, VisitLoadInst_Structure_Pointers) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = load { ptr, ptr }, ptr %p
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
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));

    size_t found_loads = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        sdfg::types::Structure base_type("__daisy_anonymous_struct0");
        sdfg::types::Pointer ptr_type(base_type);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& ref_edge = *data_flow.edges().begin();
        EXPECT_EQ(ref_edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(ref_edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_EQ(ref_edge.type(), sdfg::data_flow::MemletType::Dereference_Src);
        EXPECT_EQ(ref_edge.base_type(), ptr_type);

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.src());
        EXPECT_EQ(src.data(), "p");

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(ref_edge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_loads++;
    }
    EXPECT_EQ(found_loads, 1);
}

TEST(LiftingTest, VisitGetElementPtrInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = getelementptr i32, ptr %p, i64 0
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));

    bool found_load = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_load);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& edge = *data_flow.edges().begin();
        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.src());
        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.dst());

        EXPECT_EQ(src.data(), "p");
        EXPECT_EQ(dst.data(), "_0");
        EXPECT_EQ(edge.type(), sdfg::data_flow::MemletType::Reference);

        EXPECT_EQ(edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(edge.subset().at(0), sdfg::symbolic::zero()));

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Pointer ptr_type(base_type);
        EXPECT_EQ(edge.base_type(), ptr_type);

        found_load = true;
    }
}

TEST(LiftingTest, VisitGetElementPtrInst_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = getelementptr <4 x i32>, ptr %p, i64 0, i64 1
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 2);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));

    bool found_load = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_load);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& edge = *data_flow.edges().begin();
        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.src());
        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.dst());

        EXPECT_EQ(src.data(), "p");
        EXPECT_EQ(dst.data(), "_0");
        EXPECT_EQ(edge.type(), sdfg::data_flow::MemletType::Reference);

        EXPECT_EQ(edge.subset().size(), 2);
        EXPECT_TRUE(sdfg::symbolic::eq(edge.subset().at(0), sdfg::symbolic::zero()));
        EXPECT_TRUE(sdfg::symbolic::eq(edge.subset().at(1), sdfg::symbolic::one()));

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure structure_type("__daisy_vec_4_4");
        sdfg::types::Pointer ptr_type(structure_type);
        EXPECT_EQ(edge.base_type(), ptr_type);

        found_load = true;
    }
}

TEST(LiftingTest, VisitGetElementPtrInst_Symbolic) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p, i64 %i, i64 %j) {
entry:
  %0 = getelementptr <4 x i32>, ptr %p, i64 %i, i64 %j
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

    // The SDFG should contain two containers: the pointer argument and the loaded value
    EXPECT_EQ(sdfg->containers().size(), 4);
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_TRUE(sdfg->exists("j"));

    bool found_load = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_load);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& edge = *data_flow.edges().begin();
        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.src());
        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.dst());

        EXPECT_EQ(src.data(), "p");
        EXPECT_EQ(dst.data(), "_0");
        EXPECT_EQ(edge.type(), sdfg::data_flow::MemletType::Reference);

        EXPECT_EQ(edge.subset().size(), 2);
        EXPECT_TRUE(sdfg::symbolic::eq(edge.subset().at(0), sdfg::symbolic::symbol("i")));
        EXPECT_TRUE(sdfg::symbolic::eq(edge.subset().at(1), sdfg::symbolic::symbol("j")));

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure structure_type("__daisy_vec_4_4");
        sdfg::types::Pointer ptr_type(structure_type);
        EXPECT_EQ(edge.base_type(), ptr_type);

        found_load = true;
    }
}

TEST(LiftingTest, VisitICmpInst_Eq_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = icmp eq i32 %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    sdfg::types::Scalar bool_type(sdfg::types::PrimitiveType::Bool);
    sdfg::types::Scalar int_type(sdfg::types::PrimitiveType::Int32);
    EXPECT_EQ(sdfg->type("_0"), bool_type);
    EXPECT_EQ(sdfg->type("i"), int_type);
    EXPECT_EQ(sdfg->type("j"), int_type);

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_i = sdfg::symbolic::symbol("i");
    auto sym_j = sdfg::symbolic::symbol("j");
    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Eq(sym_i, sym_j)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Eq_Scalar_Constant) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i) {
entry:
  %0 = icmp eq i32 %i, 1
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("i"));

    sdfg::types::Scalar bool_type(sdfg::types::PrimitiveType::Bool);
    sdfg::types::Scalar int_type(sdfg::types::PrimitiveType::Int32);
    EXPECT_EQ(sdfg->type("_0"), bool_type);
    EXPECT_EQ(sdfg->type("i"), int_type);

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_i = sdfg::symbolic::symbol("i");
    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Eq(sym_i, sdfg::symbolic::one())));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Ne_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = icmp ne i32 %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_i = sdfg::symbolic::symbol("i");
    auto sym_j = sdfg::symbolic::symbol("j");
    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Ne(sym_i, sym_j)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Sgt_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = icmp sgt i32 %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_i = sdfg::symbolic::symbol("i");
    auto sym_j = sdfg::symbolic::symbol("j");
    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Gt(sym_i, sym_j)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Sge_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = icmp sge i32 %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_i = sdfg::symbolic::symbol("i");
    auto sym_j = sdfg::symbolic::symbol("j");
    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Ge(sym_i, sym_j)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Slt_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = icmp slt i32 %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_i = sdfg::symbolic::symbol("i");
    auto sym_j = sdfg::symbolic::symbol("j");
    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Lt(sym_i, sym_j)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Sle_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = icmp sle i32 %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_i = sdfg::symbolic::symbol("i");
    auto sym_j = sdfg::symbolic::symbol("j");
    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Le(sym_i, sym_j)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Eq_Pointer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p, ptr %q) {
entry:
  %0 = icmp eq ptr %p, %q
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
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("q"));

    sdfg::types::Scalar bool_type(sdfg::types::PrimitiveType::Bool);
    sdfg::types::Pointer ptr_type;
    EXPECT_EQ(sdfg->type("_0"), bool_type);
    EXPECT_EQ(sdfg->type("p"), ptr_type);
    EXPECT_EQ(sdfg->type("q"), ptr_type);

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_p = sdfg::symbolic::symbol("p");
    auto sym_q = sdfg::symbolic::symbol("q");

    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Eq(sym_p, sym_q)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Eq_Pointer_Constant) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p) {
entry:
  %0 = icmp eq ptr %p, null
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("p"));

    sdfg::types::Scalar bool_type(sdfg::types::PrimitiveType::Bool);
    sdfg::types::Pointer ptr_type;
    EXPECT_EQ(sdfg->type("_0"), bool_type);
    EXPECT_EQ(sdfg->type("p"), ptr_type);

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_p = sdfg::symbolic::symbol("p");

    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Eq(sym_p, sdfg::symbolic::__nullptr__())));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}

TEST(LiftingTest, VisitICmpInst_Ule_Pointer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %p, ptr %q) {
entry:
  %0 = icmp ule ptr %p, %q
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
    EXPECT_TRUE(sdfg->exists("p"));
    EXPECT_TRUE(sdfg->exists("q"));

    sdfg::types::Scalar bool_type(sdfg::types::PrimitiveType::Bool);
    sdfg::types::Pointer ptr_type;
    EXPECT_EQ(sdfg->type("_0"), bool_type);
    EXPECT_EQ(sdfg->type("p"), ptr_type);
    EXPECT_EQ(sdfg->type("q"), ptr_type);

    auto sym_0 = sdfg::symbolic::symbol("_0");
    auto sym_p = sdfg::symbolic::symbol("p");
    auto sym_q = sdfg::symbolic::symbol("q");

    bool found_cmp = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        if (edge.assignments().find(sym_0) == edge.assignments().end()) {
            continue;
        }
        EXPECT_FALSE(found_cmp);

        auto rhs = edge.assignments().at(sym_0);
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::Le(sym_p, sym_q)));
        found_cmp = true;
    }
    EXPECT_TRUE(found_cmp);
}


TEST(LiftingTest, VisitFCmpInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float %i, float %j) {
entry:
  %0 = fcmp ogt float %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    bool found_fcmp = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_fcmp);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_ogt);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");


        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        sdfg::types::Scalar output_base_type(sdfg::types::PrimitiveType::Bool);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), output_base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_fcmp = true;
    }
    EXPECT_TRUE(found_fcmp);
}

TEST(LiftingTest, VisitFCmpInst_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x float> %i, <4 x float> %j) {
entry:
  %0 = fcmp ogt <4 x float> %i, %j
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

    EXPECT_EQ(sdfg->containers().size(), 4);
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("_i0"));
    EXPECT_TRUE(sdfg->exists("i"));
    EXPECT_TRUE(sdfg->exists("j"));

    size_t found_fcmp = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_ogt);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Structure structure_type("__daisy_vec_14_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge1->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge1->base_type(), structure_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");


        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge2->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge2->base_type(), structure_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        sdfg::types::Scalar output_base_type(sdfg::types::PrimitiveType::Bool);
        sdfg::types::Structure output_structure_type("__daisy_vec_1_4");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), output_structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_fcmp++;
    }
    EXPECT_EQ(found_fcmp, 1);
}

TEST(LiftingTest, VisitFCmpInst_Vector_ConstantZeroInitializer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x float> %i) {
entry:
  %0 = fcmp ogt <4 x float> %i, zeroinitializer
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
    EXPECT_TRUE(sdfg->exists("_i0"));
    EXPECT_TRUE(sdfg->exists("i"));

    size_t found_fcmp = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_ogt);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Structure structure_type("__daisy_vec_14_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge1->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge1->base_type(), structure_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");


        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge2->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge2->base_type(), structure_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "{0.0f, 0.0f, 0.0f, 0.0f}");

        sdfg::types::Structure output_structure_type("__daisy_vec_1_4");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), output_structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_fcmp++;
    }
    EXPECT_EQ(found_fcmp, 1);
}

TEST(LiftingTest, VisitFCmpInst_Vector_ConstantSequentialData) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x float> %i) {
entry:
  %0 = fcmp ogt <4 x float> %i, <float 1.0, float 2.0, float 3.0, float 4.0>
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
    EXPECT_TRUE(sdfg->exists("_i0"));
    EXPECT_TRUE(sdfg->exists("i"));

    size_t found_fcmp = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_ogt);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Structure structure_type("__daisy_vec_14_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge1->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge1->base_type(), structure_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");


        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge2->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge2->base_type(), structure_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "{1.0f, 2.0f, 3.0f, 4.0f}");

        sdfg::types::Scalar output_base_type(sdfg::types::PrimitiveType::Bool);
        sdfg::types::Structure output_structure_type("__daisy_vec_1_4");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), output_structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_fcmp++;
    }
    EXPECT_EQ(found_fcmp, 1);
}

TEST(LiftingTest, VisitUnaryOperator_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float %a) {
entry:
  %0 = fneg float %a
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    bool found_neg = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_neg);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_neg);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto iedge = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge->dst_conn(), "_in1");
        EXPECT_EQ(iedge->subset().size(), 0);
        EXPECT_EQ(iedge->base_type(), base_type);

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge->src());
        EXPECT_EQ(src.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_neg = true;
    }
    EXPECT_TRUE(found_neg);
}

TEST(LiftingTest, VisitUnaryOperator_Scalar_ConstantFP) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  %0 = fneg float 1.0
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

    EXPECT_EQ(sdfg->containers().size(), 1);
    EXPECT_TRUE(sdfg->exists("_0"));

    bool found_neg = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_neg);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_neg);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto iedge = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge->dst_conn(), "_in1");
        EXPECT_EQ(iedge->subset().size(), 0);
        EXPECT_EQ(iedge->base_type(), base_type);

        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge->src());
        EXPECT_EQ(src.data(), "1.0f");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_neg = true;
    }
    EXPECT_TRUE(found_neg);
}

TEST(LiftingTest, VisitUnaryOperator_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x float> %a) {
entry:
  %0 = fneg <4 x float> %a
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
    EXPECT_TRUE(sdfg->exists("_i0"));
    EXPECT_TRUE(sdfg->exists("a"));

    size_t found_neg = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_neg);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Structure structure_type("__daisy_vec_14_4");

        auto iedge = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge->dst_conn(), "_in1");
        EXPECT_EQ(iedge->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge->base_type(), structure_type);

        auto& src = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge->src());
        EXPECT_EQ(src.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_neg++;
    }
    EXPECT_EQ(found_neg, 1);
}

TEST(LiftingTest, VisitUnaryOperator_Vector_ConstantDataSequential) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  %0 = fneg <4 x float> <float -1.0, float 0.0, float 1.0, float 2.0>
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("_i0"));

    size_t found_neg = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_neg);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Structure structure_type("__daisy_vec_14_4");

        auto iedge = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge->dst_conn(), "_in1");
        EXPECT_EQ(iedge->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge->base_type(), structure_type);

        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge->src());
        EXPECT_EQ(src.data(), "{-1.0f, 0.0f, 1.0f, 2.0f}");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_neg++;
    }
    EXPECT_EQ(found_neg, 1);
}

TEST(LiftingTest, VisitUnaryOperator_Vector_ConstantAggregateZero) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  %0 = fneg <4 x float> zeroinitializer
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("_i0"));

    size_t found_neg = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_neg);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Structure structure_type("__daisy_vec_14_4");

        auto iedge = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge->dst_conn(), "_in1");
        EXPECT_EQ(iedge->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge->base_type(), structure_type);

        auto& src = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge->src());
        EXPECT_EQ(src.data(), "{0.0f, 0.0f, 0.0f, 0.0f}");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_neg++;
    }
    EXPECT_EQ(found_neg, 1);
}

TEST(LiftingTest, VisitBinaryOperator_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = add i32 %i, %j
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
    EXPECT_TRUE(sdfg->exists("j"));

    bool found_add = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_add);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_add);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");


        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(LiftingTest, VisitBinaryOperator_Scalar_SameInput) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i) {
entry:
  %0 = add i32 %i, %i
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("i"));

    bool found_add = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_add);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_add);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");


        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "i");
        EXPECT_EQ(&src1, &src2);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(LiftingTest, VisitBinaryOperator_Scalar_Constant) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i) {
entry:
  %0 = add i32 %i, 1
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("i"));

    bool found_add = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_add);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_add);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "1");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");


        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(LiftingTest, VisitBinaryOperator_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x i32> %a, <4 x i32> %b) {
entry:
  %0 = add <4 x i32> %a, %b
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

    EXPECT_EQ(sdfg->containers().size(), 4);
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("_i0"));
    EXPECT_TRUE(sdfg->exists("a"));
    EXPECT_TRUE(sdfg->exists("b"));

    size_t found_add_count = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_add);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure structure_type("__daisy_vec_4_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge1->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge1->base_type(), structure_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "a");

        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge2->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge2->base_type(), structure_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "b");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add_count++;
    }
    EXPECT_EQ(found_add_count, 1);
}

TEST(LiftingTest, VisitBinaryOperator_Vector_SameInput) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x i32> %a) {
entry:
  %0 = add <4 x i32> %a, %a
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
    EXPECT_TRUE(sdfg->exists("_i0"));
    EXPECT_TRUE(sdfg->exists("a"));

    size_t found_add_count = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_add);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure structure_type("__daisy_vec_4_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge1->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge1->base_type(), structure_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "a");

        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge2->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge2->base_type(), structure_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "a");
        EXPECT_EQ(&src1, &src2);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add_count++;
    }
    EXPECT_EQ(found_add_count, 1);
}

TEST(LiftingTest, VisitBinaryOperator_Vector_Constant) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x i32> %a) {
entry:
  %0 = add <4 x i32> %a, <i32 1, i32 2, i32 3, i32 4>
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
    EXPECT_TRUE(sdfg->exists("_i0"));
    EXPECT_TRUE(sdfg->exists("a"));

    size_t found_add_count = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_add);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
        sdfg::types::Structure structure_type("__daisy_vec_4_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge1->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge1->base_type(), structure_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "a");

        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(iedge2->subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(iedge2->base_type(), structure_type);

        auto& src2 = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "{1, 2, 3, 4}");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(oedge.subset().at(0), sdfg::symbolic::symbol("_i0")));
        EXPECT_EQ(oedge.base_type(), structure_type);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add_count++;
    }
    EXPECT_EQ(found_add_count, 1);
}

TEST(LiftingTest, VisitCastInst_FPExtInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float %a) {
entry:
  %0 = fpext float %a to double
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
    sdfg::types::Scalar base_type_double(sdfg::types::PrimitiveType::Double);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_double);

    bool found_fpext = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_fpext);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_double);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_fpext = true;
    }
    EXPECT_TRUE(found_fpext);
}

TEST(LiftingTest, VisitCastInst_FPToSIInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(double %a) {
entry:
  %0 = fptosi double %a to i32
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Double);
    sdfg::types::Scalar base_type_int32(sdfg::types::PrimitiveType::Int32);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_int32);

    bool found_fptosi = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_fptosi);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_int32);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_fptosi = true;
    }
    EXPECT_TRUE(found_fptosi);
}

TEST(LiftingTest, VisitCastInst_FPToUIInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(double %a) {
entry:
  %0 = fptoui double %a to i32
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Double);
    sdfg::types::Scalar base_type_int32(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Scalar base_type_uint32(sdfg::types::PrimitiveType::UInt32);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_int32);

    bool found_fptoui = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_fptoui);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_uint32);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_fptoui = true;
    }
    EXPECT_TRUE(found_fptoui);
}

TEST(LiftingTest, VisitCastInst_FPTruncInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(double %a) {
entry:
  %0 = fptrunc double %a to float
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Double);
    sdfg::types::Scalar base_type_float(sdfg::types::PrimitiveType::Float);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_float);

    bool found_fptrunc = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_fptrunc);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_float);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_fptrunc = true;
    }
    EXPECT_TRUE(found_fptrunc);
}

TEST(LiftingTest, VisitCastInst_SExtInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a) {
entry:
  %0 = sext i32 %a to i64
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Scalar base_type_int64(sdfg::types::PrimitiveType::Int64);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_int64);

    bool found_sext = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_sext);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_int64);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_sext = true;
    }
    EXPECT_TRUE(found_sext);
}

TEST(LiftingTest, VisitCastInst_SExtInst_i1) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i1 %a) {
entry:
  %0 = sext i1 %a to i32
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Bool);
    sdfg::types::Scalar base_type_int32(sdfg::types::PrimitiveType::Int32);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_int32);

    bool found_sext = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_sext);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_mul);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        auto iedge = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge->dst_conn() == "_in2") {
            std::swap(iedge, iedge2);
        }
        EXPECT_EQ(iedge->dst_conn(), "_in1");
        EXPECT_EQ(iedge->subset().size(), 0);
        EXPECT_EQ(iedge->base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge->src());
        EXPECT_EQ(src1.data(), "a");

        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type_int32);

        auto& src2 = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "-1");
        EXPECT_EQ(src2.type(), base_type_int32);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_int32);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_sext = true;
    }
    EXPECT_TRUE(found_sext);
}

TEST(LiftingTest, VisitCastInst_SIToFPInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i64 %a) {
entry:
  %0 = sitofp i64 %a to float
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int64);
    sdfg::types::Scalar base_type_float(sdfg::types::PrimitiveType::Float);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_float);

    bool found_sitofp = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_sitofp);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_float);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_sitofp = true;
    }
    EXPECT_TRUE(found_sitofp);
}

TEST(LiftingTest, VisitCastInst_TruncInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a) {
entry:
  %0 = trunc i32 %a to i16
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Scalar base_type_int16(sdfg::types::PrimitiveType::Int16);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_int16);

    sdfg::types::Scalar ubase_type(sdfg::types::PrimitiveType::UInt32);
    sdfg::types::Scalar ubase_type_int16(sdfg::types::PrimitiveType::UInt16);

    bool found_trunc = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_trunc);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), ubase_type);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), ubase_type_int16);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_trunc = true;
    }
    EXPECT_TRUE(found_trunc);
}

TEST(LiftingTest, VisitCastInst_UIToFPInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a) {
entry:
  %0 = uitofp i32 %a to float
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Scalar base_type_uint32(sdfg::types::PrimitiveType::UInt32);
    sdfg::types::Scalar base_type_float(sdfg::types::PrimitiveType::Float);
    EXPECT_EQ(sdfg->type("a"), base_type);
    EXPECT_EQ(sdfg->type("_0"), base_type_float);

    bool found_uitofp = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_uitofp);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type_uint32);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_float);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_uitofp = true;
    }
    EXPECT_TRUE(found_uitofp);
}

TEST(LiftingTest, VisitCastInst_ZExtInst_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a) {
entry:
  %0 = zext i32 %a to i64
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type_int32(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Scalar base_type_int64(sdfg::types::PrimitiveType::Int64);
    sdfg::types::Scalar base_type_uint32(sdfg::types::PrimitiveType::UInt32);
    sdfg::types::Scalar base_type_uint64(sdfg::types::PrimitiveType::UInt64);
    EXPECT_EQ(sdfg->type("a"), base_type_int32);
    EXPECT_EQ(sdfg->type("_0"), base_type_int64);

    bool found_zext = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_zext);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type_uint32);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_uint64);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_zext = true;
    }
    EXPECT_TRUE(found_zext);
}

TEST(LiftingTest, VisitCastInst_ZExtInst_NNeg) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %a) {
entry:
  %0 = zext nneg i32 %a to i64
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Scalar base_type_int32(sdfg::types::PrimitiveType::Int32);
    sdfg::types::Scalar base_type_int64(sdfg::types::PrimitiveType::Int64);
    EXPECT_EQ(sdfg->type("a"), base_type_int32);
    EXPECT_EQ(sdfg->type("_0"), base_type_int64);

    bool found_sext = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_sext);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type(), base_type_int32);

        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge.src());
        EXPECT_EQ(src1.data(), "a");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type_int64);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_sext = true;
    }
    EXPECT_TRUE(found_sext);
}

TEST(LiftingTest, VisitCastInst_SelectInst_Scalar_Condition_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i1 %a, i32 %b, i32 %c) {
entry:
  %0 = select i1 %a, i32 %b, i32 %c
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

    EXPECT_EQ(sdfg->containers().size(), 4);
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));
    EXPECT_TRUE(sdfg->exists("b"));
    EXPECT_TRUE(sdfg->exists("c"));

    size_t found_select = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in");

        auto& iedge = *data_flow.in_edges(*tasklet).begin();
        EXPECT_EQ(iedge.dst_conn(), "_in");
        EXPECT_EQ(iedge.subset().size(), 0);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.src_conn(), tasklet->output());
        EXPECT_EQ(oedge.subset().size(), 0);

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_select++;
    }
    EXPECT_EQ(found_select, 2);

    bool found_true = false;
    bool found_false = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.is_unconditional()) {
            continue;
        }
        if (sdfg::symbolic::
                eq(edge.condition(), sdfg::symbolic::Ne(sdfg::symbolic::symbol("a"), sdfg::symbolic::__false__()))) {
            EXPECT_FALSE(found_true);
            found_true = true;
        } else if (sdfg::symbolic::
                       eq(edge.condition(),
                          sdfg::symbolic::Eq(sdfg::symbolic::symbol("a"), sdfg::symbolic::__false__()))) {
            EXPECT_FALSE(found_false);
            found_false = true;
        }
    }
    EXPECT_TRUE(found_true);
    EXPECT_TRUE(found_false);
}

TEST(LiftingTest, VisitCastInst_SelectInst_Scalar_Condition_Pointer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i1 %a, ptr %b, ptr %c) {
entry:
  %0 = select i1 %a, ptr %b, ptr %c
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

    EXPECT_EQ(sdfg->containers().size(), 4);
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));
    EXPECT_TRUE(sdfg->exists("b"));
    EXPECT_TRUE(sdfg->exists("c"));

    size_t found_select = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 2);
        EXPECT_EQ(data_flow.edges().size(), 1);

        auto& edge = *data_flow.edges().begin();
        EXPECT_EQ(edge.type(), sdfg::data_flow::MemletType::Reference);
        EXPECT_EQ(edge.subset().size(), 1);
        EXPECT_TRUE(sdfg::symbolic::eq(edge.subset().at(0), sdfg::symbolic::zero()));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(edge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_select++;
    }
    EXPECT_EQ(found_select, 2);

    bool found_true = false;
    bool found_false = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.is_unconditional()) {
            continue;
        }
        if (sdfg::symbolic::
                eq(edge.condition(), sdfg::symbolic::Ne(sdfg::symbolic::symbol("a"), sdfg::symbolic::__false__()))) {
            EXPECT_FALSE(found_true);
            found_true = true;
        } else if (sdfg::symbolic::
                       eq(edge.condition(),
                          sdfg::symbolic::Eq(sdfg::symbolic::symbol("a"), sdfg::symbolic::__false__()))) {
            EXPECT_FALSE(found_false);
            found_false = true;
        }
    }
    EXPECT_TRUE(found_true);
    EXPECT_TRUE(found_false);
}

TEST(LiftingTest, VisitCastInst_PtrToInt_Scalar) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr %a) {
entry:
  %0 = ptrtoint ptr %a to i64
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
    EXPECT_TRUE(sdfg->exists("_0"));
    EXPECT_TRUE(sdfg->exists("a"));

    sdfg::types::Pointer opaque_ptr;
    sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int64);
    EXPECT_EQ(sdfg->type("a"), opaque_ptr);
    EXPECT_EQ(sdfg->type("_0"), base_type);

    bool found_cast = false;
    for (auto& edge : sdfg->edges()) {
        if (edge.assignments().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_cast);

        auto rhs = edge.assignments().at(sdfg::symbolic::symbol("_0"));
        EXPECT_TRUE(sdfg::symbolic::eq(rhs, sdfg::symbolic::symbol("a")));
        found_cast = true;
    }
    EXPECT_TRUE(found_cast);
}

TEST(LiftingTest, VisitReturnInst_Undef_Integer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
entry:
  ret i32 undef
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

    EXPECT_EQ(sdfg->containers().size(), 0);
    auto terminal_states = sdfg->terminal_states();
    EXPECT_EQ(std::distance(terminal_states.begin(), terminal_states.end()), 1);

    auto ret_node = dynamic_cast<const sdfg::control_flow::ReturnState*>(&(*terminal_states.begin()));
    EXPECT_EQ(ret_node->data(), "0");
}

TEST(LiftingTest, VisitReturnInst_Undef_Pointer) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo() {
entry:
  ret ptr undef
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

    EXPECT_EQ(sdfg->containers().size(), 0);
    auto terminal_states = sdfg->terminal_states();
    EXPECT_EQ(std::distance(terminal_states.begin(), terminal_states.end()), 1);

    auto ret_node = dynamic_cast<const sdfg::control_flow::ReturnState*>(&(*terminal_states.begin()));
    EXPECT_EQ(ret_node->data(), sdfg::symbolic::__nullptr__()->get_name());
}

TEST(LiftingTest, VisitUnreachableInst) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define ptr @foo() {
entry:
  unreachable
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

    EXPECT_EQ(sdfg->containers().size(), 0);
    auto terminal_states = sdfg->terminal_states();
    EXPECT_EQ(std::distance(terminal_states.begin(), terminal_states.end()), 1);

    auto ret_node = dynamic_cast<const sdfg::control_flow::State*>(&(*terminal_states.begin()));
    EXPECT_EQ(ret_node->dataflow().nodes().size(), 1);
    EXPECT_EQ(ret_node->dataflow().edges().size(), 0);

    auto libnode = *ret_node->dataflow().library_nodes().begin();
    EXPECT_EQ(libnode->code(), sdfg::stdlib::LibraryNodeType_Unreachable);
}
