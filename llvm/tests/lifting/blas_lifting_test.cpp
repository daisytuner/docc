#include <gtest/gtest.h>

#include "docc/lifting/lifting.h"
#include "lifting/test_utils.h"

#include <sdfg/data_flow/library_nodes/math/math.h>

using namespace docc;

TEST(BLASLiftingTest, VisitBLASLifting_SDOT) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare float @cblas_sdot(i32, ptr, i32, ptr, i32)

define void @foo(ptr %x, ptr %y) {
  %1 = tail call float @cblas_sdot(i32 32, ptr %x, i32 1, ptr %y, i32 1)
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
    EXPECT_TRUE(sdfg->exists("x"));
    EXPECT_TRUE(sdfg->exists("y"));
    EXPECT_TRUE(sdfg->exists("cblas_sdot"));

    bool found_dot = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        for (auto& node : data_flow.nodes()) {
            if (dynamic_cast<const sdfg::math::blas::DotNode*>(&node)) {
                auto& library_node = dynamic_cast<const sdfg::math::blas::DotNode&>(node);
                EXPECT_EQ(library_node.code(), sdfg::math::blas::LibraryNodeType_DOT);
                EXPECT_EQ(library_node.implementation_type(), sdfg::math::blas::ImplementationType_BLAS);
                EXPECT_EQ(library_node.precision(), sdfg::math::blas::BLAS_Precision::s);

                found_dot = true;
            }
        }
    }
    EXPECT_TRUE(found_dot);
}

TEST(BLASLiftingTest, VisitBLASLifting_DDOT) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare double @cblas_ddot(i32, ptr, i32, ptr, i32)

define void @foo(ptr %x, ptr %y) {
  %1 = tail call double @cblas_ddot(i32 32, ptr %x, i32 1, ptr %y, i32 1)
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
    EXPECT_TRUE(sdfg->exists("x"));
    EXPECT_TRUE(sdfg->exists("y"));

    EXPECT_TRUE(sdfg->exists("cblas_ddot"));
    EXPECT_TRUE(sdfg->is_external("cblas_ddot"));

    bool found_dot = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 4);
        EXPECT_EQ(data_flow.edges().size(), 3);

        for (auto& node : data_flow.nodes()) {
            if (dynamic_cast<const sdfg::math::blas::DotNode*>(&node)) {
                auto& library_node = dynamic_cast<const sdfg::math::blas::DotNode&>(node);
                EXPECT_EQ(library_node.code(), sdfg::math::blas::LibraryNodeType_DOT);
                EXPECT_EQ(library_node.implementation_type(), sdfg::math::blas::ImplementationType_BLAS);
                EXPECT_EQ(library_node.precision(), sdfg::math::blas::BLAS_Precision::d);

                found_dot = true;
            }
        }
    }
    EXPECT_TRUE(found_dot);
}

TEST(BLASLiftingTest, VisitBLASLifting_SGEMM) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @cblas_sgemm(i32, i32, i32, i32, i32, i32, float, ptr, i32, ptr, i32, float, ptr, i32)

define void @foo(ptr %A, ptr %B, ptr %C) {
  tail call void @cblas_sgemm(i32 101, i32 111, i32 111, i32 4, i32 6, i32 8, float 1.0, ptr %A, i32 8, ptr %B, i32 6, float 0.0, ptr %C, i32 4)
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
    EXPECT_TRUE(sdfg->exists("A"));
    EXPECT_TRUE(sdfg->exists("B"));
    EXPECT_TRUE(sdfg->exists("C"));

    EXPECT_TRUE(sdfg->exists("cblas_sgemm"));
    EXPECT_TRUE(sdfg->is_external("cblas_sgemm"));

    bool found_gemm = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 7);
        EXPECT_EQ(data_flow.edges().size(), 6);

        for (auto& node : data_flow.nodes()) {
            if (dynamic_cast<const sdfg::math::blas::GEMMNode*>(&node)) {
                auto& library_node = dynamic_cast<const sdfg::math::blas::GEMMNode&>(node);
                EXPECT_EQ(library_node.code(), sdfg::math::blas::LibraryNodeType_GEMM);
                EXPECT_EQ(library_node.implementation_type(), sdfg::math::blas::ImplementationType_BLAS);
                EXPECT_EQ(library_node.precision(), sdfg::math::blas::BLAS_Precision::s);

                found_gemm = true;
            }
        }
    }
    EXPECT_TRUE(found_gemm);
}

TEST(BLASLiftingTest, VisitBLASLifting_DGEMM) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @cblas_dgemm(i32, i32, i32, i32, i32, i32, double, ptr, i32, ptr, i32, double, ptr, i32)

define void @foo(ptr %A, ptr %B, ptr %C) {
  tail call void @cblas_dgemm(i32 101, i32 111, i32 111, i32 4, i32 6, i32 8, double 1.0, ptr %A, i32 8, ptr %B, i32 6, double 0.0, ptr %C, i32 4)
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
    EXPECT_TRUE(sdfg->exists("A"));
    EXPECT_TRUE(sdfg->exists("B"));
    EXPECT_TRUE(sdfg->exists("C"));

    EXPECT_TRUE(sdfg->exists("cblas_dgemm"));
    EXPECT_TRUE(sdfg->is_external("cblas_dgemm"));

    bool found_gemm = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 7);
        EXPECT_EQ(data_flow.edges().size(), 6);

        for (auto& node : data_flow.nodes()) {
            if (dynamic_cast<const sdfg::math::blas::GEMMNode*>(&node)) {
                auto& library_node = dynamic_cast<const sdfg::math::blas::GEMMNode&>(node);
                EXPECT_EQ(library_node.code(), sdfg::math::blas::LibraryNodeType_GEMM);
                EXPECT_EQ(library_node.implementation_type(), sdfg::math::blas::ImplementationType_BLAS);
                EXPECT_EQ(library_node.precision(), sdfg::math::blas::BLAS_Precision::d);

                found_gemm = true;
            }
        }
    }
    EXPECT_TRUE(found_gemm);
}
