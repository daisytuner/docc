#include <gtest/gtest.h>

#include "docc/lifting/lifting.h"
#include "lifting/test_utils.h"

#include <sdfg/data_flow/library_nodes/math/math.h>
#include <sdfg/data_flow/library_nodes/stdlib/stdlib.h>

using namespace docc;

TEST(IntrinsicLiftingTest, VisitMathIntrinsic_Scalar_FP32) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float %i, float %j) {
entry:
  %0 = call float @llvm.fma.f32(float %i, float %j, float 1.0)
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
        EXPECT_EQ(data_flow.nodes().size(), 5);
        EXPECT_EQ(data_flow.edges().size(), 4);

        auto tasklet = dynamic_cast<const sdfg::math::cmath::CMathNode*>(*data_flow.library_nodes().begin());
        EXPECT_EQ(tasklet->name(), "fmaf");
        EXPECT_EQ(tasklet->inputs().size(), 3);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");
        EXPECT_EQ(tasklet->inputs().at(2), "_in3");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        auto iedge3 = &(*++(++data_flow.in_edges(*tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge3);
        }
        if (iedge2->dst_conn() != "_in2") {
            std::swap(iedge2, iedge3);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        EXPECT_EQ(iedge3->subset().size(), 0);
        EXPECT_EQ(iedge3->base_type(), base_type);
        EXPECT_EQ(iedge3->dst_conn(), "_in3");
        auto& src3 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge3->src());
        EXPECT_EQ(src3.data(), "1.0f");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, VisitMathIntrinsic_Scalar_FP64) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(double %i, double %j) {
entry:
  %0 = call double @llvm.fma.f64(double %i, double %j, double 1.0)
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
        EXPECT_EQ(data_flow.nodes().size(), 5);
        EXPECT_EQ(data_flow.edges().size(), 4);

        auto tasklet = dynamic_cast<const sdfg::math::cmath::CMathNode*>(*data_flow.library_nodes().begin());
        EXPECT_EQ(tasklet->name(), "fma");
        EXPECT_EQ(tasklet->inputs().size(), 3);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");
        EXPECT_EQ(tasklet->inputs().at(2), "_in3");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Double);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        auto iedge3 = &(*++(++data_flow.in_edges(*tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge3);
        }
        if (iedge2->dst_conn() != "_in2") {
            std::swap(iedge2, iedge3);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        EXPECT_EQ(iedge3->subset().size(), 0);
        EXPECT_EQ(iedge3->base_type(), base_type);
        EXPECT_EQ(iedge3->dst_conn(), "_in3");
        auto& src3 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge3->src());
        EXPECT_EQ(src3.data(), "1.0");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, VisitMathIntrinsic_Scalar_SameInput) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float %i) {
entry:
  %0 = call float @llvm.pow.f32(float %i, float %i)
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

        auto tasklet = dynamic_cast<const sdfg::math::cmath::CMathNode*>(*data_flow.library_nodes().begin());
        EXPECT_EQ(tasklet->name(), "powf");
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "i");
        EXPECT_EQ(&src1, &src2);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, VisitMathIntrinsic_Vector) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x float> %i, <4 x float> %j) {
entry:
  %0 = call <4 x float> @llvm.fma.v4f32(<4 x float> %i, <4 x float> %j, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
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

    size_t found_add = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 5);
        EXPECT_EQ(data_flow.edges().size(), 4);

        auto tasklet = dynamic_cast<const sdfg::math::cmath::CMathNode*>(*data_flow.library_nodes().begin());
        EXPECT_EQ(tasklet->name(), "fmaf");
        EXPECT_EQ(tasklet->inputs().size(), 3);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");
        EXPECT_EQ(tasklet->inputs().at(2), "_in3");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Structure struct_type("__daisy_vec_14_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        auto iedge3 = &(*++(++data_flow.in_edges(*tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge3);
        }
        if (iedge2->dst_conn() != "_in2") {
            std::swap(iedge2, iedge3);
        }

        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_EQ(iedge1->base_type(), struct_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_EQ(iedge2->base_type(), struct_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        EXPECT_EQ(iedge3->subset().size(), 1);
        EXPECT_EQ(iedge3->base_type(), struct_type);
        EXPECT_EQ(iedge3->dst_conn(), "_in3");
        auto& src3 = dynamic_cast<const sdfg::data_flow::ConstantNode&>(iedge3->src());
        EXPECT_EQ(src3.data(), "{1.0f, 1.0f, 1.0f, 1.0f}");
        EXPECT_EQ(src3.type(), sdfg::types::Structure("__daisy_vec_14_4"));

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_EQ(oedge.base_type(), struct_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add++;
    }
    EXPECT_EQ(found_add, 4);
}

TEST(IntrinsicLiftingTest, VisitMathIntrinsic_Vector_SameInput) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(<4 x float> %i) {
entry:
  %0 = call <4 x float> @llvm.pow.v4f32(<4 x float> %i, <4 x float> %i)
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

    size_t found_add = 0;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 3);

        auto tasklet = dynamic_cast<const sdfg::math::cmath::CMathNode*>(*data_flow.library_nodes().begin());
        EXPECT_EQ(tasklet->name(), "powf");
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);
        sdfg::types::Structure struct_type("__daisy_vec_14_4");

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 1);
        EXPECT_EQ(iedge1->base_type(), struct_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 1);
        EXPECT_EQ(iedge2->base_type(), struct_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "i");
        EXPECT_EQ(&src1, &src2);

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 1);
        EXPECT_EQ(oedge.base_type(), struct_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add++;
    }
    EXPECT_EQ(found_add, 4);
}

TEST(IntrinsicLiftingTest, VisitFMulAdd) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float %i, float %j) {
entry:
  %0 = call float @llvm.fmuladd.f32(float %i, float %j, float 1.0)
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
        EXPECT_EQ(data_flow.nodes().size(), 5);
        EXPECT_EQ(data_flow.edges().size(), 4);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::fp_fma);
        EXPECT_EQ(tasklet->inputs().size(), 3);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");
        EXPECT_EQ(tasklet->inputs().at(2), "_in3");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        auto iedge3 = &(*++(++data_flow.in_edges(*tasklet).begin()));
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge3);
        }
        if (iedge2->dst_conn() != "_in2") {
            std::swap(iedge2, iedge3);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        EXPECT_EQ(iedge3->subset().size(), 0);
        EXPECT_EQ(iedge3->base_type(), base_type);
        EXPECT_EQ(iedge3->dst_conn(), "_in3");
        auto& src3 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge3->src());
        EXPECT_EQ(src3.data(), "1.0f");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, Visitsmax) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = call i32 @llvm.smax.i32(i32 %i, i32 %j)
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
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_smax);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, Visitsmin) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = call i32 @llvm.smin.i32(i32 %i, i32 %j)
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
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_smin);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, Visitscmp) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = call i32 @llvm.scmp.i32(i32 %i, i32 %j)
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
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_scmp);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, Visitumax) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = call i32 @llvm.umax.i32(i32 %i, i32 %j)
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
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_umax);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, Visitumin) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = call i32 @llvm.umin.i32(i32 %i, i32 %j)
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
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_umin);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, Visitucmp) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i, i32 %j) {
entry:
  %0 = call i32 @llvm.ucmp.i32(i32 %i, i32 %j)
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
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_ucmp);
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() != "_in1") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}

TEST(IntrinsicLiftingTest, Visitabs_strict) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i) {
entry:
  %0 = call i32 @llvm.abs.i32(i32 %i, i1 true)
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

    bool found_tasklet = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_tasklet);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::int_abs);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_tasklet = true;
    }
    EXPECT_TRUE(found_tasklet);
}

TEST(IntrinsicLiftingTest, Visit_expect) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i) {
entry:
  %0 = call i32 @llvm.expect.i32(i32 %i, i32 12)
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

    bool found_tasklet = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_tasklet);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_tasklet = true;
    }
    EXPECT_TRUE(found_tasklet);
}

TEST(IntrinsicLiftingTest, Visit_expect_with_probability) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(i32 %i) {
entry:
  %0 = call i32 @llvm.expect.with.probability.i32(i32 %i, i32 12, double 0.9)
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

    bool found_tasklet = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_tasklet);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 3);
        EXPECT_EQ(data_flow.edges().size(), 2);

        auto tasklet = *data_flow.tasklets().begin();
        EXPECT_EQ(tasklet->code(), sdfg::data_flow::TaskletCode::assign);
        EXPECT_EQ(tasklet->inputs().size(), 1);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Int32);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_tasklet = true;
    }
    EXPECT_TRUE(found_tasklet);
}

TEST(IntrinsicLiftingTest, Visit_trap) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo() {
entry:
  tail call void @llvm.trap()
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

    EXPECT_EQ(sdfg->containers().size(), 0);

    bool found_libnode = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_libnode);

        auto& data_flow = state.dataflow();
        EXPECT_EQ(data_flow.nodes().size(), 1);
        EXPECT_EQ(data_flow.edges().size(), 0);

        auto libnode = *data_flow.library_nodes().begin();
        EXPECT_EQ(libnode->code(), sdfg::stdlib::LibraryNodeType_Trap);
        EXPECT_EQ(libnode->inputs().size(), 0);
        EXPECT_EQ(libnode->outputs().size(), 0);

        found_libnode = true;
    }
    EXPECT_TRUE(found_libnode);
}

TEST(IntrinsicLiftingTest, VisitPowi) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(float %i, i32 %j) {
entry:
  %0 = call float @llvm.powi.f32(float %i, i32 %j)
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

        auto tasklet = dynamic_cast<const sdfg::math::cmath::CMathNode*>(*data_flow.library_nodes().begin());
        EXPECT_EQ(tasklet->name(), "powf");
        EXPECT_EQ(tasklet->inputs().size(), 2);
        EXPECT_EQ(tasklet->inputs().at(0), "_in1");
        EXPECT_EQ(tasklet->inputs().at(1), "_in2");

        sdfg::types::Scalar base_type(sdfg::types::PrimitiveType::Float);

        auto iedge1 = &(*data_flow.in_edges(*tasklet).begin());
        auto iedge2 = &(*++data_flow.in_edges(*tasklet).begin());
        if (iedge1->dst_conn() == "_in2") {
            std::swap(iedge1, iedge2);
        }

        EXPECT_EQ(iedge1->subset().size(), 0);
        EXPECT_EQ(iedge1->base_type(), base_type);
        EXPECT_EQ(iedge1->dst_conn(), "_in1");
        auto& src1 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge1->src());
        EXPECT_EQ(src1.data(), "i");

        EXPECT_EQ(iedge2->subset().size(), 0);
        EXPECT_EQ(iedge2->base_type(), base_type);
        EXPECT_EQ(iedge2->dst_conn(), "_in2");
        auto& src2 = dynamic_cast<const sdfg::data_flow::AccessNode&>(iedge2->src());
        EXPECT_EQ(src2.data(), "j");

        auto& oedge = *data_flow.out_edges(*tasklet).begin();
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), base_type);
        EXPECT_EQ(oedge.src_conn(), tasklet->outputs().at(0));

        auto& dst = dynamic_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");

        found_add = true;
    }
    EXPECT_TRUE(found_add);
}
