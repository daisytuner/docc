#include <gtest/gtest.h>

#include "docc/lifting/lifting.h"
#include "lifting/test_utils.h"

#include <sdfg/data_flow/library_nodes/call_node.h>
#include <sdfg/data_flow/library_nodes/invoke_node.h>

using namespace docc;

TEST(FunctionListingTest, VisitCall) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_ZN4Foam5UListIfE5beginEv()

define void @foo() {
entry:
  %0 = tail call ptr @_ZN4Foam5UListIfE5beginEv()
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
    EXPECT_TRUE(sdfg->exists("_ZN4Foam5UListIfE5beginEv"));
    EXPECT_TRUE(sdfg->is_external("_ZN4Foam5UListIfE5beginEv"));

    bool found_call = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_call);

        const sdfg::data_flow::CallNode* call_node = nullptr;
        for (auto& node : state.dataflow().nodes()) {
            if (auto lib_node = dynamic_cast<const sdfg::data_flow::CallNode*>(&node)) {
                call_node = lib_node;
                found_call = true;
                break;
            }
        }
        EXPECT_NE(call_node, nullptr);

        EXPECT_EQ(state.dataflow().in_degree(*call_node), 0);
        EXPECT_EQ(state.dataflow().out_degree(*call_node), 1);

        auto& oedge = *state.dataflow().out_edges(*call_node).begin();
        EXPECT_EQ(oedge.src_conn(), "_ret");
        EXPECT_EQ(oedge.dst_conn(), "void");
        EXPECT_EQ(oedge.subset().size(), 0);
        EXPECT_EQ(oedge.base_type(), sdfg::types::Pointer());

        auto& dst = static_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
        EXPECT_EQ(dst.data(), "_0");
    }
    EXPECT_TRUE(found_call);
}

TEST(FunctionListingTest, VisitCall_ReadonlyArgs) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_ZN4Foam5UListIfE5beginEv(i32, ptr)

define void @foo(ptr %a, i32 %b) {
entry:
  %0 = tail call ptr @_ZN4Foam5UListIfE5beginEv(i32 %b, ptr %a)
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
    EXPECT_TRUE(sdfg->exists("_ZN4Foam5UListIfE5beginEv"));
    EXPECT_TRUE(sdfg->is_external("_ZN4Foam5UListIfE5beginEv"));

    bool found_call = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }
        EXPECT_FALSE(found_call);

        const sdfg::data_flow::CallNode* call_node = nullptr;
        for (auto& node : state.dataflow().nodes()) {
            if (auto lib_node = dynamic_cast<const sdfg::data_flow::CallNode*>(&node)) {
                call_node = lib_node;
                found_call = true;
                break;
            }
        }
        EXPECT_NE(call_node, nullptr);

        EXPECT_EQ(state.dataflow().in_degree(*call_node), 2);
        EXPECT_EQ(state.dataflow().out_degree(*call_node), 2);

        auto oedge1 = &(*state.dataflow().out_edges(*call_node).begin());
        auto oedge2 = &(*++state.dataflow().out_edges(*call_node).begin());
        if (oedge1->src_conn() != "_ret") {
            std::swap(oedge1, oedge2);
        }
        EXPECT_EQ(oedge1->src_conn(), "_ret");

        EXPECT_EQ(oedge1->src_conn(), "_ret");
        EXPECT_EQ(oedge1->dst_conn(), "void");
        EXPECT_EQ(oedge1->subset().size(), 0);
        EXPECT_EQ(oedge1->base_type(), sdfg::types::Pointer());

        auto& dst = static_cast<const sdfg::data_flow::AccessNode&>(oedge1->dst());
        EXPECT_EQ(dst.data(), "_0");

        EXPECT_EQ(oedge2->src_conn(), "_arg1");
    }
    EXPECT_TRUE(found_call);
}

TEST(FunctionListingTest, VisitInvoke) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare ptr @_ZN4Foam5UListIfE5beginEv()

define i1 @foo() personality ptr @__gxx_personality_v0 {
entry:
  %0 = invoke ptr @_ZN4Foam5UListIfE5beginEv()
          to label %normal unwind label %exception

normal:
  ret i1 true

exception:
  ret i1 false
}

declare i32 @__gxx_personality_v0(...)
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
    EXPECT_TRUE(sdfg->exists("__unwind__0"));
    EXPECT_TRUE(sdfg->exists("_ZN4Foam5UListIfE5beginEv"));
    EXPECT_TRUE(sdfg->is_external("_ZN4Foam5UListIfE5beginEv"));

    bool found_call = false;
    bool found_unwind = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        const sdfg::data_flow::InvokeNode* invoke_node = nullptr;
        for (auto& node : state.dataflow().nodes()) {
            if (auto lib_node = dynamic_cast<const sdfg::data_flow::InvokeNode*>(&node)) {
                invoke_node = lib_node;
                found_call = true;
                break;
            }
        }

        if (invoke_node != nullptr) {
            EXPECT_EQ(state.dataflow().in_degree(*invoke_node), 0);
            EXPECT_EQ(state.dataflow().out_degree(*invoke_node), 2); // _ret and _unwind

            // Check for return value edge
            bool found_ret_edge = false;
            bool found_unwind_edge = false;
            for (auto& oedge : state.dataflow().out_edges(*invoke_node)) {
                if (oedge.src_conn() == "_ret") {
                    found_ret_edge = true;
                    EXPECT_EQ(oedge.dst_conn(), "void");
                    EXPECT_EQ(oedge.subset().size(), 0);
                    EXPECT_EQ(oedge.base_type(), sdfg::types::Pointer());
                    auto& dst = static_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
                    EXPECT_EQ(dst.data(), "_0");
                } else if (oedge.src_conn() == "_unwind") {
                    found_unwind_edge = true;
                    found_unwind = true;
                    EXPECT_EQ(oedge.base_type(), sdfg::types::Scalar(sdfg::types::PrimitiveType::Bool));
                    auto& dst = static_cast<const sdfg::data_flow::AccessNode&>(oedge.dst());
                    EXPECT_EQ(dst.data(), "__unwind__0");
                }
            }
            EXPECT_TRUE(found_ret_edge);
            EXPECT_TRUE(found_unwind_edge);
        }
    }
    EXPECT_TRUE(found_call);
    EXPECT_TRUE(found_unwind);
}

TEST(FunctionListingTest, VisitCall_ByVal) {
    const std::string ir = R"(
; ModuleID = 'test'
source_filename = "test.ll"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32 }

declare void @bar(ptr byval(%struct.A))

define void @foo(ptr %a) {
entry:
  call void @bar(ptr byval(%struct.A) %a)
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
    EXPECT_TRUE(sdfg->exists("bar"));

    auto& bar_type = sdfg->type("bar");
    EXPECT_EQ(bar_type.type_id(), sdfg::types::TypeID::Function);
    auto& func_type = static_cast<const sdfg::types::Function&>(bar_type);
    ASSERT_EQ(func_type.num_params(), 1);
    EXPECT_EQ(func_type.param_type(sdfg::symbolic::zero()).type_id(), sdfg::types::TypeID::Pointer);

    bool found_call = false;
    for (auto& state : sdfg->states()) {
        if (state.dataflow().nodes().size() == 0) {
            continue;
        }

        const sdfg::data_flow::CallNode* call_node = nullptr;
        for (auto& node : state.dataflow().nodes()) {
            if (auto lib_node = dynamic_cast<const sdfg::data_flow::CallNode*>(&node)) {
                call_node = lib_node;
                found_call = true;
                break;
            }
        }
        if (!call_node) continue;

        EXPECT_EQ(state.dataflow().in_degree(*call_node), 1);
        EXPECT_EQ(state.dataflow().out_degree(*call_node), 0);

        auto& iedge = *state.dataflow().in_edges(*call_node).begin();
        EXPECT_EQ(iedge.dst_conn(), "_arg0");
        EXPECT_EQ(iedge.subset().size(), 0);
        EXPECT_EQ(iedge.base_type().type_id(), sdfg::types::TypeID::Pointer);
    }
    EXPECT_TRUE(found_call);
}
