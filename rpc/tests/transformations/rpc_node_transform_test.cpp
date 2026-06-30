#include <gtest/gtest.h>
#include <memory>
#include <sdfg/transformations/rpc_node_transform.h>
#include <sdfg/util/utils_curl.h>
#include <unordered_set>


#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/deepcopy/structured_sdfg_deep_copy.h"
#include "sdfg/passes/rpc/rpc_context.h"
#include "sdfg/serializer/json_serializer.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/transformations/loop_interchange.h"
#include "sdfg/transformations/loop_tiling.h"
#include "sdfg/transformations/recorder.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/type.h"

using namespace sdfg;

class RPCNodeTransformTest : public ::testing::Test {
protected:
    std::shared_ptr<passes::rpc::RpcContext> ctx_;

    std::unique_ptr<builder::StructuredSDFGBuilder> builder_;
    nlohmann::json desc_;

    void SetUp() override {
        passes::rpc::SimpleRpcContextBuilder ctxBuilder;
        ctx_ = ctxBuilder
                   .initialize_local_default() // localhost:8080/docc
                   .from_env() // $SDFG_RPC_CONFIG can override
                   .from_header_env() // $RPC_HEADER can override/add headers
                   .build();


        builder_ = std::make_unique<builder::StructuredSDFGBuilder>("sdfg_test", FunctionType_CPU);

        auto& root = builder_->subject().root();
        types::Scalar base_desc(types::PrimitiveType::Float);
        types::Array desc_1(base_desc, symbolic::integer(64));
        types::Pointer desc_2(desc_1);

        builder_->add_container("A", desc_2, true);
        builder_->add_container("B", desc_2, true);
        builder_->add_container("C", desc_2, true);

        types::Scalar sym_desc(types::PrimitiveType::UInt64);
        builder_->add_container("K", sym_desc, true);
        builder_->add_container("N", sym_desc, true);
        builder_->add_container("M", sym_desc, true);
        builder_->add_container("i", sym_desc);
        builder_->add_container("j", sym_desc);
        builder_->add_container("k", sym_desc);

        // Define loop 1
        auto bound = symbolic::integer(64);
        auto indvar = symbolic::symbol("i");

        auto& loop = builder_->add_map(
            root,
            indvar,
            symbolic::Lt(symbolic::symbol("i"), bound),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );
        auto& body = loop.root();

        // Define loop 2
        auto bound_2 = symbolic::integer(64);
        auto indvar_2 = symbolic::symbol("j");

        auto& loop_2 = builder_->add_for(
            body,
            indvar_2,
            symbolic::Lt(symbolic::symbol("j"), bound_2),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("j"), symbolic::integer(1))
        );

        auto& body_2 = loop_2.root();

        // Define loop 3
        auto bound_3 = symbolic::integer(64);
        auto indvar_3 = symbolic::symbol("k");

        auto& loop_3 = builder_->add_map(
            body_2,
            indvar_3,
            symbolic::Lt(symbolic::symbol("k"), bound_3),
            symbolic::integer(0),
            symbolic::add(symbolic::symbol("k"), symbolic::integer(1)),
            structured_control_flow::ScheduleType_Sequential::create()
        );

        auto& body_3 = loop_3.root();

        // Add computation
        auto& block = builder_->add_block(body_3);
        auto& a_in = builder_->add_access(block, "A");
        auto& b_in = builder_->add_access(block, "B");
        auto& c_in = builder_->add_access(block, "C");
        auto& c_out = builder_->add_access(block, "C");

        {
            auto& tasklet =
                builder_->add_tasklet(block, data_flow::TaskletCode::fp_fma, "_out", {"_in1", "_in2", "_in3"});
            builder_
                ->add_computational_memlet(block, a_in, tasklet, "_in1", {symbolic::symbol("i"), symbolic::symbol("j")});
            builder_
                ->add_computational_memlet(block, b_in, tasklet, "_in2", {symbolic::symbol("j"), symbolic::symbol("k")});
            builder_
                ->add_computational_memlet(block, c_in, tasklet, "_in3", {symbolic::symbol("i"), symbolic::symbol("k")});
            builder_
                ->add_computational_memlet(block, tasklet, "_out", c_out, {symbolic::symbol("i"), symbolic::symbol("k")});
        }
    };

    void TearDown() override {
        // Cleanup if necessary
    };
};

TEST_F(RPCNodeTransformTest, Matmul_FMA) {
    auto sdfg_initial = builder_->subject().clone();
    sdfg::builder::StructuredSDFGBuilder builder(sdfg_initial);

    // Transfer tuning replayer

    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();
    EXPECT_EQ(outer_loops.size(), 1);

    auto outer_loop = static_cast<structured_control_flow::StructuredLoop*>(outer_loops[0]);
    sdfg::transformations::RPCNodeTransform transfer_tuning(*outer_loop, "sequential", "server", *ctx_, true);
    ASSERT_TRUE(transfer_tuning.can_be_applied(builder, analysis_manager));
    transfer_tuning.apply(builder, analysis_manager);

    sdfg::analysis::AnalysisManager test_analysis_manager(builder.subject());

    auto& test_loop_analysis = test_analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto loop_nest_tree = test_loop_analysis.loop_tree();

    EXPECT_EQ(test_loop_analysis.find_loop_by_indvar("k"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j")]);
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("j_tile0"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k")]
    );

    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("k_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("j_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("i_tile0"), nullptr);

    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("k_tile0"),
        loop_nest_tree[test_loop_analysis.find_loop_by_indvar("j_tile0")]
    );
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("i"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("k_tile0")]
    );
    EXPECT_EQ(
        test_loop_analysis.find_loop_by_indvar("i_tile0"), loop_nest_tree[test_loop_analysis.find_loop_by_indvar("i")]
    );
};

TEST_F(RPCNodeTransformTest, Double_Matmul) {
    {
        analysis::AnalysisManager analysis_manager(builder_->subject());
        auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
        auto outer_loops = loop_analysis.outermost_loops();
        EXPECT_EQ(outer_loops.size(), 1);
        auto loopnest = outer_loops[0];
        sdfg::deepcopy::StructuredSDFGDeepCopy deep_copy(*builder_, builder_->subject().root(), *loopnest);
        deep_copy.copy();
    }

    auto sdfg_initial = builder_->subject().clone();
    sdfg::builder::StructuredSDFGBuilder builder(sdfg_initial);

    // Transfer tuning replayer

    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();
    EXPECT_EQ(outer_loops.size(), 2);

    auto outer_loop = static_cast<structured_control_flow::StructuredLoop*>(outer_loops[0]);
    sdfg::transformations::RPCNodeTransform transfer_tuning(*outer_loop, "sequential", "server", *ctx_, true);
    ASSERT_TRUE(transfer_tuning.can_be_applied(builder, analysis_manager));
    transfer_tuning.apply(builder, analysis_manager);

    sdfg::analysis::AnalysisManager test_analysis_manager(builder.subject());

    auto& test_loop_analysis = test_analysis_manager.get<sdfg::analysis::LoopAnalysis>();

    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("k_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("j_tile0"), nullptr);
    EXPECT_NE(test_loop_analysis.find_loop_by_indvar("i_tile0"), nullptr);
}

TEST_F(RPCNodeTransformTest, UniqueElementIDs) {
    auto sdfg_initial = builder_->subject().clone();
    sdfg::builder::StructuredSDFGBuilder main_builder(sdfg_initial);
    sdfg::analysis::AnalysisManager main_analysis_manager(main_builder.subject());

    auto& initial_loop_analysis = main_analysis_manager.get<analysis::LoopAnalysis>();
    auto initial_outer_loops = initial_loop_analysis.outermost_loops();

    // --- Apply first transform ---
    size_t element_counter_before_first_apply = main_builder.subject().element_counter();

    auto* first_outer_loop = static_cast<structured_control_flow::StructuredLoop*>(initial_outer_loops[0]);
    sdfg::transformations::RPCNodeTransform first_rpc_transform(*first_outer_loop, "sequential", "server", *ctx_, false);
    ASSERT_TRUE(first_rpc_transform.can_be_applied(main_builder, main_analysis_manager));
    first_rpc_transform.apply(main_builder, main_analysis_manager);

    size_t element_counter_after_first_apply = main_builder.subject().element_counter();
    EXPECT_GT(element_counter_after_first_apply, element_counter_before_first_apply);

    // --- Apply second transform ---
    main_analysis_manager.invalidate_all();
    auto& second_loop_analysis = main_analysis_manager.get<analysis::LoopAnalysis>();
    auto remaining_outer_loops = second_loop_analysis.outermost_loops();
    ASSERT_GE(remaining_outer_loops.size(), 1);

    auto* second_outer_loop = static_cast<structured_control_flow::StructuredLoop*>(remaining_outer_loops.back());
    sdfg::transformations::RPCNodeTransform
        second_rpc_transform(*second_outer_loop, "sequential", "server", *ctx_, false);
    ASSERT_TRUE(second_rpc_transform.can_be_applied(main_builder, main_analysis_manager));
    second_rpc_transform.apply(main_builder, main_analysis_manager);

    main_analysis_manager.invalidate_all();
    auto& final_loop_analysis = main_analysis_manager.get<analysis::LoopAnalysis>();

    std::unordered_set<size_t> observed_element_ids;
    for (auto* loop_node : final_loop_analysis.loops()) {
        size_t element_id = loop_node->element_id();
        EXPECT_TRUE(observed_element_ids.insert(element_id).second)
            << "Duplicate element_id " << element_id << " found in final SDFG";
    }
}

TEST_F(RPCNodeTransformTest, HandleUnauthenticatedError) {
    // Test that 401 authentication errors are properly caught and handled
    // This ensures the error handling added for authentication is preserved

    HttpResult result;
    result.curl_code = CURLE_OK;
    result.http_status = 401;
    result.body = R"({"error": "Unauthorized"})";
    result.error_message = "[ERROR] RPC optimization query authentication issue: 401, body: " + result.body;

    sdfg::analysis::AnalysisManager analysis_manager(builder_->subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();
    EXPECT_EQ(outer_loops.size(), 1);
    auto outer_loop = static_cast<structured_control_flow::StructuredLoop*>(outer_loops[0]);

    sdfg::transformations::RPCNodeTransform transform(*outer_loop, "sequential", "server", *ctx_, true);

    // Test parse_rpc_response handles 401 errors
    auto response = transform.parse_rpc_response(result);

    // Should return error string, not RpcOptResponse
    ASSERT_TRUE(std::holds_alternative<std::string>(response));

    // Verify error message contains authentication info
    std::string error = std::get<std::string>(response);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("authentication"), std::string::npos);
}

TEST_F(RPCNodeTransformTest, HandleOtherHttpErrors) {
    // Test that other HTTP errors (like 500) are also caught

    HttpResult result;
    result.curl_code = CURLE_OK;
    result.http_status = 500;
    result.body = R"({"error": "Internal Server Error"})";
    result.error_message = "HTTP error: 500, body: " + result.body;

    sdfg::analysis::AnalysisManager analysis_manager(builder_->subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();
    EXPECT_EQ(outer_loops.size(), 1);
    auto outer_loop = static_cast<structured_control_flow::StructuredLoop*>(outer_loops[0]);

    sdfg::transformations::RPCNodeTransform transform(*outer_loop, "sequential", "server", *ctx_, true);

    // Test parse_rpc_response handles HTTP errors
    auto response = transform.parse_rpc_response(result);

    // Should return error string, not RpcOptResponse
    ASSERT_TRUE(std::holds_alternative<std::string>(response));

    // Verify error message is propagated
    std::string error = std::get<std::string>(response);
    EXPECT_FALSE(error.empty());
    EXPECT_NE(error.find("HTTP error: 500"), std::string::npos);
}

TEST_F(RPCNodeTransformTest, ApplyUnwrapsReferenceTypedContainersFromResponse) {
    // Build a "response" SDFG: clone the main SDFG and add a Reference-typed container.
    // Before the Reference handling was added to apply(), this container would have been
    // stored with Reference type in the main SDFG, causing the final assertion to fail.
    auto response_sdfg = builder_->subject().clone();
    {
        sdfg::builder::StructuredSDFGBuilder resp_builder(*response_sdfg);
        sdfg::types::Scalar inner_type(sdfg::types::PrimitiveType::Float);
        sdfg::codegen::Reference ref_type(inner_type);
        resp_builder.add_container("ref_tmp", ref_type, false, false);
    }

    // Serialize the response SDFG to a temp file so the transfer server can return it.
    const std::string sdfg_path = "/tmp/rpc_ref_test.sdfg.json";
    sdfg::serializer::JSONSerializer serializer;
    sdfg::serializer::JSONSerializer::writeToFile(*response_sdfg, sdfg_path);
    setenv("SDFG_TEST_SDFG_PATH", sdfg_path.c_str(), 1);

    // Build an RPC context that sends the file path as SDFG-Result-Path header,
    // so the transfer server loads this SDFG as its response (same pattern as matmul tests).
    passes::rpc::SimpleRpcContextBuilder ctx_builder;
    auto test_ctx = ctx_builder.initialize_local_default()
                        .add_header("SDFG-Result-Path", std::string(std::getenv("SDFG_TEST_SDFG_PATH")))
                        .build();

    auto sdfg_initial = builder_->subject().clone();
    sdfg::builder::StructuredSDFGBuilder builder(sdfg_initial);
    sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
    auto& loop_analysis = analysis_manager.get<sdfg::analysis::LoopAnalysis>();
    auto outer_loops = loop_analysis.outermost_loops();
    ASSERT_EQ(outer_loops.size(), 1);
    auto outer_loop = static_cast<structured_control_flow::StructuredLoop*>(outer_loops[0]);

    sdfg::transformations::RPCNodeTransform transform(*outer_loop, "sequential", "server", *test_ctx, false);
    ASSERT_TRUE(transform.can_be_applied(builder, analysis_manager));
    transform.apply(builder, analysis_manager);

    // The Reference-typed container must have been added with the inner (Float) type, not
    // wrapped as Reference. Before the fix, this assertion would fail.
    ASSERT_TRUE(builder.subject().exists("ref_tmp"));
    auto& added_type = builder.subject().type("ref_tmp");
    EXPECT_EQ(added_type.type_id(), sdfg::types::TypeID::Scalar);
    EXPECT_EQ(added_type.primitive_type(), sdfg::types::PrimitiveType::Float);
}
