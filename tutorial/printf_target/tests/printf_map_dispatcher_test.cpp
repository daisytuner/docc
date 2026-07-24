#include <gtest/gtest.h>
#include <memory>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/instrumentation/instrumentation_info.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/tasklet.h"
#include "sdfg/function.h"
#include "sdfg/symbolic/symbolic.h"

#include "printf_map_dispatcher.h"
#include "printf_target.h"

namespace sdfg::printf_target {

TEST(PrintfMapDispatcher, BasicDispatcherTest) {
    // Create a simple SDFG with a map
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar float_desc(types::PrimitiveType::Float);
    types::Pointer pointer_type(float_desc);
    types::Scalar int_desc(types::PrimitiveType::Int32);

    builder.add_container("i", int_desc);
    builder.add_container("A", pointer_type);

    // Create a map with Printf schedule
    structured_control_flow::ScheduleType printf_schedule = ScheduleType_Printf::create();

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(100));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, printf_schedule);

    // Add a simple tasklet in the map body
    auto& block = builder.add_block(map.root());
    auto& access = builder.add_access(block, "A");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "out_", {"in_"});
    auto& constant = builder.add_constant(block, "0.0f", types::Scalar(types::PrimitiveType::Float));

    builder.add_computational_memlet(block, constant, tasklet, "in_", {}, types::Scalar(types::PrimitiveType::Float));
    builder.add_computational_memlet(block, tasklet, "out_", access, {symbolic::symbol("i")}, pointer_type);

    // Create dispatcher and generate code
    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());

    PrintfMapDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, map, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_snippet_factory;

    dispatcher.dispatch_node(main_stream, globals_stream, library_snippet_factory);

    std::string generated_code = main_stream.str();

    std::cout << "Generated Code:\n" << generated_code << std::endl;

    // Check that printf statements are generated
    EXPECT_NE(generated_code.find("[PRINTF_TARGET] Entering map"), std::string::npos);
    EXPECT_NE(generated_code.find("[PRINTF_TARGET] Exiting map"), std::string::npos);
    EXPECT_NE(generated_code.find("Indvar: "), std::string::npos);
    EXPECT_NE(generated_code.find("Iterations: "), std::string::npos);

    // Check that a for loop is generated
    EXPECT_NE(generated_code.find("for (long i = 0;"), std::string::npos);
}

TEST(PrintfMapDispatcher, ScheduleTypeTest) {
    // Test ScheduleType_Printf creation and value
    auto schedule = ScheduleType_Printf::create();

    EXPECT_EQ(schedule.value(), "Printf");
    EXPECT_EQ(ScheduleType_Printf::value(), "Printf");
}

TEST(PrintfMapDispatcher, InstrumentationInfoTest) {
    // Create a simple SDFG with a map
    sdfg::builder::StructuredSDFGBuilder builder("test_sdfg", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar int_desc(types::PrimitiveType::Int32);
    builder.add_container("i", int_desc);

    structured_control_flow::ScheduleType printf_schedule = ScheduleType_Printf::create();

    auto condition = symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10));
    auto init = symbolic::integer(0);
    auto update = symbolic::add(symbolic::symbol("i"), symbolic::integer(1));

    auto& map = builder.add_map(root, symbolic::symbol("i"), condition, init, update, printf_schedule);

    // Add empty block to satisfy requirements
    builder.add_block(map.root());

    codegen::CLanguageExtension language_extension(builder.subject());
    auto instrumentation = codegen::InstrumentationPlan::none(builder.subject());
    auto arg_capture = codegen::ArgCapturePlan::none(builder.subject());
    analysis::AnalysisManager analysis_manager(builder.subject());

    PrintfMapDispatcher
        dispatcher(language_extension, builder.subject(), analysis_manager, map, *instrumentation, *arg_capture);

    auto info = dispatcher.instrumentation_info();

    EXPECT_EQ(info.target_type().value(), TargetType_Printf.value());
    EXPECT_EQ(info.element_desc(), "map");
}

} // namespace sdfg::printf_target
