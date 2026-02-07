#include "sdfg/codegen/dispatchers/map_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/dispatchers/node_dispatcher_registry.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/structured_control_flow/map.h"

using namespace sdfg;

TEST(SequentialMapDispatcherTest, DispatchNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType_Sequential::create()
    );

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::SequentialMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "// Map\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n");
}

TEST(MapDispatcherTest, RedirectDispatch) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& loop = builder.add_map(
        root,
        symbolic::symbol("i"),
        symbolic::Lt(symbolic::symbol("i"), symbolic::integer(10)),
        symbolic::integer(0),
        symbolic::add(symbolic::symbol("i"), symbolic::integer(1)),
        structured_control_flow::ScheduleType("mock", structured_control_flow::ScheduleTypeCategory::None)
    );

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);

    bool dispatch_called = false;
    bool dispatch_node_called = false;

    class MockDispatcher : public codegen::NodeDispatcher {
    public:
        bool& dispatch_called;
        bool& dispatch_node_called;

        MockDispatcher(
            codegen::LanguageExtension& language_extension,
            StructuredSDFG& sdfg,
            analysis::AnalysisManager& analysis_manager,
            structured_control_flow::ControlFlowNode& node,
            codegen::InstrumentationPlan& instrumentation_plan,
            codegen::ArgCapturePlan& arg_capture_plan,
            bool& dispatch_called,
            bool& dispatch_node_called
        )
            : NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
              dispatch_called(dispatch_called), dispatch_node_called(dispatch_node_called) {}

        void dispatch_node(
            codegen::PrettyPrinter& main_stream,
            codegen::PrettyPrinter& globals_stream,
            codegen::CodeSnippetFactory& library_snippet_factory
        ) override {
            dispatch_node_called = true;
        }

        void dispatch(
            codegen::PrettyPrinter& main_stream,
            codegen::PrettyPrinter& globals_stream,
            codegen::CodeSnippetFactory& library_snippet_factory
        ) override {
            dispatch_called = true;
            NodeDispatcher::dispatch(main_stream, globals_stream, library_snippet_factory);
        }
    };

    codegen::MapDispatcherRegistry::instance().register_map_dispatcher(
        "mock",
        [&](codegen::LanguageExtension& language_extension,
            StructuredSDFG& sdfg,
            analysis::AnalysisManager& analysis_manager,
            structured_control_flow::Map& map,
            codegen::InstrumentationPlan& instrumentation,
            codegen::ArgCapturePlan& arg_capture) {
            auto it = std::make_unique<MockDispatcher>(
                language_extension,
                sdfg,
                analysis_manager,
                map,
                instrumentation,
                arg_capture,
                dispatch_called,
                dispatch_node_called
            );
            return it;
        }
    );

    codegen::MapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;

    dispatcher.dispatch(main_stream, globals_stream, library_factory);

    EXPECT_TRUE(dispatch_called);
    EXPECT_TRUE(dispatch_node_called);
}
