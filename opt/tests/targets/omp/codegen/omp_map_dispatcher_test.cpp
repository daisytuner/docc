#include "sdfg/targets/omp/codegen/omp_map_dispatcher.h"
#include "sdfg/targets/omp/schedule.h"

#include <sdfg/codegen/language_extensions/c_language_extension.h>

#include <gtest/gtest.h>

using namespace sdfg;

TEST(OMPMapDispatcherTest, DispatchNode) {
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
        omp::ScheduleType_OMP::create()
    );

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    omp::OMPMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(main_stream.str(), "// Map\n#pragma omp parallel for schedule(static)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n");
}

TEST(OMPMapDispatcherTest, DispatchNodeScheduleDynamic) {
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
        omp::ScheduleType_OMP::create()
    );

    ScheduleType schedule = omp::ScheduleType_OMP::create();

    omp::ScheduleType_OMP::omp_schedule(schedule, omp::OpenMPSchedule::Dynamic);

    builder.update_schedule_type(loop, schedule);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    omp::OMPMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(
        main_stream.str(), "// Map\n#pragma omp parallel for schedule(dynamic)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n"
    );
}

TEST(OMPMapDispatcherTest, DispatchNodeNumThreads) {
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
        omp::ScheduleType_OMP::create()
    );

    ScheduleType schedule = omp::ScheduleType_OMP::create();

    omp::ScheduleType_OMP::omp_schedule(schedule, omp::OpenMPSchedule::Dynamic);
    omp::ScheduleType_OMP::num_threads(schedule, symbolic::integer(4));

    builder.update_schedule_type(loop, schedule);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    omp::OMPMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(
        main_stream.str(),
        "// Map\n#pragma omp parallel for schedule(dynamic) num_threads(4)\nfor(i = 0;i < 10;i = 1 + i)\n{\n}\n"
    );
}

// Regression test: a conjunctive loop condition (i < N && i < i_tile0 + 8) is NOT a
// legal OpenMP canonical loop form. The dispatcher must collapse it via
// canonical_bound_upper() into a single relational expression `i < min(N, i_tile0 + 8)`
// that the OpenMP compiler will accept.
TEST(OMPMapDispatcherTest, DispatchNodeConjunctiveConditionCollapsedToCanonicalBound) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar int_type(types::PrimitiveType::Int32);
    builder.add_container("i", int_type);
    builder.add_container("N", int_type, true);
    builder.add_container("i_tile0", int_type, true);

    auto i = symbolic::symbol("i");
    auto N = symbolic::symbol("N");
    auto i_tile0 = symbolic::symbol("i_tile0");

    auto& loop = builder.add_map(
        root,
        i,
        symbolic::And(symbolic::Lt(i, N), symbolic::Lt(i, symbolic::add(i_tile0, symbolic::integer(8)))),
        i_tile0,
        symbolic::add(i, symbolic::integer(1)),
        omp::ScheduleType_OMP::create()
    );

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    omp::OMPMapDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, loop, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    const std::string out = main_stream.str();

    // The raw conjunction would have embedded "&&", which is illegal in an OpenMP
    // canonical loop form. The dispatcher must collapse it via canonical_bound_upper().
    EXPECT_EQ(out.find("&&"), std::string::npos)
        << "OpenMP dispatcher must not emit a conjunctive loop condition. Output:\n"
        << out;

    // The collapsed condition uses the min(...) of the two upper bounds.
    EXPECT_NE(out.find("i < __daisy_min("), std::string::npos)
        << "Expected collapsed canonical bound `i < __daisy_min(...)` in output:\n"
        << out;
    EXPECT_NE(out.find("N"), std::string::npos);
    EXPECT_NE(out.find("i_tile0"), std::string::npos);
}
