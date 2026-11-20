#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"

using namespace sdfg;

TEST(BlockDispatcherTest, DispatchNode_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::BlockDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, block, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(globals_stream.str(), "");
    EXPECT_EQ(main_stream.str(), "");
}

TEST(BlockDispatcherTest, DispatchNode_TopologicalOrder) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& access_node_in1 = builder.add_access(block, "a");
    auto& access_node_out1 = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_node_in1, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_node_out1, {});

    auto& access_node_out2 = builder.add_access(block, "a");
    auto& tasklet_2 = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, access_node_out1, tasklet_2, "_in", {});
    builder.add_computational_memlet(block, tasklet_2, "_out", access_node_out2, {});

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::BlockDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, block, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(
        main_stream.str(),
        "{\n    int _in = a;\n    int _out;\n\n    _out = _in;\n\n    b = _out;\n}\n{\n    int _in = b;\n    int "
        "_out;\n\n    _out = _in;\n\n    a = _out;\n}\n"
    );
    EXPECT_TRUE(library_factory.snippets().empty());
    EXPECT_EQ(globals_stream.str(), "");
}


TEST(DataFlowDispatcherTest, DispatchTasklet) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("a", desc);
    builder.add_container("b", desc);
    builder.add_container("c", desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    auto& access_node_3 = builder.add_access(block, "c");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::int_add, "_out", {"_in1", "_in2"});
    builder.add_computational_memlet(block, access_node_1, tasklet, "_in1", {});
    builder.add_computational_memlet(block, access_node_2, tasklet, "_in2", {});
    builder.add_computational_memlet(block, tasklet, "_out", access_node_3, {});

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(
        main_stream.str(),
        "{\n    int _in1 = a;\n    int _in2 = b;\n    int _out;\n\n    _out = _in1 + "
        "_in2;\n\n    c = _out;\n}\n"
    );
}

TEST(DataFlowDispatcherTest, DispatchRef_Empty) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_reference_memlet(block, access_node_1, access_node_2, {}, pointer_type);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    b = &a;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Subset) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_reference_memlet(block, access_node_1, access_node_2, {symbolic::integer(1)}, pointer_type);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    b = &((int *) a)[1];\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Nullptr) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Void);
    types::Pointer pointer_type(base_type);

    types::Pointer opaque_desc;
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_constant(block, symbolic::__nullptr__()->get_name(), types::Pointer());
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_reference_memlet(block, access_node_1, access_node_2, {}, pointer_type);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    b = NULL;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_Address) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Void);
    types::Pointer pointer_type(base_type);

    types::Pointer opaque_desc;
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_constant(block, "100", types::Scalar(types::PrimitiveType::Int64));
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_reference_memlet(block, access_node_1, access_node_2, {}, pointer_type);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    b = 100;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchDeref_Load) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    types::Pointer pointer_type_2(static_cast<const types::IType&>(pointer_type));

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_dereference_memlet(block, access_node_1, access_node_2, true, pointer_type_2);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    b = *((int **) a);\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchDeref_Store) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    types::Pointer pointer_type_2(static_cast<const types::IType&>(pointer_type));

    types::Pointer opaque_desc;
    builder.add_container("a", opaque_desc);
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_dereference_memlet(block, access_node_1, access_node_2, false, pointer_type_2);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    *((int **) b) = a;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchDeref_Store_Nullptr) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    types::Pointer pointer_type_2(static_cast<const types::IType&>(pointer_type));

    types::Pointer opaque_desc;
    builder.add_container("b", opaque_desc);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_constant(block, symbolic::__nullptr__()->get_name(), types::Pointer());
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_dereference_memlet(block, access_node_1, access_node_2, false, pointer_type_2);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    codegen::CLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    *((int **) b) = NULL;\n}\n");
}

namespace {

/// Spells connectors, element addresses and converting assignments its own way, as a language other than C might.
class HookedLanguageExtension : public codegen::CLanguageExtension {
public:
    using CLanguageExtension::CLanguageExtension;

    std::string identifier(const std::string& name) const override {
        return "v" + name;
    }

    /// Reads its inputs by their identifiers, as the dispatcher declares them.
    std::string tasklet(const data_flow::Tasklet& tasklet) override {
        return identifier(tasklet.inputs().at(0));
    }

    std::string converting_assignment(const std::string& lhs, const types::Scalar& type, const std::string& rhs) override {
        return lhs + " = convert(" + rhs + ");";
    }

    std::string
    element_address(const std::string& base, const types::IType& base_type, const data_flow::Subset& indices) override {
        return "address(" + base + ", " + expression(indices.at(0)) + ")";
    }
};

} // namespace

TEST(AssignmentDispatcherTest, ConvertsAssignmentsToScalars) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& root = builder.subject().root();
    builder.add_container("i", types::Scalar(types::PrimitiveType::Int32));
    auto& block = builder.add_assignments(root, {{symbolic::symbol("i"), symbolic::integer(10)}});

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    HookedLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    auto arg_capture = codegen::ArgCapturePlan::none(*final_sdfg);
    codegen::AssignmentDispatcher
        dispatcher(language_extension, *final_sdfg, analysis_manager, block, *instrumentation, *arg_capture);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_stream;
    codegen::CodeSnippetFactory library_factory;
    dispatcher.dispatch_node(main_stream, globals_stream, library_factory);

    EXPECT_EQ(main_stream.str(), "{\n    i = convert(10);\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchTasklet_Identifiers) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar desc(types::PrimitiveType::Int32);
    builder.add_container("a", desc);
    builder.add_container("b", desc);

    auto& block = builder.add_block(root);
    auto& in = builder.add_access(block, "a");
    auto& out = builder.add_access(block, "b");
    auto& tasklet = builder.add_tasklet(block, data_flow::TaskletCode::assign, "_out", {"_in"});
    builder.add_computational_memlet(block, in, tasklet, "_in", {});
    builder.add_computational_memlet(block, tasklet, "_out", out, {});

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    HookedLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    int v_in = a;\n    int v_out;\n\n    v_out = v_in;\n\n    b = v_out;\n}\n");
}

TEST(DataFlowDispatcherTest, DispatchRef_ElementAddress) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& root = builder.subject().root();

    types::Scalar base_type(types::PrimitiveType::Int32);
    types::Pointer pointer_type(base_type);
    builder.add_container("a", pointer_type);
    builder.add_container("b", pointer_type);

    auto& block = builder.add_block(root);
    auto& access_node_1 = builder.add_access(block, "a");
    auto& access_node_2 = builder.add_access(block, "b");
    builder.add_reference_memlet(block, access_node_1, access_node_2, {symbolic::integer(1)}, pointer_type);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    HookedLanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "{\n    b = address(((int *) a), 1);\n}\n");
}
