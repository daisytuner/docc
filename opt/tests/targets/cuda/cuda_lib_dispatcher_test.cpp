#include "sdfg/codegen/dispatchers/block_dispatcher.h"

#include <gtest/gtest.h>

#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/language_extensions/c_language_extension.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/atomic_op_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/targets/cuda/codegen/cuda_language_extension.h"

using namespace sdfg;

TEST(CudaLibraryNodeDispatcherTest, DispatchBarrierLocalNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    auto& block = builder.add_block(root);
    builder.add_library_node<sdfg::data_flow::BarrierLocalNode>(block, DebugInfo());

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    cuda::CUDALanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    EXPECT_EQ(main_stream.str(), "__syncthreads();\n");
}

TEST(CudaLibraryNodeDispatcherTest, DispatchAtomicAccumulateNode) {
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar base_type(types::PrimitiveType::Float);
    types::Pointer ptr(base_type);
    builder.add_container("dst", ptr); // pre-offset pointer to a global slot
    builder.add_container("src", base_type); // scalar value to accumulate

    auto& block = builder.add_block(root);
    auto& dst = builder.add_access(block, "dst");
    auto& src = builder.add_access(block, "src");
    auto& node = builder.add_library_node<sdfg::data_flow::AtomicScalarOpNode>(
        block,
        DebugInfo(),
        types::PrimitiveType::Float,
        data_flow::AtomicOpType::Add,
        sdfg::data_flow::AtomicScalarOpCudaImpl::instance()
    );
    // No subset on either edge: dst is already offset, src is a scalar.
    builder.add_computational_memlet(block, dst, node, "_dst", {}, ptr);
    builder.add_computational_memlet(block, src, node, "_src", {}, base_type);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    cuda::CUDALanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    auto out = main_stream.str();
    EXPECT_NE(out.find("atomicAdd("), std::string::npos);
    EXPECT_NE(out.find("dst"), std::string::npos);
    EXPECT_NE(out.find("src"), std::string::npos);
    EXPECT_EQ(out.find("[i]"), std::string::npos); // no subset addressing
}

TEST(CudaLibraryNodeDispatcherTest, DISABLED_DispatchAtomicAccumulateNode_NestedArray) { // the Node is only for scalar.
                                                                                         // For
    // now LocalStorage handles the
    // iterating. In the future we plan to
    // have a eltwise version of this
    // again!
    builder::StructuredSDFGBuilder builder("sdfg_a", FunctionType_CPU);
    auto& sdfg = builder.subject();
    auto& root = sdfg.root();

    types::Scalar f(types::PrimitiveType::Float);
    // _src: a CY x CX register tile (nested array).
    types::Array tile(types::Array(f, symbolic::integer(3)), symbolic::integer(2));
    // _dst: pointer to array-of-N (the global row stride), pre-offset to the tile base.
    types::Pointer dst_ptr(types::Array(f, symbolic::integer(8)));
    builder.add_container("dst", dst_ptr);
    builder.add_container("src", tile);

    auto& block = builder.add_block(root);
    auto& dst = builder.add_access(block, "dst");
    auto& src = builder.add_access(block, "src");
    auto& node = builder.add_library_node<sdfg::data_flow::AtomicScalarOpNode>(
        block,
        DebugInfo(),
        f.primitive_type(),
        data_flow::AtomicOpType::Add,
        sdfg::data_flow::AtomicScalarOpCudaImpl::instance()
    );
    builder.add_computational_memlet(block, dst, node, "_dst", {}, dst_ptr);
    builder.add_computational_memlet(block, src, node, "_src", {}, tile);

    auto final_sdfg = builder.move();
    analysis::AnalysisManager analysis_manager(*final_sdfg);

    cuda::CUDALanguageExtension language_extension(*final_sdfg);
    auto instrumentation = codegen::InstrumentationPlan::none(*final_sdfg);
    codegen::DataFlowDispatcher dispatcher(language_extension, *final_sdfg, block.dataflow(), *instrumentation);

    codegen::PrettyPrinter main_stream;
    codegen::PrettyPrinter globals_printer;
    codegen::CodeSnippetFactory snippet_factory;
    dispatcher.dispatch(main_stream, globals_printer, snippet_factory);

    auto out = main_stream.str();
    // Two nested loops over the tile extents, atomicAdd of each element.
    EXPECT_NE(out.find("for (int __daisy_aa_"), std::string::npos);
    EXPECT_NE(out.find("< 2;"), std::string::npos);
    EXPECT_NE(out.find("< 3;"), std::string::npos);
    EXPECT_NE(out.find("atomicAdd(&("), std::string::npos);
}
