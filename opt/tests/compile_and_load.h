#pragma once

#include <dlfcn.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "sdfg/analysis/analysis.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/code_generators/cpp_code_generator.h"
#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/instrumentation/arg_capture_plan.h"
#include "sdfg/codegen/instrumentation/instrumentation_plan.h"
#include "sdfg/passes/dataflow/tensor_to_pointer_conversion.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"

namespace sdfg::test {

/**
 * Expands the library nodes of @p builder's SDFG, linearizes its tensor accesses, generates C++ for it, compiles it
 * into a shared library and loads it; returns the SDFG's function. The library stays loaded for the rest of the
 * process.
 */
inline void* compile_and_load(builder::StructuredSDFGBuilder& builder) {
    auto& sdfg = builder.subject();
    {
        analysis::AnalysisManager analysis_manager(sdfg);
        passes::LibraryNodeExpansionPass<> expansion;
        expansion.run(builder, analysis_manager);
        // As in the compile pipeline: code generation indexes pointers, not tensors.
        passes::TensorToPointerConversionPass tensor_to_pointer;
        tensor_to_pointer.run(builder, analysis_manager);
    }
    sdfg.validate();

    analysis::AnalysisManager analysis_manager(sdfg);
    auto instrumentation_plan = codegen::InstrumentationPlan::none(sdfg);
    auto arg_capture_plan = codegen::ArgCapturePlan::none(sdfg);
    auto snippets = std::make_shared<codegen::CodeSnippetFactory>();
    codegen::CPPCodeGenerator generator(sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan, snippets);
    generator.generate();

    auto folder = std::filesystem::temp_directory_path() / ("docc_test_" + std::to_string(::getpid()));
    std::filesystem::create_directories(folder);
    auto source = folder / (sdfg.name() + ".cpp");
    // dlopen returns an already loaded library for the same path, so every build gets its own.
    static int builds = 0;
    auto library = folder / ("lib" + sdfg.name() + "_" + std::to_string(builds++) + ".so");
    {
        std::ofstream out(source);
        out << generator.includes().str() << std::endl;
        out << generator.classes().str() << std::endl;
        out << generator.globals().str() << std::endl;
        out << generator.function_definition() << std::endl << "{" << std::endl;
        out << generator.main().str() << std::endl << "}" << std::endl;
    }
    std::string command = std::string(DOCC_TEST_CXX_COMPILER) + " -std=c++17 -O1 -shared -fPIC -w -I" +
                          DOCC_TEST_RTL_INCLUDE + " " + source.string() + " -o " + library.string() + " 2>&1";
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("Compiling " + source.string() + " failed: " + command);
    }
    void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        throw std::runtime_error(std::string("Loading failed: ") + dlerror());
    }
    void* function = dlsym(handle, sdfg.name().c_str());
    if (!function) {
        throw std::runtime_error("Symbol " + sdfg.name() + " not found");
    }
    return function;
}

} // namespace sdfg::test
