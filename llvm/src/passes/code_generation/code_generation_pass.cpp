#include "docc/passes/code_generation/code_generation_pass.h"

#include <docc/target/tenstorrent/plugin.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/codegen/code_generator.h>
#include <sdfg/codegen/code_generators/c_code_generator.h>
#include <sdfg/codegen/code_generators/cpp_code_generator.h>
#include <sdfg/codegen/code_generators/cuda_code_generator.h>
#include <sdfg/data_flow/library_nodes/call_node.h>
#include <sdfg/data_flow/library_nodes/stdlib/stdlib.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/targets/cuda/cuda.h>
#include <sdfg/targets/offloading/utils.h>
#include <sdfg/util/offloading_instrumentation_plan.h>
#include <sdfg/util/offloading_type_collector.h>
#include <sdfg/visualizer/dot_visualizer.h>

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "docc/analysis/attributes.h"
#include "docc/docc_paths.h"
#include "docc/target/tenstorrent/tenstorrent_offloading_node.h"
#include "docc/utils.h"
#include "sdfg/builder/structured_sdfg_builder.h"
#include "sdfg/codegen/loop_report.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/data_flow_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/stdlib/alloca.h"
#include "sdfg/data_flow/library_nodes/stdlib/calloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/free.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/data_flow/library_nodes/stdlib/memcpy.h"
#include "sdfg/data_flow/library_nodes/stdlib/memmove.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/gpu/gpu_schedule_type.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/offloading/external_offloading_node.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/types/function.h"
#include "sdfg/types/pointer.h"
#include "sdfg/types/scalar.h"

namespace docc {
namespace passes {

llvm::cl::opt<bool> codegen_capture_args_results(
    "docc-capture-args",
    llvm::cl::desc("Instruments all DOCC-lifted code to capture arguments and results"),
    llvm::cl::init(false)
);

llvm::cl::opt<std::string> codegen_instrumentation_mode(
    "docc-instrument", llvm::cl::desc("Determines the instrumentation mode"), llvm::cl::init("")
);

/// Ideally, this would be a facility in the link-time passes and we take a string to indicate after
/// which passes we dump, with an option to dump after every pass. But this would require output
/// path handling outside of codegen
llvm::cl::opt<bool> codegen_dot_output(
    "docc-dot-codegen",
    llvm::cl::desc("Write out every SDFG for which we generate code to a DOT file for debugging purposes"),
    llvm::cl::init(false)
);

CodeGenerationPass::CodeGenerationPass() { add_subcomp_opts_from_arg(sub_compile_opts_); }

void CodeGenerationPass::add_subcomp_opts_from_arg(std::vector<std::string>& out_opts) const {
    for (const auto& opt : DOCC_COMP_OPTS) {
        if (!opt.empty()) {
            out_opts.push_back(opt);
        }
    }
}

bool wants_to_replay(const std::string& F) {
    return (
        /* ABI or machine code */
        F == "-fPIC" || F == "-fpic" || F == "-fno-pic" || F.starts_with("-ffast-math") ||
        F.starts_with("-funroll-loops") || F.starts_with("-fopenmp") || F.starts_with("-O") ||
        F.starts_with("-fsanitize=") || F.starts_with("-march") || F.starts_with("-mcpu=") ||
        F.starts_with("-mtune=") || F.starts_with("-mabi=") || F.starts_with("-mfloat-abi=") || F.starts_with("-m64") ||
        F.starts_with("-m32") || F.starts_with("-DTRACY_ENABLE")
    );
}

void add_schedule_type_specific_linker_args(const ScheduleType& schedule_type, std::set<std::string>& linker_args) {
    if (schedule_type.value() == sdfg::cuda::ScheduleType_CUDA::value()) {
        linker_args.emplace("/usr/local/cuda/lib64/libcudart.so");
        linker_args.emplace("/usr/local/cuda/lib64/libcublas.so");

    } else if (schedule_type.value() == sdfg::rocm::ScheduleType_ROCM::value()) {
        linker_args.emplace("/opt/rocm/lib/libamdhip64.so");
        linker_args.emplace("/opt/rocm/lib/libhiprtc.so");
        linker_args.emplace("/opt/rocm/lib/libhipblas.so");
    } else if (sdfg::tenstorrent::is_tenstorrent_schedule(schedule_type)) {
        if (auto tt_path = std::getenv("TT_METAL_INSTALL_DIR")) {
            linker_args.emplace("-L" + std::string(tt_path) + "/lib");
        } else if (auto tt_path = std::getenv("TT_METAL_HOME")) {
            linker_args.emplace("-L" + std::string(tt_path) + "/build/lib");
        } else {
            throw std::runtime_error("Neither TT_METAL_INSTALL_DIR nor TT_METAL_HOME environment variable are set");
        }
        linker_args.emplace("-ltt_metal");
        linker_args.emplace("-lstdc++");
    } else {
        // no 2nd link stage needed
    }
}

void add_schedule_type_specific_compile_args(
    const ScheduleType& schedule_type,
    llvm::dwarf::SourceLanguage& language,
    std::vector<std::string>& compile_args,
    const utils::DoccPaths& docc_paths
) {
    if (schedule_type.value() == sdfg::rocm::ScheduleType_ROCM::value()) {
        compile_args.emplace_back("-x hip");
        compile_args.emplace_back("--offload-arch=gfx1201");
        compile_args.emplace_back("--offload-host-only");
        compile_args.emplace_back("--rocm-path=/opt/rocm");
        compile_args.emplace_back("-I/opt/rocm/include");
        language = llvm::dwarf::DW_LANG_C_plus_plus;
    } else if (schedule_type.value() == sdfg::cuda::ScheduleType_CUDA::value()) {
        compile_args.emplace_back("-x cuda");
        compile_args.emplace_back("--cuda-gpu-arch=sm_70");
        compile_args.emplace_back("--cuda-host-only");
        language = llvm::dwarf::DW_LANG_C_plus_plus;
    } else if (sdfg::tenstorrent::is_tenstorrent_schedule(schedule_type)) {
        compile_args.emplace_back("-x c++");

        std::string tt_path;
        if (auto tt_i_path = std::getenv("TT_METAL_INSTALL_DIR")) {
            tt_path = std::string(tt_i_path);
            auto boost_base_path = tt_path + "/boost-src";
            compile_args.emplace_back("-I" + boost_base_path + "/libs/container/include");
            compile_args.emplace_back("-I" + boost_base_path + "/libs/assert/include");
            compile_args.emplace_back("-I" + boost_base_path + "/libs/config/include");
            compile_args.emplace_back("-I" + boost_base_path + "/libs/intrusive/include");
            compile_args.emplace_back("-I" + boost_base_path + "/libs/move/include");
            compile_args.emplace_back("-I" + boost_base_path + "/libs/core/include");
            compile_args.emplace_back("-I" + boost_base_path + "/libs/static_assert/include");
            compile_args.emplace_back("-I" + boost_base_path + "/libs/throw_exception/include");
        } else if (auto tt_m_path = std::getenv("TT_METAL_HOME")) {
            tt_path = std::string(tt_m_path) + "/build";
            // expect new enough boost installed
        } else {
            throw std::runtime_error("Neither TT_METAL_INSTALL_DIR nor TT_METAL_HOME environment variable are set");
        }
        compile_args.emplace_back("-I" + tt_path + "/include");
        compile_args.emplace_back("-I" + tt_path + "/include/metalium-thirdparty");
        compile_args.emplace_back("-DSPDLOG_FMT_EXTERNAL");
        compile_args.emplace_back("-std=c++20");
        language = llvm::dwarf::DW_LANG_C_plus_plus;
    } else {
        if (language == llvm::dwarf::DW_LANG_C) {
            compile_args.emplace_back("-x c");
        } else if (llvm::dwarf::isCPlusPlus(language)) {
            compile_args.emplace_back("-x c++");
        } else {
            throw std::runtime_error("Unsupported source language: " + std::to_string(language));
        }
    }
    // add our own include dirs (for example arg_capture & daisy_rtl) to the search path
    for (const auto& inc : docc_paths.target_inc_paths()) {
        compile_args.push_back("-I" + inc.string());
    }
}

bool CodeGenerationPass::compile_additional_files(
    const std::filesystem::path& build_path,
    const std::vector<std::string>& compile_args,
    std::set<std::string>& link_2nd_args,
    std::unordered_map<std::string, std::vector<std::filesystem::path>> files_for_post_processing,
    const utils::DoccPaths& docc_paths
) {
    bool success = true;

    auto it = files_for_post_processing.find("cu");
    if (it != files_for_post_processing.end()) {
        std::vector<std::string> comp_args = compile_args;
        for (const auto& inc : docc_paths.target_inc_paths()) {
            comp_args.push_back("-I" + inc.string());
        }
        auto& cu_files = it->second;
        for (auto& cu_file : cu_files) {
            auto fn = cu_file.filename().string();
            auto obj_file = build_path / (fn.substr(0, fn.size() - 2) + "o");
            auto comp_success = compile_to_object_file(cu_file, obj_file, comp_args);
            if (!comp_success) {
                LLVM_DEBUG_PRINTLN("Compilation of additional file " << cu_file.string() << " failed");
            } else {
                link_2nd_args.emplace(obj_file.string());
            }
            success &= comp_success;
        }
    }

    auto it_rocm = files_for_post_processing.find("rocm.cpp");
    if (it_rocm != files_for_post_processing.end()) {
        std::vector<std::string> comp_args = compile_args;
        comp_args.push_back("-x");
        comp_args.push_back("hip");
        comp_args.push_back("--offload-arch=gfx1201");
        comp_args.push_back("--rocm-path=/opt/rocm");
        comp_args.push_back("-I/opt/rocm/include");
        for (const auto& inc : docc_paths.target_inc_paths()) {
            comp_args.push_back("-I" + inc.string());
        }
        auto& rocm_files = it_rocm->second;
        for (auto& rocm_file : rocm_files) {
            auto fn = rocm_file.filename().string();
            auto obj_file = build_path / (fn.substr(0, fn.size() - 7) + "o");
            auto comp_success = compile_to_object_file(rocm_file, obj_file, comp_args);
            if (!comp_success) {
                LLVM_DEBUG_PRINTLN("Compilation of additional file " << rocm_file.string() << " failed");
            } else {
                link_2nd_args.emplace(obj_file.string());
            }
            success &= comp_success;
        }
    }

    return success;
}
bool CodeGenerationPass::generate_code(
    llvm::Module& Mod,
    sdfg::StructuredSDFG& sdfg,
    const analysis::Attributes& attributes,
    llvm::dwarf::SourceLanguage& language,
    const std::vector<std::string>& mod_wide_compile_flags,
    std::set<std::string>& link_2nd_args,
    utils::DoccPaths docc_paths
) {
    LLVM_DEBUG_PRINTLN("Generating code for SDFG: " << sdfg.name());
    bool success = true;

    std::filesystem::path sdfg_path = sdfg.metadata("sdfg_file");
    std::filesystem::path build_path = sdfg_path.parent_path();

    // Dump DOT if requested
    if (codegen_dot_output) {
        std::filesystem::path dot_path = sdfg_path;
        dot_path.replace_extension(".dot");
        sdfg::visualizer::DotVisualizer::writeToFile(sdfg, &dot_path);
    }

    // Determine additional compile args for accelerators
    std::vector<std::string> add_compile_args;
    {
        auto sdfgMeta = sdfg.metadata();
        auto it = sdfgMeta.find("compile_opts");
        if (it != sdfgMeta.end()) {
            add_compile_args.push_back(it->second); // may contain multiple, space-separated args
        }
    }
    llvm::dwarf::SourceLanguage actual_language = language;
    {
        ScheduleType special_schedule_type = ScheduleType_Sequential::create();
        sdfg::builder::StructuredSDFGBuilder builder(sdfg);
        sdfg::analysis::AnalysisManager analysis_manager(builder.subject());
        sdfg::auto_util::OffloadingTypeCollector v(builder, analysis_manager, special_schedule_type);
        v.visit();

        special_schedule_type = v.schedule_type();

        // Only use container-based detection as a fallback if the
        // OffloadingTypeCollector did not find a GPU schedule type.
        if (!sdfg::gpu::is_gpu_schedule(special_schedule_type)) {
            for (auto& container : builder.subject().containers()) {
                if (builder.subject().type(container).storage_type().value() == "NV_Generic") {
                    special_schedule_type = sdfg::cuda::ScheduleType_CUDA::create();
                    break;
                }
                if (builder.subject().type(container).storage_type().value() == "AMD_Generic") {
                    special_schedule_type = sdfg::rocm::ScheduleType_ROCM::create();
                    break;
                }
            }
        }

        add_schedule_type_specific_linker_args(special_schedule_type, link_2nd_args);
        add_schedule_type_specific_compile_args(special_schedule_type, actual_language, add_compile_args, docc_paths);
        language = actual_language;

        if (!builder.subject().exists("free")) {
            sdfg::types::Function free_func(sdfg::types::Scalar(sdfg::types::PrimitiveType::Void), false);
            free_func.add_param(sdfg::types::Pointer());
            builder.add_container("free", free_func, false, true);
        }
        if (!builder.subject().exists("malloc")) {
            sdfg::types::Function malloc_func(sdfg::types::Pointer(), false);
            malloc_func.add_param(sdfg::types::Scalar(sdfg::types::PrimitiveType::UInt64));
            builder.add_container("malloc", malloc_func, false, true);
        }
        if (!builder.subject().exists("exit")) {
            sdfg::types::Function exit_func(sdfg::types::Scalar(sdfg::types::PrimitiveType::Void), false);
            exit_func.add_param(sdfg::types::Scalar(sdfg::types::PrimitiveType::UInt64));
            builder.add_container("exit", exit_func, false, true);
        }
    }

    // Prepare source files
    std::filesystem::path header_path = sdfg_path;
    header_path.replace_extension(".h");

    std::filesystem::path source_path = sdfg_path;
    if (actual_language == llvm::dwarf::DW_LANG_C) {
        source_path.replace_extension(".c");
    } else {
        source_path.replace_extension(".cpp");
    }

    std::pair<std::filesystem::path, std::filesystem::path> lib_config = std::make_pair(build_path, header_path);
    std::shared_ptr<sdfg::codegen::CodeSnippetFactory> snippet_factory =
        std::make_shared<sdfg::codegen::CodeSnippetFactory>(&lib_config);

    auto sdfgs = this->split_sdfg(sdfg, attributes);
    size_t sdfg_index = 0;
    for (auto& part_sdfg : sdfgs) {
        part_sdfg->add_metadata("source_file", source_path);
        part_sdfg->add_metadata("header_file", header_path);

        // Generate code
        sdfg::analysis::AnalysisManager analysis_manager(*part_sdfg);

        // Determine instrumentation plan
        std::unique_ptr<sdfg::codegen::InstrumentationPlan> instrumentation_plan;
        if (codegen_instrumentation_mode.empty()) {
            instrumentation_plan = sdfg::codegen::InstrumentationPlan::none(*part_sdfg);
        } else if (codegen_instrumentation_mode == "ols") {
            instrumentation_plan = sdfg::codegen::InstrumentationPlan::outermost_loops_plan(*part_sdfg);
            sdfg::auto_util::add_offloading_instrumentations(*instrumentation_plan, *part_sdfg);
        } else {
            throw std::runtime_error("Unsupported instrumentation plan: " + codegen_instrumentation_mode);
        }
        std::unique_ptr<sdfg::codegen::ArgCapturePlan> arg_capture_plan;
        if (codegen_capture_args_results) {
            arg_capture_plan = sdfg::codegen::ArgCapturePlan::outermost_loops_plan(*part_sdfg);
        } else {
            arg_capture_plan = sdfg::codegen::ArgCapturePlan::none(*part_sdfg);
        }

        // Create code generator
        std::unique_ptr<sdfg::codegen::CodeGenerator> code_generator;
        if (actual_language == llvm::dwarf::DW_LANG_C) {
            code_generator = std::make_unique<sdfg::codegen::CCodeGenerator>(
                *part_sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan, snippet_factory, "__daisy_ext_"
            );
        } else {
            code_generator = std::make_unique<sdfg::codegen::CPPCodeGenerator>(
                *part_sdfg, analysis_manager, *instrumentation_plan, *arg_capture_plan, snippet_factory, "__daisy_ext_"
            );
        }
        success &= code_generator->generate();
        if (!success) {
            throw std::runtime_error("Code generation failed");
        }

        // Dump generated code to files
        if (sdfg_index == 0) {
            success &= code_generator->as_source(header_path, source_path);
        } else {
            std::ofstream ofs_source(source_path, std::ofstream::out | std::ofstream::app);
            ofs_source << std::endl;
            ofs_source << code_generator->globals().str() << std::endl;
            code_generator->append_function_source(ofs_source);
            ofs_source.close();
        }

        // Write library snippets to files
        std::unordered_set<std::string> lib_files;
        std::unordered_map<std::string, std::vector<std::filesystem::path>> files_for_post_processing;
        success &= cuda_offloading_codegen::write_library_snippets_to_files(
            build_path, lib_files, snippet_factory->snippets(), files_for_post_processing, "cu"
        );
        success &= cuda_offloading_codegen::write_library_snippets_to_files(
            build_path, lib_files, snippet_factory->snippets(), files_for_post_processing, "rocm.cpp"
        );
        success &= compile_additional_files(
            build_path, mod_wide_compile_flags, link_2nd_args, files_for_post_processing, docc_paths
        );

        sdfg_index++;
    }

    // Link into original module
    link_into_existing_module(Mod, mod_wide_compile_flags, add_compile_args, source_path, sdfg.name());

    // Set attributes
    auto function = Mod.getFunction(sdfg.name());
    function->removeFnAttr(llvm::Attribute::NoInline);
    function->removeFnAttr(llvm::Attribute::OptimizeNone);

    return success;
}

llvm::StringRef getEnvVar(const char* Name) {
    const char* Val = ::getenv(Name);
    return Val ? llvm::StringRef(Val) : llvm::StringRef();
}

bool CodeGenerationPass::link_into_existing_module(
    llvm::Module& Mod,
    const std::vector<std::string>& compile_args,
    const std::vector<std::string>& add_compiler_args,
    const std::filesystem::path& source_path,
    const std::string& steal_from
) {
    // Step 1: Compile source file into new module
    auto subM =
        compile_to_ir_in_memory(Mod.getContext(), source_path, compile_args, add_compiler_args, Mod.getTargetTriple());
    if (!subM) {
        LLVM_DEBUG_PRINTLN("Failed to compile " << source_path << " to IR");
        exit(EXIT_FAILURE);
    }

    // Step 2: Prepare all global symbols in new module for linking
    // Strip '__daisy_ext_' prefix from all functions in the new module
    for (llvm::Function& F : *subM) {
        std::string name = F.getName().str();
        if (name.starts_with("__daisy_ext_")) {
            std::string new_name = name.substr(strlen("__daisy_ext_"));
            auto* already_existing = subM->getFunction(new_name);
            if (already_existing) {
                F.replaceAllUsesWith(already_existing);
            } else {
                F.setName(new_name);
            }
        }
    }
    // Strip '__daisy_ext_' prefix from all global variables in the new module
    for (llvm::GlobalVariable& G : subM->globals()) {
        std::string name = G.getName().str();
        if (name.starts_with("__daisy_ext_")) {
            std::string new_name = name.substr(strlen("__daisy_ext_"));
            G.setName(new_name);
        }
    }

    // Step 3: Prepare symbols for linking: Handle collisions by making both sides external
    std::unordered_map<std::string, llvm::GlobalValue::LinkageTypes> global_collisions;
    for (llvm::GlobalVariable& G : subM->globals()) {
        llvm::GlobalVariable* orig_global = Mod.getGlobalVariable(G.getName());
        if (orig_global) {
            global_collisions[G.getName().str()] = orig_global->getLinkage();

            // Apply attributes from original global
            G.setAlignment(orig_global->getAlign());
            G.setVisibility(orig_global->getVisibility());
            G.setDSOLocal(orig_global->isDSOLocal());
            G.setThreadLocalMode(orig_global->getThreadLocalMode());
            G.setDLLStorageClass(orig_global->getDLLStorageClass());
            G.setUnnamedAddr(orig_global->getUnnamedAddr());
            if (orig_global->hasSection()) {
                G.setSection(orig_global->getSection());
            }

            // Make both globals external for linking
            G.setLinkage(llvm::GlobalValue::ExternalLinkage);
            orig_global->setLinkage(llvm::GlobalValue::ExternalLinkage);
        }
    }
    // Handle function collisions: Add attributes from original functions
    std::unordered_map<std::string, llvm::GlobalValue::LinkageTypes> function_collisions;
    for (auto& F : subM->functions()) {
        llvm::Function* orig_function = Mod.getFunction(F.getName());
        if (orig_function) {
            function_collisions[F.getName().str()] = orig_function->getLinkage();

            // Apply attributes from original function
            F.setCallingConv(orig_function->getCallingConv());
            F.setAttributes(orig_function->getAttributes());
            F.setAlignment(orig_function->getAlign());
            F.setVisibility(orig_function->getVisibility());
            F.setDSOLocal(orig_function->isDSOLocal());
            F.setDLLStorageClass(orig_function->getDLLStorageClass());
            F.setUnnamedAddr(orig_function->getUnnamedAddr());
            if (orig_function->hasSection()) {
                F.setSection(orig_function->getSection());
            }

            for (auto* U : F.users()) {
                if (auto* CB = llvm::dyn_cast<llvm::CallBase>(U)) {
                    if (CB->getCalledFunction() == &F) {
                        CB->setAttributes(orig_function->getAttributes());
                        CB->setCallingConv(orig_function->getCallingConv());
                    }
                }
            }

            F.setLinkage(llvm::GlobalValue::ExternalLinkage);
            orig_function->setLinkage(llvm::GlobalValue::ExternalLinkage);
        }
    }

    // Step 4: Link modules
    llvm::Linker L(Mod);
    if (L.linkInModule(std::move(subM), llvm::Linker::OverrideFromSrc)) {
        throw std::
            runtime_error("Linking module from " + source_path.string() + " into " + Mod.getName().str() + " failed");
    }

    // Step 5: Restore original linkage of collided symbols
    for (llvm::GlobalVariable& G : Mod.globals()) {
        if (global_collisions.find(G.getName().str()) != global_collisions.end()) {
            G.setLinkage(global_collisions[G.getName().str()]);
        }
    }
    for (auto& func : Mod.functions()) {
        if (function_collisions.find(func.getName().str()) != function_collisions.end()) {
            func.setLinkage(function_collisions[func.getName().str()]);
        }
    }

    return true;
}

llvm::PreservedAnalyses CodeGenerationPass::
    run(llvm::Module& Mod, llvm::ModuleAnalysisManager& MAM, analysis::AnalysisManager& AM) {
    auto& registry = AM.get<analysis::SDFGRegistry>();
    if (!registry.has_module(Mod)) {
        return llvm::PreservedAnalyses::all();
    }

    llvm::dwarf::SourceLanguage language = utils::source_language(Mod);
    auto triple = Mod.getTargetTriple();
    auto datalayout = Mod.getDataLayoutStr();

    bool opt_report = false;
    auto opt_report_str = getEnvVar("DOCC_OPT_REPORT");
    if (opt_report_str == "1" || opt_report_str == "true" || opt_report_str == "on") {
        opt_report = true;
    }

    auto& sdfgs = registry.at(Mod);
    if (sdfgs.empty()) {
        return llvm::PreservedAnalyses::all();
    }

    if (opt_report) {
        // Collect report
        std::unordered_map<std::string, size_t> cumulated_report;
        docc::analysis::SDFGRegistry::
            for_each_sdfg_modifiable(sdfgs, [&](analysis::SDFGHolder&, sdfg::StructuredSDFG& sdfg) {
                sdfg::builder::StructuredSDFGBuilder builder(sdfg);
                sdfg::analysis::AnalysisManager analysis_manager(builder.subject());

                sdfg::codegen::LoopReport generator(builder, analysis_manager);
                generator.visit();

                for (auto& [key, value] : generator.report()) {
                    if (cumulated_report.find(key) == cumulated_report.end()) {
                        cumulated_report[key] = 0;
                    }
                    cumulated_report[key] += value;
                }
            });

        // Dump report
        std::stringstream opt_report_stream;
        opt_report_stream << "\nDOCC Optimization Report Start:\n";
        opt_report_stream << "  source_language: " << (language == llvm::dwarf::DW_LANG_C ? "C" : "C++") << "\n";
        opt_report_stream << "  target_triple: " << triple << "\n";
        opt_report_stream << "  data_layout: " << datalayout << "\n";
        opt_report_stream << "  sdfgs: " << sdfgs.size() << "\n";
        for (auto& [key, value] : cumulated_report) {
            opt_report_stream << "  " << key << ": " << value << "\n";
        }
        opt_report_stream << "DOCC Optimization Report End\n";

#ifndef NDEBUG
        llvm::errs() << opt_report_stream.str();
#endif
    }

    utils::DoccPaths docc_paths = utils::DoccPaths::get_instance();

    auto original_flags = utils::get_recorded_driver_flags(Mod);
    // TODO broken. captures many commandlines from many original modules (llvm.commandline is merged and thinLTO seems
    // to already be fusing other modules metadata into ours (but also not all of them, it varies per module)

    // Determine compile flags
    std::vector<std::string> subcomp_args;
    std::unordered_set<std::string> seen;
    for (const auto& flag : original_flags) {
        if (seen.find(flag) != seen.end()) { // TODO broken. we split the command lines based on ' ' and it contains for
                                             // example "-D NDEBUG"
            continue;
        }
        seen.insert(flag);

        if (wants_to_replay(flag)) {
            subcomp_args.push_back(flag);
        }
    }
    subcomp_args.insert(subcomp_args.end(), this->sub_compile_opts_.begin(), this->sub_compile_opts_.end());

    std::set<std::string> link_2nd_args = {"-L/usr/lib/llvm-19/lib/", "-lomp"};
    bool success = true;
    docc::analysis::SDFGRegistry::for_each_sdfg_modifiable(sdfgs, [&](analysis::SDFGHolder&, sdfg::StructuredSDFG& sdfg) {
        auto& attributes = registry.attributes(sdfg.name());
        success &= generate_code(Mod, sdfg, attributes, language, subcomp_args, link_2nd_args, docc_paths);
    });

    if (!success) {
        throw std::runtime_error("Code generation/compilation failed");
    }

    if (codegen_capture_args_results || !codegen_instrumentation_mode.empty()) {
        link_2nd_args.emplace("-ldaisy_rtl");
        link_2nd_args.emplace("-larg_capture_io");
        link_2nd_args.emplace("-lstdc++");
    }

    if (!link_2nd_args.empty()) {
        auto [lock, sdfg] = sdfgs.begin()->second->get_for_read();
        std::filesystem::path build_path = sdfg->metadata("sdfg_file");
        build_path = build_path.parent_path();
        std::ofstream link_opts(build_path / "LINK_OPTS", std::ofstream::trunc);
        if (!link_opts.is_open()) {
            throw std::runtime_error("Failed to open link options file");
        }
        for (const auto& arg : link_2nd_args) {
            link_opts << arg << "\n";
        }
        link_opts.close();
    }

#ifndef NDEBUG
    bool verify_dbg = false;
    bool verified = llvm::verifyModule(Mod, &llvm::errs(), &verify_dbg);
    if (verified) {
        // Real IR errors, not just debug info
        LLVM_DEBUG_PRINTLN("Module " << Mod.getName() << " is broken after codegen.");
        throw sdfg::InvalidSDFGException("Module is broken after codegen sdfg.");
    }
#endif

    // Inline newly linked functions so they are optimized within the current module
    {
        llvm::PassBuilder PB;
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;

        PB.registerModuleAnalyses(MAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
        llvm::ModulePassManager MPM;
        MPM.addPass(PB.buildInlinerPipeline(llvm::OptimizationLevel::O2, llvm::ThinOrFullLTOPhase::None));
        MPM.addPass(llvm::GlobalDCEPass());
        MPM.run(Mod, MAM);
    }

    return llvm::PreservedAnalyses::none();
}

bool CodeGenerationPass::compile_to_object_file(
    const std::filesystem::path& source_path,
    const std::filesystem::path& object_file,
    const std::vector<std::string>& compile_args
) {
    std::vector<std::string> cmd{"clang++-19"};
    for (auto& arg : compile_args) {
        cmd.push_back(arg);
    }
    cmd.push_back("-fopenmp");
    cmd.push_back("-Wno-parentheses-equality");
    cmd.push_back("-c");
    cmd.push_back(source_path.string());
    cmd.push_back("-o");
    cmd.push_back(object_file.string());

    auto compile_cmd = sdfg::helpers::join(cmd, " ");

    int ret = std::system(compile_cmd.c_str());

    if (ret != 0) {
        LLVM_DEBUG_PRINTLN("Failed to compile additional source: " << "'" << compile_cmd << "'");
        exit(EXIT_FAILURE);
    }

    return true;
}

std::unique_ptr<llvm::Module> CodeGenerationPass::compile_to_ir_in_memory(
    llvm::LLVMContext& ctx,
    const std::filesystem::path& source_path,
    const std::vector<std::string>& compile_args,
    const std::vector<std::string>& add_compile_args,
    llvm::StringRef target_triple
) {
    std::ostringstream cmd;
    cmd << "clang-19 ";
    for (auto& arg : add_compile_args) {
        cmd << arg << " ";
    }

    cmd << " -Wno-parentheses-equality -emit-llvm -S -c"
        << " -target " << target_triple.str() << " ";
    for (const auto& flag : compile_args) {
        cmd << flag << " ";
    }

    cmd << "-fopenmp ";


    cmd << source_path.string() << " -o -";

    auto final_cmd = cmd.str();
    LLVM_DEBUG_PRINTLN("Subcompile: " << final_cmd);

    // Use unique_ptr with a custom deleter for pclose
    FILE* pipe = popen(final_cmd.c_str(), "r");
    if (!pipe) {
        exit(EXIT_FAILURE);
    }

    std::unique_ptr<FILE, decltype(&pclose)> pipe_ptr(pipe, pclose);

    std::string ir_text;
    char buffer[4096];
    while (size_t read = fread(buffer, 1, sizeof(buffer), pipe_ptr.get())) {
        ir_text.append(buffer, read);
    }

    int retcode = pclose(pipe_ptr.release());
    if (retcode != 0) {
        return nullptr;
    }

    auto mem_buf = llvm::MemoryBuffer::getMemBuffer(ir_text);
    llvm::SMDiagnostic err;
    auto submod = llvm::parseIR(*mem_buf, err, ctx);
    if (!submod) {
        err.print("compile_to_ir_in_memory", llvm::errs());
        exit(EXIT_FAILURE);
    }

    return submod;
}

std::list<std::unique_ptr<sdfg::StructuredSDFG>> CodeGenerationPass::
    split_sdfg(sdfg::StructuredSDFG& sdfg, const analysis::Attributes& attributes) {
    // No splitting when all buffers are empty or a copy target is TT
    bool all_empty = true;
    bool unsupported = false;
    for (auto& attrs : attributes.arguments) {
        if (!attrs.copy_buffer.empty()) {
            all_empty = false;
        }
        if (attrs.copy_target == "TENSTORRENT") {
            unsupported = true;
        }
    }
    if (all_empty || unsupported) {
        auto new_sdfg = sdfg.clone();
        std::list<std::unique_ptr<sdfg::StructuredSDFG>> result;
        result.push_back(std::move(new_sdfg));
        return result;
    }

    std::list<std::unique_ptr<sdfg::StructuredSDFG>> result;
    if (sdfg.return_type() != sdfg::types::Scalar(sdfg::types::PrimitiveType::Void)) {
        throw std::runtime_error(
            "SDFGs with non-void return types cannot be split for code generation when argument "
            "copying is enabled"
        );
    }

    // Create the kernel sdfg
    auto kernel_sdfg = sdfg.clone();
    kernel_sdfg->name(sdfg.name() + "_kernel");
    sdfg::builder::StructuredSDFGBuilder kernel_builder(kernel_sdfg);
    auto& kernel_root = kernel_builder.subject().root();
    sdfg::types::Scalar void_type(sdfg::types::PrimitiveType::Void);
    sdfg::types::Pointer opaque_ptr;
    std::list<std::unique_ptr<sdfg::StructuredSDFG>> tmp_result;

    // If return is present, remove it and everything after it
    size_t return_index = kernel_root.size();
    for (size_t i = 0; i < kernel_root.size(); i++) {
        if (dynamic_cast<sdfg::structured_control_flow::Return*>(&kernel_root.at(i).first)) {
            return_index = i;
            break;
        }
    }
    while (return_index < kernel_root.size()) {
        kernel_builder.remove_child(kernel_root, return_index);
    }

    // Declare the buffer containers as arguments
    for (auto& attrs : attributes.arguments) {
        if (attrs.copy_buffer.empty()) {
            continue;
        }
        if (kernel_builder.subject().is_transient(attrs.copy_buffer)) {
            auto buffer_type = kernel_builder.subject().type(attrs.copy_buffer).clone();
            kernel_builder.remove_container(attrs.copy_buffer);
            kernel_builder.add_container(attrs.copy_buffer, *buffer_type, true, false);
        }
    }

    // Create copy-ins and allocations at the beginning of SDFG
    while (kernel_root.size() > 0) {
        // Assignments are not allowed
        if (!kernel_root.at(0).second.empty()) {
            break;
        }

        // Child must be a block
        auto* block = dynamic_cast<sdfg::structured_control_flow::Block*>(&kernel_root.at(0).first);
        if (!block) {
            break;
        }

        // Block must contain exactly one library node
        auto& dfg = block->dataflow();
        if (dfg.library_nodes().size() != 1 || dfg.tasklets().size() != 0) {
            break;
        }

        // Disallow containers with managed types
        bool managed = false;
        for (auto* access_node : dfg.data_nodes()) {
            if (dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                continue;
            }
            if (sdfg.type(access_node->data()).storage_type().allocation() == sdfg::types::StorageType::Managed ||
                sdfg.type(access_node->data()).storage_type().deallocation() == sdfg::types::StorageType::Managed) {
                managed = true;
                break;
            }
        }
        if (managed) {
            break;
        }

        // Library node must be a copy-in or allocation
        auto* libnode = *dfg.library_nodes().begin();
        bool alloc = false, copy_in = false;
        std::string copy_buffer;
        if (dynamic_cast<sdfg::stdlib::AllocaNode*>(libnode) || dynamic_cast<sdfg::stdlib::CallocNode*>(libnode) ||
            dynamic_cast<sdfg::stdlib::MallocNode*>(libnode)) {
            alloc = true;
            auto& oedge = *dfg.out_edges(*libnode).begin();
            auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
            copy_buffer = dst.data();
        } else if (dynamic_cast<sdfg::stdlib::MemcpyNode*>(libnode) ||
                   dynamic_cast<sdfg::stdlib::MemmoveNode*>(libnode)) {
            copy_in = true;
            auto& oedge = *dfg.out_edges(*libnode).begin();
            auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
            copy_buffer = dst.data();
        } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
            if (cuda_offloading_node->is_h2d()) {
                copy_in = true;
                alloc = cuda_offloading_node->is_alloc();
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            } else if (cuda_offloading_node->is_alloc()) {
                alloc = true;
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            }
        } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
            if (tt_offloading_node->is_h2d()) {
                copy_in = true;
                alloc = tt_offloading_node->is_alloc();
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            } else if (tt_offloading_node->is_alloc()) {
                alloc = true;
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            }
        } else if (auto* external_offloading_node = dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode
                   )) {
            if (external_offloading_node->is_h2d()) {
                copy_in = true;
                alloc = external_offloading_node->is_alloc();
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            } else if (external_offloading_node->is_alloc()) {
                alloc = true;
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            }
        } else if (auto* rocm_offloading_node = dynamic_cast<sdfg::rocm::ROCMDataOffloadingNode*>(libnode)) {
            if (rocm_offloading_node->is_h2d()) {
                copy_in = true;
                alloc = rocm_offloading_node->is_alloc();
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            } else if (rocm_offloading_node->is_alloc()) {
                alloc = true;
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            }
        }

        if (!alloc && !copy_in) {
            break;
        }
        if (copy_buffer.empty()) {
            throw std::runtime_error("Could not find copy buffer while splitting SDFG '" + sdfg.name() + "'");
        }
        if (sdfg.is_argument(copy_buffer) || sdfg.is_external(copy_buffer)) {
            break;
        }

        // Find corresponding attributes
        size_t attr_index = attributes.arguments.size();
        for (size_t i = 0; i < attributes.arguments.size(); i++) {
            if (attributes.arguments[i].copy_buffer == copy_buffer) {
                attr_index = i;
                break;
            }
        }
        if (attr_index >= attributes.arguments.size()) {
            if (alloc && !copy_in) {
                break; // Single allocation ...
            }
            throw std::runtime_error("Could not find attributes for copy buffer '" + copy_buffer + "'");
        }

        // Handle allocation
        if (alloc) {
            // Create builder for allocation SDFG
            sdfg::builder::StructuredSDFGBuilder builder(
                sdfg.name() + "_alloc_" + std::to_string(attr_index),
                sdfg.type(),
                sdfg.type(attributes.arguments.at(attr_index).copy_buffer)
            );
            for (auto [key, value] : sdfg.metadata()) {
                builder.subject().add_metadata(key, value);
            }
            for (auto& arg : sdfg.arguments()) {
                builder.add_container(arg, sdfg.type(arg), true, false);
            }
            builder.add_container(copy_buffer, sdfg.type(copy_buffer), false, false);
            auto& root = builder.subject().root();

            // Copy the access nodes into a new block in the allocation SDFG
            auto& new_block = builder.add_block(root);
            std::unordered_map<sdfg::data_flow::DataFlowNode*, sdfg::data_flow::AccessNode*> access_node_map;
            for (auto& oedge : dfg.out_edges(*libnode)) {
                auto* access_node = dynamic_cast<sdfg::data_flow::AccessNode*>(&oedge.dst());
                if (auto* constant_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                    access_node_map.insert(
                        {constant_node,
                         &builder.add_constant(
                             new_block, constant_node->data(), constant_node->type(), constant_node->debug_info()
                         )}
                    );
                } else {
                    access_node_map.insert(
                        {access_node, &builder.add_access(new_block, access_node->data(), access_node->debug_info())}
                    );
                }
            }

            // Copy allocation
            sdfg::data_flow::LibraryNode* new_libnode = nullptr;
            if (auto* alloca_node = dynamic_cast<sdfg::stdlib::AllocaNode*>(libnode)) {
                new_libnode = &builder.add_library_node<
                    sdfg::stdlib::AllocaNode>(new_block, alloca_node->debug_info(), alloca_node->size());
            } else if (auto* calloc_node = dynamic_cast<sdfg::stdlib::CallocNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::stdlib::CallocNode>(
                    new_block, calloc_node->debug_info(), calloc_node->num(), calloc_node->size()
                );
            } else if (auto* malloc_node = dynamic_cast<sdfg::stdlib::MallocNode*>(libnode)) {
                new_libnode = &builder.add_library_node<
                    sdfg::stdlib::MallocNode>(new_block, malloc_node->debug_info(), malloc_node->size());
            } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::cuda::CUDADataOffloadingNode>(
                    new_block,
                    cuda_offloading_node->debug_info(),
                    cuda_offloading_node->size(),
                    cuda_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::ALLOC
                );
            } else if (auto* rocm_offloading_node = dynamic_cast<sdfg::rocm::ROCMDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::rocm::ROCMDataOffloadingNode>(
                    new_block,
                    rocm_offloading_node->debug_info(),
                    rocm_offloading_node->size(),
                    rocm_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::ALLOC
                );
            } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::tenstorrent::TTDataOffloadingNode>(
                    new_block,
                    tt_offloading_node->debug_info(),
                    tt_offloading_node->blocking(),
                    tt_offloading_node->device_handle(),
                    tt_offloading_node->size(),
                    tt_offloading_node->page_size(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::ALLOC,
                    tt_offloading_node->cq_no()
                );
            } else if (auto* external_offloading_node =
                           dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode)) {
                const std::string callee = external_offloading_node->callee_name() + "_alloc_" +
                                           std::to_string(external_offloading_node->transfer_index());
                builder.add_container(callee, sdfg.type(callee), false, true);
                std::vector<std::string>
                    inputs(external_offloading_node->inputs().begin(), external_offloading_node->inputs().end() - 1);
                if (!external_offloading_node->has_transfer() && external_offloading_node->is_alloc()) {
                    inputs.push_back(external_offloading_node->inputs().back());
                }
                new_libnode = &builder.add_library_node<sdfg::offloading::ExternalDataOffloadingNode>(
                    new_block,
                    external_offloading_node->debug_info(),
                    inputs,
                    external_offloading_node->callee_name(),
                    external_offloading_node->transfer_index(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::ALLOC
                );
                for (auto& iedge : dfg.in_edges(*libnode)) {
                    if (external_offloading_node->has_transfer() && iedge.dst_conn() == "_ret") {
                        continue;
                    }
                    auto* access_node = dynamic_cast<sdfg::data_flow::AccessNode*>(&iedge.src());
                    sdfg::data_flow::AccessNode* new_access_node = nullptr;
                    if (auto* constant_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                        new_access_node = &builder.add_constant(
                            new_block, constant_node->data(), constant_node->type(), constant_node->debug_info()
                        );
                    } else {
                        new_access_node =
                            &builder.add_access(new_block, access_node->data(), access_node->debug_info());
                    }
                    builder.add_memlet(
                        new_block,
                        *new_access_node,
                        iedge.src_conn(),
                        *new_libnode,
                        iedge.dst_conn(),
                        iedge.subset(),
                        iedge.base_type(),
                        iedge.debug_info()
                    );
                }
            }

            // Copy memlets
            for (auto& oedge : dfg.out_edges(*libnode)) {
                builder.add_memlet(
                    new_block,
                    *new_libnode,
                    "_ret",
                    *access_node_map.at(&oedge.dst()),
                    oedge.dst_conn(),
                    oedge.subset(),
                    oedge.base_type(),
                    oedge.debug_info()
                );
            }

            // Add return
            builder.add_return(root, copy_buffer);

            // Add allocation sdfg to result
            auto allocation_sdfg = builder.move();
            tmp_result.push_back(std::move(allocation_sdfg));
        }

        // Handle copy-in
        if (copy_in) {
            // Create builder for copy-in SDFG
            sdfg::builder::StructuredSDFGBuilder builder(
                sdfg.name() + "_in_" + std::to_string(attr_index),
                sdfg.type(),
                sdfg.type(attributes.arguments.at(attr_index).copy_buffer)
            );
            for (auto [key, value] : sdfg.metadata()) {
                builder.subject().add_metadata(key, value);
            }
            for (auto& arg : sdfg.arguments()) {
                builder.add_container(arg, sdfg.type(arg), true, false);
            }
            builder.add_container(copy_buffer, sdfg.type(copy_buffer), true, false);
            auto& root = builder.subject().root();

            // Copy the access nodes into a new block in the copy-in SDFG
            auto& new_block = builder.add_block(root);
            std::unordered_map<sdfg::data_flow::DataFlowNode*, sdfg::data_flow::AccessNode*> access_node_map;
            for (auto* access_node : dfg.data_nodes()) {
                if (auto* constant_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                    access_node_map.insert(
                        {constant_node,
                         &builder.add_constant(
                             new_block, constant_node->data(), constant_node->type(), constant_node->debug_info()
                         )}
                    );
                } else {
                    access_node_map.insert(
                        {access_node, &builder.add_access(new_block, access_node->data(), access_node->debug_info())}
                    );
                }
            }

            // Copy copy-in
            sdfg::data_flow::LibraryNode* new_libnode = nullptr;
            if (auto* memcpy_node = dynamic_cast<sdfg::stdlib::MemcpyNode*>(libnode)) {
                new_libnode = &builder.add_library_node<
                    sdfg::stdlib::MemcpyNode>(new_block, memcpy_node->debug_info(), memcpy_node->count());
            } else if (auto* memmove_node = dynamic_cast<sdfg::stdlib::MemmoveNode*>(libnode)) {
                new_libnode = &builder.add_library_node<
                    sdfg::stdlib::MemmoveNode>(new_block, memmove_node->debug_info(), memmove_node->count());
            } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::cuda::CUDADataOffloadingNode>(
                    new_block,
                    cuda_offloading_node->debug_info(),
                    cuda_offloading_node->size(),
                    cuda_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::H2D,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE
                );
            } else if (auto* rocm_offloading_node = dynamic_cast<sdfg::rocm::ROCMDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::rocm::ROCMDataOffloadingNode>(
                    new_block,
                    rocm_offloading_node->debug_info(),
                    rocm_offloading_node->size(),
                    rocm_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::H2D,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE
                );
            } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::tenstorrent::TTDataOffloadingNode>(
                    new_block,
                    tt_offloading_node->debug_info(),
                    tt_offloading_node->blocking(),
                    tt_offloading_node->device_handle(),
                    tt_offloading_node->size(),
                    tt_offloading_node->page_size(),
                    sdfg::offloading::DataTransferDirection::H2D,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE,
                    tt_offloading_node->cq_no()
                );
            } else if (auto* external_offloading_node =
                           dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode)) {
                const std::string callee = external_offloading_node->callee_name() + "_in_" +
                                           std::to_string(external_offloading_node->transfer_index());
                builder.add_container(callee, sdfg.type(callee), false, true);
                new_libnode = &builder.add_library_node<sdfg::offloading::ExternalDataOffloadingNode>(
                    new_block,
                    external_offloading_node->debug_info(),
                    std::vector<std::string>(
                        external_offloading_node->inputs().begin(), external_offloading_node->inputs().end() - 1
                    ),
                    external_offloading_node->callee_name(),
                    external_offloading_node->transfer_index(),
                    sdfg::offloading::DataTransferDirection::H2D,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE
                );
            }

            // Copy memlets
            for (auto& iedge : dfg.in_edges(*libnode)) {
                builder.add_memlet(
                    new_block,
                    *access_node_map.at(&iedge.src()),
                    iedge.src_conn(),
                    *new_libnode,
                    iedge.dst_conn(),
                    iedge.subset(),
                    iedge.base_type(),
                    iedge.debug_info()
                );
            }
            for (auto& oedge : dfg.out_edges(*libnode)) {
                builder.add_memlet(
                    new_block,
                    *new_libnode,
                    oedge.src_conn(),
                    *access_node_map.at(&oedge.dst()),
                    oedge.dst_conn(),
                    oedge.subset(),
                    oedge.base_type(),
                    oedge.debug_info()
                );
            }

            // Add return
            builder.add_return(root, copy_buffer);

            // Add copy-in sdfg to result
            auto copy_in_sdfg = builder.move();
            tmp_result.push_back(std::move(copy_in_sdfg));
        }

        // Delete child
        kernel_builder.remove_child(kernel_root, 0);
    }

    // Create copy-outs and frees at the end of SDFG
    while (kernel_root.size() > 0) {
        // Assignments are not allowed
        if (!kernel_root.at(kernel_root.size() - 1).second.empty()) {
            break;
        }

        // Child must be a block
        auto* block = dynamic_cast<sdfg::structured_control_flow::Block*>(&kernel_root.at(kernel_root.size() - 1).first
        );
        if (!block) {
            break;
        }

        // Block must contain exactly one library node
        auto& dfg = block->dataflow();
        if (dfg.library_nodes().size() != 1 || dfg.tasklets().size() != 0) {
            break;
        }

        // Library node symbols must only depend on arguments
        auto* libnode = *dfg.library_nodes().begin();
        bool symbols_invalid = false;
        for (auto& symbol : libnode->symbols()) {
            if (!sdfg.is_argument(symbol->get_name())) {
                symbols_invalid = true;
                break;
            }
        }
        if (symbols_invalid) {
            break;
        }

        // Disallow containers with managed types
        bool managed = false;
        for (auto* access_node : dfg.data_nodes()) {
            if (dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                continue;
            }
            if (sdfg.type(access_node->data()).storage_type().allocation() == sdfg::types::StorageType::Managed ||
                sdfg.type(access_node->data()).storage_type().deallocation() == sdfg::types::StorageType::Managed) {
                managed = true;
                break;
            }
        }
        if (managed) {
            break;
        }

        // Library node must be a copy-out or free
        bool copy_out = false, free = false;
        std::string copy_buffer;
        if (dynamic_cast<sdfg::stdlib::MemcpyNode*>(libnode) || dynamic_cast<sdfg::stdlib::MemmoveNode*>(libnode)) {
            copy_out = true;
            auto& iedge = *dfg.in_edges(*libnode).begin();
            auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
            copy_buffer = src.data();
        } else if (dynamic_cast<sdfg::stdlib::FreeNode*>(libnode)) {
            free = true;
            auto& iedge = *dfg.in_edges(*libnode).begin();
            auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
            copy_buffer = src.data();
        } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
            if (cuda_offloading_node->is_d2h()) {
                copy_out = true;
                free = cuda_offloading_node->is_free();
                auto& iedge = *dfg.in_edges(*libnode).begin();
                auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
                copy_buffer = src.data();
            } else if (cuda_offloading_node->is_free()) {
                free = true;
                auto& iedge = *dfg.in_edges(*libnode).begin();
                auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
                copy_buffer = src.data();
            }
        } else if (auto* rocm_offloading_node = dynamic_cast<sdfg::rocm::ROCMDataOffloadingNode*>(libnode)) {
            if (rocm_offloading_node->is_d2h()) {
                copy_out = true;
                free = rocm_offloading_node->is_free();
                auto& iedge = *dfg.in_edges(*libnode).begin();
                auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
                copy_buffer = src.data();
            } else if (rocm_offloading_node->is_free()) {
                free = true;
                auto& iedge = *dfg.in_edges(*libnode).begin();
                auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
                copy_buffer = src.data();
            }
        } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
            if (tt_offloading_node->is_d2h()) {
                copy_out = true;
                free = tt_offloading_node->is_free();
                auto& iedge = *dfg.in_edges(*libnode).begin();
                auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
                copy_buffer = src.data();
            } else if (tt_offloading_node->is_free()) {
                free = true;
                auto& iedge = *dfg.in_edges(*libnode).begin();
                auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
                copy_buffer = src.data();
            }
        } else if (auto* external_offloading_node = dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode
                   )) {
            if (external_offloading_node->is_d2h()) {
                copy_out = true;
                free = external_offloading_node->is_free();
                const std::string& conn = external_offloading_node->inputs().back();
                for (auto& iedge : dfg.in_edges(*libnode)) {
                    if (iedge.dst_conn() == conn) {
                        auto& src = static_cast<sdfg::data_flow::AccessNode&>(iedge.src());
                        copy_buffer = src.data();
                        break;
                    }
                }
            } else if (external_offloading_node->is_free()) {
                free = true;
                auto& oedge = *dfg.out_edges(*libnode).begin();
                auto& dst = static_cast<sdfg::data_flow::AccessNode&>(oedge.dst());
                copy_buffer = dst.data();
            }
        }
        if (!copy_out && !free) {
            break;
        }
        if (copy_buffer.empty()) {
            throw std::runtime_error("Could not find copy buffer while splitting SDFG '" + sdfg.name() + "'");
        }
        if (sdfg.is_argument(copy_buffer) || sdfg.is_external(copy_buffer)) {
            break;
        }

        // Find corresponding attributes
        size_t attr_index = attributes.arguments.size();
        for (size_t i = 0; i < attributes.arguments.size(); i++) {
            if (attributes.arguments[i].copy_buffer == copy_buffer) {
                attr_index = i;
                break;
            }
        }
        if (attr_index >= attributes.arguments.size()) {
            if (!copy_out && free) {
                break; // Single free ...
            }
            throw std::runtime_error("Could not find attributes for copy buffer '" + copy_buffer + "'");
        }

        // Handle copy-out
        if (copy_out) {
            // Create builder for copy-out SDFG
            sdfg::builder::StructuredSDFGBuilder
                builder(sdfg.name() + "_out_" + std::to_string(attr_index), sdfg.type());
            for (auto [key, value] : sdfg.metadata()) {
                builder.subject().add_metadata(key, value);
            }
            for (auto& arg : sdfg.arguments()) {
                builder.add_container(arg, sdfg.type(arg), true, false);
            }
            builder.add_container(copy_buffer, sdfg.type(copy_buffer), true, false);
            auto& root = builder.subject().root();

            // Copy the access nodes into a new block in the copy-out SDFG
            auto& new_block = builder.add_block(root);
            std::unordered_map<sdfg::data_flow::DataFlowNode*, sdfg::data_flow::AccessNode*> access_node_map;
            for (auto* access_node : dfg.data_nodes()) {
                if (auto* constant_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                    access_node_map.insert(
                        {constant_node,
                         &builder.add_constant(
                             new_block, constant_node->data(), constant_node->type(), constant_node->debug_info()
                         )}
                    );
                } else {
                    access_node_map.insert(
                        {access_node, &builder.add_access(new_block, access_node->data(), access_node->debug_info())}
                    );
                }
            }

            // Copy copy-out
            sdfg::data_flow::LibraryNode* new_libnode = nullptr;
            if (auto* memcpy_node = dynamic_cast<sdfg::stdlib::MemcpyNode*>(libnode)) {
                new_libnode = &builder.add_library_node<
                    sdfg::stdlib::MemcpyNode>(new_block, memcpy_node->debug_info(), memcpy_node->count());
            } else if (auto* memmove_node = dynamic_cast<sdfg::stdlib::MemmoveNode*>(libnode)) {
                new_libnode = &builder.add_library_node<
                    sdfg::stdlib::MemmoveNode>(new_block, memmove_node->debug_info(), memmove_node->count());
            } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::cuda::CUDADataOffloadingNode>(
                    new_block,
                    cuda_offloading_node->debug_info(),
                    cuda_offloading_node->size(),
                    cuda_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::D2H,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE
                );
            } else if (auto* rocm_offloading_node = dynamic_cast<sdfg::rocm::ROCMDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::rocm::ROCMDataOffloadingNode>(
                    new_block,
                    rocm_offloading_node->debug_info(),
                    rocm_offloading_node->size(),
                    rocm_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::D2H,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE
                );
            } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::tenstorrent::TTDataOffloadingNode>(
                    new_block,
                    tt_offloading_node->debug_info(),
                    tt_offloading_node->blocking(),
                    tt_offloading_node->device_handle(),
                    tt_offloading_node->size(),
                    tt_offloading_node->page_size(),
                    sdfg::offloading::DataTransferDirection::D2H,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE,
                    tt_offloading_node->cq_no()
                );
            } else if (auto* external_offloading_node =
                           dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode)) {
                const std::string callee = external_offloading_node->callee_name() + "_out_" +
                                           std::to_string(external_offloading_node->transfer_index());
                builder.add_container(callee, sdfg.type(callee), false, true);
                new_libnode = &builder.add_library_node<sdfg::offloading::ExternalDataOffloadingNode>(
                    new_block,
                    external_offloading_node->debug_info(),
                    std::vector<std::string>(
                        external_offloading_node->inputs().begin(), external_offloading_node->inputs().end() - 1
                    ),
                    external_offloading_node->callee_name(),
                    external_offloading_node->transfer_index(),
                    sdfg::offloading::DataTransferDirection::D2H,
                    sdfg::offloading::BufferLifecycle::NO_CHANGE
                );
            }

            // Copy memlets
            for (auto& iedge : dfg.in_edges(*libnode)) {
                builder.add_memlet(
                    new_block,
                    *access_node_map.at(&iedge.src()),
                    iedge.src_conn(),
                    *new_libnode,
                    iedge.dst_conn(),
                    iedge.subset(),
                    iedge.base_type(),
                    iedge.debug_info()
                );
            }
            for (auto& oedge : dfg.out_edges(*libnode)) {
                builder.add_memlet(
                    new_block,
                    *new_libnode,
                    oedge.src_conn(),
                    *access_node_map.at(&oedge.dst()),
                    oedge.dst_conn(),
                    oedge.subset(),
                    oedge.base_type(),
                    oedge.debug_info()
                );
            }

            // Add copy-out sdfg to result
            auto copy_out_sdfg = builder.move();
            tmp_result.push_back(std::move(copy_out_sdfg));
        }

        // Handle free
        if (free) {
            // Create builder for free SDFG
            sdfg::builder::StructuredSDFGBuilder builder(
                sdfg.name() + "_free_" + std::to_string(attr_index),
                sdfg.type(),
                sdfg.type(attributes.arguments.at(attr_index).copy_buffer)
            );
            for (auto [key, value] : sdfg.metadata()) {
                builder.subject().add_metadata(key, value);
            }
            for (auto& arg : sdfg.arguments()) {
                builder.add_container(arg, sdfg.type(arg), true, false);
            }
            builder.add_container(copy_buffer, sdfg.type(copy_buffer), true, false);
            auto& root = builder.subject().root();

            // Copy the access nodes into a new block in the free SDFG
            auto& new_block = builder.add_block(root);
            auto& iedge = *dfg.in_edges(*libnode).begin();
            auto* access_node = dynamic_cast<sdfg::data_flow::AccessNode*>(&iedge.src());
            auto& in_node = builder.add_access(new_block, access_node->data(), access_node->debug_info());
            auto& out_node = builder.add_access(new_block, access_node->data(), access_node->debug_info());

            // Copy free
            sdfg::data_flow::LibraryNode* new_libnode = nullptr;
            if (auto* free_node = dynamic_cast<sdfg::stdlib::FreeNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::stdlib::FreeNode>(new_block, free_node->debug_info());
            } else if (auto* cuda_offloading_node = dynamic_cast<sdfg::cuda::CUDADataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::cuda::CUDADataOffloadingNode>(
                    new_block,
                    cuda_offloading_node->debug_info(),
                    cuda_offloading_node->size(),
                    cuda_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::FREE
                );
            } else if (auto* rocm_offloading_node = dynamic_cast<sdfg::rocm::ROCMDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::rocm::ROCMDataOffloadingNode>(
                    new_block,
                    rocm_offloading_node->debug_info(),
                    rocm_offloading_node->size(),
                    rocm_offloading_node->device_id(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::FREE
                );
            } else if (auto* tt_offloading_node = dynamic_cast<sdfg::tenstorrent::TTDataOffloadingNode*>(libnode)) {
                new_libnode = &builder.add_library_node<sdfg::tenstorrent::TTDataOffloadingNode>(
                    new_block,
                    tt_offloading_node->debug_info(),
                    tt_offloading_node->blocking(),
                    tt_offloading_node->device_handle(),
                    tt_offloading_node->size(),
                    tt_offloading_node->page_size(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::FREE,
                    tt_offloading_node->cq_no()
                );
            } else if (auto* external_offloading_node =
                           dynamic_cast<sdfg::offloading::ExternalDataOffloadingNode*>(libnode)) {
                const std::string callee = external_offloading_node->callee_name() + "_free_" +
                                           std::to_string(external_offloading_node->transfer_index());
                builder.add_container(callee, sdfg.type(callee), false, true);
                new_libnode = &builder.add_library_node<sdfg::offloading::ExternalDataOffloadingNode>(
                    new_block,
                    external_offloading_node->debug_info(),
                    std::vector<std::string>(
                        external_offloading_node->inputs().begin(), external_offloading_node->inputs().end() - 1
                    ),
                    external_offloading_node->callee_name(),
                    external_offloading_node->transfer_index(),
                    sdfg::offloading::DataTransferDirection::NONE,
                    sdfg::offloading::BufferLifecycle::FREE
                );
                for (auto& iedge : dfg.in_edges(*libnode)) {
                    if ((external_offloading_node->has_transfer() &&
                         iedge.dst_conn() == external_offloading_node->inputs().back()) ||
                        (!external_offloading_node->has_transfer() && iedge.dst_conn() == "_ptr")) {
                        continue;
                    }
                    auto* access_node = dynamic_cast<sdfg::data_flow::AccessNode*>(&iedge.src());
                    sdfg::data_flow::AccessNode* new_access_node = nullptr;
                    if (auto* constant_node = dynamic_cast<sdfg::data_flow::ConstantNode*>(access_node)) {
                        new_access_node = &builder.add_constant(
                            new_block, constant_node->data(), constant_node->type(), constant_node->debug_info()
                        );
                    } else {
                        new_access_node =
                            &builder.add_access(new_block, access_node->data(), access_node->debug_info());
                    }
                    builder.add_memlet(
                        new_block,
                        *new_access_node,
                        iedge.src_conn(),
                        *new_libnode,
                        iedge.dst_conn(),
                        iedge.subset(),
                        iedge.base_type(),
                        iedge.debug_info()
                    );
                }
            }

            // Copy memlets
            builder.add_memlet(
                new_block, in_node, "void", *new_libnode, "_ptr", iedge.subset(), iedge.base_type(), iedge.debug_info()
            );
            builder.add_memlet(
                new_block, *new_libnode, "_ptr", out_node, "void", iedge.subset(), iedge.base_type(), iedge.debug_info()
            );

            // Add return
            builder.add_return(root, copy_buffer);

            // Add copy-out sdfg to result
            auto copy_out_sdfg = builder.move();
            tmp_result.push_back(std::move(copy_out_sdfg));
        }

        // Delete child
        kernel_builder.remove_child(kernel_root, kernel_root.size() - 1);
    }

    // Add kernel sdfg to result
    kernel_sdfg = kernel_builder.move();
    result.push_back(std::move(kernel_sdfg));

    // Add other allocation, copy-in, copy-out, and free SDFGs to result
    while (!tmp_result.empty()) {
        auto tmp_sdfg = std::move(tmp_result.front());
        tmp_result.pop_front();
        result.push_back(std::move(tmp_sdfg));
    }

    // Construct wrapper SDFG
    sdfg::builder::StructuredSDFGBuilder wrapper_builder(sdfg.name(), sdfg.type());
    for (auto [key, value] : sdfg.metadata()) {
        wrapper_builder.subject().add_metadata(key, value);
    }
    for (auto& arg : sdfg.arguments()) {
        wrapper_builder.add_container(arg, sdfg.type(arg), true, false);
    }
    auto& wrapper_root = wrapper_builder.subject().root();

    // Add buffer containers and function containers
    sdfg::types::Function kernel_type(void_type, false);
    for (auto& arg : sdfg.arguments()) {
        kernel_type.add_param(sdfg.type(arg));
    }
    for (size_t i = 0; i < attributes.arguments.size(); i++) {
        auto& attrs = attributes.arguments.at(i);
        if (attrs.copy_buffer.empty()) {
            continue;
        }
        auto& buffer_type = sdfg.type(attrs.copy_buffer);
        wrapper_builder.add_container(attrs.copy_buffer, buffer_type, false, false);
        kernel_type.add_param(buffer_type);
        if (attrs.alloc) {
            sdfg::types::Function alloc_type(buffer_type, false);
            for (auto& arg : sdfg.arguments()) {
                alloc_type.add_param(sdfg.type(arg));
            }
            wrapper_builder.add_container(sdfg.name() + "_alloc_" + std::to_string(i), alloc_type, false, true);
        }
        if (attrs.copy_in) {
            sdfg::types::Function copy_in_type(buffer_type, false);
            for (auto& arg : sdfg.arguments()) {
                copy_in_type.add_param(sdfg.type(arg));
            }
            copy_in_type.add_param(buffer_type);
            wrapper_builder.add_container(sdfg.name() + "_in_" + std::to_string(i), copy_in_type, false, true);
        }
        if (attrs.copy_out) {
            sdfg::types::Function copy_out_type(void_type, false);
            for (auto& arg : sdfg.arguments()) {
                copy_out_type.add_param(sdfg.type(arg));
            }
            copy_out_type.add_param(buffer_type);
            wrapper_builder.add_container(sdfg.name() + "_out_" + std::to_string(i), copy_out_type, false, true);
        }
        if (attrs.free) {
            sdfg::types::Function free_type(buffer_type, false);
            for (auto& arg : sdfg.arguments()) {
                free_type.add_param(sdfg.type(arg));
            }
            free_type.add_param(buffer_type);
            wrapper_builder.add_container(sdfg.name() + "_free_" + std::to_string(i), free_type, false, true);
        }
    }
    wrapper_builder.add_container(sdfg.name() + "_kernel", kernel_type, false, true);

    // Add kernel call
    auto& kernel_block = wrapper_builder.add_block(wrapper_root);
    {
        std::vector<std::string> conns;
        for (size_t i = 0; i < sdfg.arguments().size(); i++) {
            conns.push_back("_arg" + std::to_string(i));
        }
        for (auto& attrs : attributes.arguments) {
            if (!attrs.copy_buffer.empty()) {
                conns.push_back("_arg" + std::to_string(conns.size()));
            }
        }

        auto& libnode = wrapper_builder.add_library_node<
            sdfg::data_flow::CallNode>(kernel_block, sdfg::DebugInfo(), sdfg.name() + "_kernel", conns, conns);

        for (size_t i = 0; i < sdfg.arguments().size(); i++) {
            auto& arg = sdfg.arguments().at(i);
            auto& in_node = wrapper_builder.add_access(kernel_block, arg);
            auto& out_node = wrapper_builder.add_access(kernel_block, arg);
            wrapper_builder.add_computational_memlet(kernel_block, in_node, libnode, conns.at(i), {}, sdfg.type(arg));
            wrapper_builder.add_computational_memlet(kernel_block, libnode, conns.at(i), out_node, {}, sdfg.type(arg));
        }
        size_t i = sdfg.arguments().size();
        for (auto& attrs : attributes.arguments) {
            if (attrs.copy_buffer.empty()) {
                continue;
            }
            auto& in_node = wrapper_builder.add_access(kernel_block, attrs.copy_buffer);
            auto& out_node = wrapper_builder.add_access(kernel_block, attrs.copy_buffer);
            wrapper_builder
                .add_computational_memlet(kernel_block, in_node, libnode, conns.at(i), {}, sdfg.type(attrs.copy_buffer));
            wrapper_builder
                .add_computational_memlet(kernel_block, libnode, conns.at(i), out_node, {}, sdfg.type(attrs.copy_buffer));
            i++;
        }
    }

    // Add allocations, copy-ins, copy-outs, and frees
    for (size_t i = 0; i < attributes.arguments.size(); i++) {
        auto& attrs = attributes.arguments.at(i);
        if (attrs.copy_buffer.empty()) {
            continue;
        }
        if (attrs.alloc || attrs.copy_in) {
            auto& block = wrapper_builder.add_block_before(wrapper_root, kernel_block, {}, sdfg::DebugInfo());
            std::vector<std::string> inputs;
            for (size_t j = 0; j < sdfg.arguments().size(); j++) {
                inputs.push_back("_arg" + std::to_string(j));
            }
            auto& libnode = wrapper_builder.add_library_node<
                sdfg::offloading::ExternalDataOffloadingNode,
                const std::vector<std::string>&,
                const std::string&,
                size_t,
                sdfg::offloading::DataTransferDirection,
                sdfg::offloading::BufferLifecycle>(
                block,
                sdfg::DebugInfo(),
                inputs,
                sdfg.name(),
                i,
                (attrs.copy_in ? sdfg::offloading::DataTransferDirection::H2D
                               : sdfg::offloading::DataTransferDirection::NONE),
                (attrs.alloc ? sdfg::offloading::BufferLifecycle::ALLOC : sdfg::offloading::BufferLifecycle::NO_CHANGE)
            );
            for (size_t j = 0; j < sdfg.arguments().size(); j++) {
                auto& arg = sdfg.arguments().at(j);
                auto& access_node = wrapper_builder.add_access(block, arg);
                wrapper_builder.add_computational_memlet(block, access_node, libnode, inputs.at(j), {}, sdfg.type(arg));
            }
            if (attrs.copy_in) {
                auto& access_node = wrapper_builder.add_access(block, attrs.copy_buffer);
                wrapper_builder
                    .add_computational_memlet(block, access_node, libnode, "_ret", {}, sdfg.type(attrs.copy_buffer));
            }
            auto& access_node = wrapper_builder.add_access(block, attrs.copy_buffer);
            wrapper_builder
                .add_computational_memlet(block, libnode, "_ret", access_node, {}, sdfg.type(attrs.copy_buffer));
        }
        if (attrs.copy_out || attrs.free) {
            auto& block = wrapper_builder.add_block_after(wrapper_root, kernel_block, {}, sdfg::DebugInfo());
            std::vector<std::string> inputs;
            for (size_t j = 0; j < sdfg.arguments().size(); j++) {
                inputs.push_back("_arg" + std::to_string(j));
            }
            auto& libnode = wrapper_builder.add_library_node<
                sdfg::offloading::ExternalDataOffloadingNode,
                const std::vector<std::string>&,
                const std::string&,
                size_t,
                sdfg::offloading::DataTransferDirection,
                sdfg::offloading::BufferLifecycle>(
                block,
                sdfg::DebugInfo(),
                inputs,
                sdfg.name(),
                i,
                (attrs.copy_out ? sdfg::offloading::DataTransferDirection::D2H
                                : sdfg::offloading::DataTransferDirection::NONE),
                (attrs.free ? sdfg::offloading::BufferLifecycle::FREE : sdfg::offloading::BufferLifecycle::NO_CHANGE)
            );
            for (size_t j = 0; j < sdfg.arguments().size(); j++) {
                auto& arg = sdfg.arguments().at(j);
                auto& access_node = wrapper_builder.add_access(block, arg);
                wrapper_builder.add_computational_memlet(block, access_node, libnode, inputs.at(j), {}, sdfg.type(arg));
            }
            auto& in_node = wrapper_builder.add_access(block, attrs.copy_buffer);
            auto& out_node = wrapper_builder.add_access(block, attrs.copy_buffer);
            if (attrs.copy_out) {
                wrapper_builder.add_computational_memlet(
                    block, in_node, libnode, "_arg" + std::to_string(inputs.size()), {}, sdfg.type(attrs.copy_buffer)
                );
                wrapper_builder
                    .add_computational_memlet(block, libnode, inputs.at(i), out_node, {}, sdfg.type(attrs.copy_buffer));
            } else {
                wrapper_builder
                    .add_computational_memlet(block, in_node, libnode, "_ptr", {}, sdfg.type(attrs.copy_buffer));
                wrapper_builder
                    .add_computational_memlet(block, libnode, "_ptr", out_node, {}, sdfg.type(attrs.copy_buffer));
            }
        }
    }

    // Add wrapper sdfg to result
    auto wrapper_sdfg = wrapper_builder.move();
    result.push_back(std::move(wrapper_sdfg));

    return result;
}

} // namespace passes
} // namespace docc
