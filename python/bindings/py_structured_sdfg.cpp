#include "py_structured_sdfg.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json_fwd.hpp>
#include <sstream>

#include <dlfcn.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

#include <sdfg/analysis/analysis.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/code_generators/cpp_code_generator.h>
#include <sdfg/codegen/instrumentation/arg_capture_plan.h>
#include <sdfg/codegen/instrumentation/instrumentation_plan.h>
#include <sdfg/codegen/loop_report.h>
#include <sdfg/passes/dataflow/constant_propagation.h>
#include <sdfg/passes/dataflow/dead_data_elimination.h>
#include <sdfg/passes/dataflow/tensor_to_pointer_conversion.h>
#include <sdfg/passes/dot_expansion_pass.h>
#include <sdfg/passes/gemm_expansion_pass.h>
#include <sdfg/passes/normalization/normalization.h>
#include <sdfg/passes/offloading/cuda_library_node_rewriter_pass.h>
#include <sdfg/passes/offloading/onnx_library_node_rewriter_pass.h>
#include <sdfg/passes/opt_pipeline.h>
#include <sdfg/passes/pipeline.h>
#include <sdfg/passes/scheduler/cuda_scheduler.h>
#include <sdfg/passes/scheduler/highway_scheduler.h>
#include <sdfg/passes/scheduler/loop_scheduling_pass.h>
#include <sdfg/passes/scheduler/omp_scheduler.h>
#include <sdfg/passes/scheduler/polly_scheduler.h>
#include <sdfg/passes/scheduler/scheduler_registry.h>
#include <sdfg/passes/structured_control_flow/common_assignment_elimination.h>
#include <sdfg/passes/structured_control_flow/condition_elimination.h>
#include <sdfg/passes/structured_control_flow/for2map.h>
#include <sdfg/passes/structured_control_flow/loop_normalization.h>
#include <sdfg/passes/structured_control_flow/pointer_evolution.h>
#include <sdfg/passes/structured_control_flow/while_to_for_conversion.h>
#include <sdfg/passes/symbolic/symbol_evolution.h>
#include <sdfg/passes/symbolic/symbol_promotion.h>
#include <sdfg/passes/symbolic/symbol_propagation.h>
#include <sdfg/passes/symbolic/type_minimization.h>
#include <sdfg/serializer/json_serializer.h>

// Platform-specific compiler selection
#if defined(__APPLE__)
#define DOCC_CXX_COMPILER "clang++"
#elif defined(__linux__)
#define DOCC_CXX_COMPILER "clang-19"
#else
#error "Unsupported platform"
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;

PyStructuredSDFG::PyStructuredSDFG(std::unique_ptr<sdfg::StructuredSDFG>& sdfg) : sdfg_(std::move(sdfg)) {}

PyStructuredSDFG PyStructuredSDFG::parse(const std::string& sdfg_text) {
    json j = json::parse(sdfg_text);
    sdfg::serializer::JSONSerializer serializer;
    auto sdfg = serializer.deserialize(j);

    return PyStructuredSDFG(sdfg);
}

PyStructuredSDFG PyStructuredSDFG::from_file(const std::string& file_path) {
    std::ifstream sdfg_file(file_path);
    if (!sdfg_file.is_open()) {
        throw std::runtime_error("Failed to open SDFG file: " + file_path);
    }

    json j;
    sdfg_file >> j;
    sdfg::serializer::JSONSerializer serializer;
    auto sdfg = serializer.deserialize(j);

    return PyStructuredSDFG(sdfg);
}

std::string PyStructuredSDFG::name() const { return sdfg_->name(); }

const sdfg::types::IType& PyStructuredSDFG::return_type() const { return sdfg_->return_type(); }

const sdfg::types::IType& PyStructuredSDFG::type(const std::string& name) const { return sdfg_->type(name); }

bool PyStructuredSDFG::exists(const std::string& name) const { return sdfg_->exists(name); }

bool PyStructuredSDFG::is_argument(const std::string& name) const { return sdfg_->is_argument(name); }

bool PyStructuredSDFG::is_transient(const std::string& name) const { return sdfg_->is_transient(name); }

std::vector<std::string> PyStructuredSDFG::arguments() const { return sdfg_->arguments(); }

pybind11::dict PyStructuredSDFG::containers() const {
    pybind11::dict result;
    for (const auto& name : sdfg_->containers()) {
        result[name.c_str()] = pybind11::cast(sdfg_->type(name), pybind11::return_value_policy::reference);
    }
    return result;
}

namespace {
void _anchor() {}
} // namespace

void PyStructuredSDFG::validate() { sdfg_->validate(); }

void PyStructuredSDFG::expand() {
    sdfg::builder::StructuredSDFGBuilder builder_opt(*sdfg_);
    sdfg::analysis::AnalysisManager analysis_manager(*sdfg_);

    sdfg::passes::Pipeline libnode_expansion = sdfg::passes::Pipeline::expansion();
    libnode_expansion.run(builder_opt, analysis_manager);

    sdfg::passes::TensorToPointerConversionPass tensor_to_pointer_conversion_pass;
    tensor_to_pointer_conversion_pass.run(builder_opt, analysis_manager);
}

void PyStructuredSDFG::simplify() {
    sdfg::builder::StructuredSDFGBuilder builder_opt(*sdfg_);
    sdfg::analysis::AnalysisManager analysis_manager(*sdfg_);

    // Expand Dot nodes
    sdfg::passes::DotExpansionPass dot_expansion_pass;
    dot_expansion_pass.run(builder_opt, analysis_manager);
    // sdfg::passes::GemmExpansionPass gemm_expansion_pass;
    // gemm_expansion_pass.run(builder_opt, analysis_manager);

    // Optimization Pipelines
    sdfg::passes::Pipeline dataflow_simplification = sdfg::passes::Pipeline::dataflow_simplification();
    sdfg::passes::Pipeline symbolic_simplification = sdfg::passes::Pipeline::symbolic_simplification();
    sdfg::passes::Pipeline dce = sdfg::passes::Pipeline::dead_code_elimination();
    sdfg::passes::Pipeline memlet_combine = sdfg::passes::Pipeline::memlet_combine();
    sdfg::passes::DeadDataElimination dde;
    sdfg::passes::SymbolPropagation symbol_propagation_pass;

    // Promote tasklets into symbolic assignments
    sdfg::passes::SymbolPromotion symbol_promotion_pass;
    symbol_promotion_pass.run(builder_opt, analysis_manager);

    // Minimize SDFG by fusing blocks, tasklets and sequences
    dataflow_simplification.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);

    // Minimize SDFG by fusing symbolic expressions
    symbolic_simplification.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);

    /***** Structured Loops *****/

    // Unify continue/break inside branches
    {
        sdfg::passes::CommonAssignmentElimination common_assignment_elimination;
        bool applies = false;
        do {
            applies = false;
            applies |= common_assignment_elimination.run(builder_opt, analysis_manager);
        } while (applies);
        dde.run(builder_opt, analysis_manager);
        dce.run(builder_opt, analysis_manager);
        symbolic_simplification.run(builder_opt, analysis_manager);
    }

    // Propagate variables into constants
    {
        sdfg::passes::ConstantPropagation constant_propagation_pass;
        bool applies = false;
        do {
            applies = false;
            applies |= constant_propagation_pass.run(builder_opt, analysis_manager);
        } while (applies);
    }

    // Convert loops into structured loops
    sdfg::passes::WhileToForConversion for_conversion_pass;
    for_conversion_pass.run(builder_opt, analysis_manager);

    // Propagate for simpler indvar usage
    symbol_propagation_pass.run(builder_opt, analysis_manager);

    // Eliminate redundant branches
    {
        bool applies = false;
        sdfg::passes::ConditionEliminationPass condition_elimination_pass;
        do {
            applies = false;
            applies |= condition_elimination_pass.run(builder_opt, analysis_manager);
        } while (applies);
    }

    // Normalize loop condition and update (run twice)
    sdfg::passes::LoopNormalizationPass loop_normalization_pass;
    loop_normalization_pass.run(builder_opt, analysis_manager);
    loop_normalization_pass.run(builder_opt, analysis_manager);

    // Eliminate symbols correlated to loop iterators
    // sdfg::passes::SymbolEvolution symbol_evolution_pass;
    // symbol_evolution_pass.run(builder_opt, analysis_manager);

    // Dead code elimination
    symbol_propagation_pass.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);

    /***** Data Parallelism *****/

    // Combine address calculations in memlets
    memlet_combine.run(builder_opt, analysis_manager);

    // Move code out of loops where possible
    sdfg::passes::Pipeline code_motion = sdfg::passes::code_motion();
    code_motion.run(builder_opt, analysis_manager);

    // Convert pointer-based iterators to indvar usage
    sdfg::passes::PointerEvolution pointer_evolution_pass;
    pointer_evolution_pass.run(builder_opt, analysis_manager);
    loop_normalization_pass.run(builder_opt, analysis_manager);

    sdfg::passes::TypeMinimizationPass type_minimization_pass;
    type_minimization_pass.run(builder_opt, analysis_manager);
    type_minimization_pass.run(builder_opt, analysis_manager);

    // Dead code elimination
    symbol_propagation_pass.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);
    dde.run(builder_opt, analysis_manager);

    // Convert for loops into maps
    sdfg::passes::For2MapPass map_conversion_pass;
    map_conversion_pass.run(builder_opt, analysis_manager);

    // Move code out of maps where possible
    code_motion.run(builder_opt, analysis_manager);

    // Dead code elimination
    dde.run(builder_opt, analysis_manager);
    dce.run(builder_opt, analysis_manager);
    dataflow_simplification.run(builder_opt, analysis_manager);
}

void PyStructuredSDFG::dump(const std::string& path) {
    fs::path build_path(path);
    if (!fs::exists(build_path)) {
        fs::create_directories(build_path);
    }

    // Add metadata to SDFG
    fs::path sdfg_file = build_path / (sdfg_->name() + ".json");
    fs::path features_file = build_path / (sdfg_->name() + ".npz");
    fs::path arg_captures_path = build_path / "arg_captures";
    sdfg_->add_metadata("sdfg_file", sdfg_file.string());
    sdfg_->add_metadata("arg_capture_path", arg_captures_path.string());
    sdfg_->add_metadata("features_file", features_file.string());
    sdfg_->add_metadata("opt_report_file", (build_path / (sdfg_->name() + ".opt_report.json")).string());

    // Dump json
    sdfg::serializer::JSONSerializer serializer;
    nlohmann::json j = serializer.serialize(*this->sdfg_);

    std::ofstream ofs(sdfg_file);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file: " + sdfg_file.string());
    }
    ofs << j.dump(2);
    ofs.close();
}

void PyStructuredSDFG::normalize() {
    sdfg::builder::StructuredSDFGBuilder builder(*sdfg_);
    sdfg::analysis::AnalysisManager analysis_manager(*sdfg_);

    auto pipeline = sdfg::passes::normalization::loop_normalization();
    pipeline.run(builder, analysis_manager);
}

void PyStructuredSDFG::schedule(const std::string& target, const std::string& category, bool remote_tuning) {
    if (target == "none") {
        return;
    }

    sdfg::builder::StructuredSDFGBuilder builder(*sdfg_);
    sdfg::analysis::AnalysisManager analysis_manager(*sdfg_);

    std::vector<std::string> schedulers;

    if (remote_tuning) {
        throw std::runtime_error("Remote tuning is not yet supported in python.");
    }

    // CPU Opt Pipeline
    if (target == "sequential" || target == "openmp") {
        sdfg::passes::Pipeline dce = sdfg::passes::Pipeline::dead_code_elimination();
        sdfg::passes::DeadDataElimination dde;
        sdfg::passes::SymbolPropagation symbol_propagation_pass;
        symbol_propagation_pass.run(builder, analysis_manager);
        dde.run(builder, analysis_manager);
        dce.run(builder, analysis_manager);

        if (target == "openmp") {
            schedulers.push_back(target);
        }
        schedulers.push_back("highway");
    }
    // GPU Opt Pipeline
    else if (target == "cuda") {
        schedulers.push_back(target);
    } else if (target == "onnx") {
        sdfg::passes::ONNXLibraryNodeRewriterPass onnx_library_node_rewriter_pass;
        onnx_library_node_rewriter_pass.run(builder, analysis_manager);
    }
    sdfg::passes::scheduler::LoopSchedulingPass loop_scheduling_pass(schedulers, nullptr);
    loop_scheduling_pass.run(builder, analysis_manager);
}

std::string PyStructuredSDFG::compile(
    const std::string& output_folder,
    const std::string& target,
    const std::string& instrumentation_mode,
    bool capture_args
) const {
    fs::path build_path(output_folder);
    if (!fs::exists(build_path)) {
        fs::create_directories(build_path);
    }
    fs::path header_path = build_path / (sdfg_->name() + ".h");
    fs::path source_path = build_path / (sdfg_->name() + ".cpp");

    sdfg::analysis::AnalysisManager analysis_manager(*sdfg_);

    // Run expansion pass
    sdfg::passes::Pipeline expansion = sdfg::passes::Pipeline::expansion();
    sdfg::builder::StructuredSDFGBuilder builder_opt(*sdfg_);
    expansion.run(builder_opt, analysis_manager);

    // Instrumentation plan
    std::unique_ptr<sdfg::codegen::InstrumentationPlan> instrumentation_plan;
    if (instrumentation_mode.empty()) {
        instrumentation_plan = sdfg::codegen::InstrumentationPlan::none(*sdfg_);
    } else if (instrumentation_mode == "ols") {
        instrumentation_plan = sdfg::codegen::InstrumentationPlan::outermost_loops_plan(*sdfg_);
    } else {
        throw std::runtime_error("Unsupported instrumentation plan: " + instrumentation_mode);
    }

    // Argument capture plan
    std::unique_ptr<sdfg::codegen::ArgCapturePlan> arg_capture_plan;
    if (capture_args) {
        arg_capture_plan = sdfg::codegen::ArgCapturePlan::outermost_loops_plan(*sdfg_);
    } else {
        arg_capture_plan = sdfg::codegen::ArgCapturePlan::none(*sdfg_);
    }

    std::pair<std::filesystem::path, std::filesystem::path> lib_config = std::make_pair(build_path, header_path);
    std::shared_ptr<sdfg::codegen::CodeSnippetFactory> snippet_factory =
        std::make_shared<sdfg::codegen::CodeSnippetFactory>(&lib_config);
    sdfg::codegen::CPPCodeGenerator
        generator(*sdfg_, analysis_manager, *instrumentation_plan, *arg_capture_plan, snippet_factory);
    generator.generate();

    generator.as_source(header_path, source_path);

    // Write library snippets
    std::unordered_set<std::string> lib_files;
    for (auto& [name, snippet] : snippet_factory->snippets()) {
        if (snippet.is_as_file()) {
            auto p = build_path / (name + "." + snippet.extension());
            std::ofstream outfile_lib;
            if (lib_files.insert(p.string()).second) {
                outfile_lib.open(p, std::ios_base::out);
            } else {
                outfile_lib.open(p, std::ios_base::app);
            }
            if (!outfile_lib.is_open()) {
                throw std::runtime_error("Failed to open library file: " + p.string());
            }
            outfile_lib << snippet.stream().str() << std::endl;
            outfile_lib.close();
        }
    }

    // Find libraries relative to the module location
    Dl_info info;
    std::string package_path_str;
    std::string package_include_path_str;
    if (dladdr((void*) &_anchor, &info)) {
        fs::path lib_path = fs::canonical(info.dli_fname);
        fs::path package_path = lib_path.parent_path().parent_path();
        package_path_str = package_path.string();
        package_include_path_str = (package_path / "include").string();
    }

    bool has_highway = false;
    std::unordered_set<std::string> object_files;
    for (const auto& lib_file : lib_files) {
        std::filesystem::path lib_path(lib_file);
        std::string extension = lib_path.extension().string();
        if (extension == ".json") {
            continue;
        }

        std::string name = lib_path.stem().string();
        std::string object_file = build_path.string() + "/" + name + ".o";
        std::stringstream cmd;
        cmd << DOCC_CXX_COMPILER << " -c -fPIC -O3  -march=native -mtune=native -funroll-loops";
        if (!package_path_str.empty()) {
            cmd << " -L" << package_path_str;
            cmd << " -I" << package_include_path_str;
        }
#if defined(__APPLE__)
        cmd << " -I/opt/homebrew/include";
#endif
        if (target == "cuda") {
            cmd << " -x cuda --cuda-gpu-arch=sm_70 --cuda-path=/usr/local/cuda";
        }

        cmd << " " << lib_file;
        cmd << " -o " << object_file;
        if (name.starts_with("highway_")) {
            cmd << " -lhwy";
            has_highway = true;
        }
        cmd << " -lm";
        int ret = std::system(cmd.str().c_str());
        if (ret != 0) {
            throw std::runtime_error("Compilation failed: " + cmd.str());
        }
        object_files.insert(object_file);
    }

    // Compile
    {
        std::stringstream cmd;
        cmd << DOCC_CXX_COMPILER << " -c -fPIC -O3 -march=native -mtune=native -funroll-loops";
        if (!package_path_str.empty()) {
            cmd << " -L" << package_path_str;
            cmd << " -I" << package_include_path_str;
        }
        if (target == "cuda") {
            cmd << " -x cuda -lcuda";
        }
        cmd << " " << source_path.string();
        cmd << " -o " << (build_path / (sdfg_->name() + ".o")).string();
        int ret = std::system(cmd.str().c_str());
        if (ret != 0) {
            throw std::runtime_error("Compilation failed: " + cmd.str());
        }
        object_files.insert((build_path / (sdfg_->name() + ".o")).string());
    }

    // Link into shared library
    fs::path lib_path = build_path / ("lib" + sdfg_->name() + ".so");

    std::stringstream cmd;
#if defined(__APPLE__)
    cmd << DOCC_CXX_COMPILER << " -shared -Xpreprocessor -fopenmp -fPIC -O3";
    cmd << " -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include";
    cmd << " -L/opt/homebrew/lib";
#else
    cmd << DOCC_CXX_COMPILER << " -shared -fopenmp -fPIC -O3";
#endif
    if (!package_path_str.empty()) {
        cmd << " -L" << package_path_str;
        cmd << " -I" << package_include_path_str;
    }
    // cmd << " " << source_path.string();
    for (const auto& object_file : object_files) {
        cmd << " " << object_file;
    }
    if (has_highway) {
        cmd << " -lhwy";
    }
    cmd << " -ldaisy_rtl";
    cmd << " -larg_capture_io";
#if defined(__APPLE__)
    cmd << " -lomp";
    cmd << " -framework Accelerate";
#else
    cmd << " -lblas";
#endif
    cmd << " -lm";
    cmd << " -lstdc++";
    if (target == "cuda") {
        cmd << " /usr/local/cuda/lib64/libcudart.so";
        cmd << " /usr/local/cuda/lib64/libcublas.so";
    }
    if (target == "onnx") {
        cmd << " -L/usr/local/onnxruntime/lib";
        cmd << " -lonnxruntime";
        cmd << " -ldl"; // Required for dladdr()
    }
    cmd << " -o " << lib_path.string();


    int ret = std::system(cmd.str().c_str());
    if (ret != 0) {
        throw std::runtime_error("Compilation failed: " + cmd.str());
    }

    return lib_path.string();
}

std::string PyStructuredSDFG::metadata(const std::string& key) const {
    try {
        return sdfg_->metadata(key);
    } catch (const std::out_of_range&) {
        return "";
    }
}

pybind11::dict PyStructuredSDFG::loop_report() const {
    sdfg::builder::StructuredSDFGBuilder builder(*sdfg_);
    sdfg::analysis::AnalysisManager analysis_manager(*sdfg_);

    sdfg::codegen::LoopReport report_visitor(builder, analysis_manager);
    report_visitor.visit();

    pybind11::dict result;
    for (const auto& [key, value] : report_visitor.report()) {
        result[key.c_str()] = value;
    }

    return result;
}
