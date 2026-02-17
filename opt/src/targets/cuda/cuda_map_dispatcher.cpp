#include "sdfg/targets/cuda/cuda_map_dispatcher.h"

#include "sdfg/targets/cuda/cuda.h"

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/assumptions_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/builder/structured_sdfg_builder.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/serializer/json_serializer.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/type.h>
#include <sdfg/visitor/structured_sdfg_visitor.h>
#include <string>
#include <unordered_set>


#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/codegen/language_extensions/cuda_language_extension.h"

namespace sdfg {
namespace cuda {

CUDAMapDispatcher::CUDAMapDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Map& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : codegen::NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

void CUDAMapDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    // Mark written locals as private
    analysis::AnalysisManager analysis_manager(sdfg_);
    auto& users = analysis_manager.get<analysis::Users>();
    analysis::UsersView body_users(users, node_.root());
    analysis::ArgumentsAnalysis& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();

    auto& used_arguments = arguments_analysis.arguments(analysis_manager, node_);
    auto& locals = arguments_analysis.locals(analysis_manager, node_);
    auto& argument_sizes = arguments_analysis.argument_sizes(analysis_manager, node_, true);

    // filter indvar
    auto indvar = node_.indvar();

    std::vector<std::string> scope_variables_unfiltered(locals.begin(), locals.end());
    scope_variables_unfiltered.erase(
        std::remove(scope_variables_unfiltered.begin(), scope_variables_unfiltered.end(), indvar->get_name()),
        scope_variables_unfiltered.end()
    );
    std::vector<std::string> arguments;

    for (auto& argument : used_arguments) {
        if (!sdfg_.type(argument.first).storage_type().is_nv_symbol()) {
            arguments.push_back(argument.first);
        }
    }

    std::sort(arguments.begin(), arguments.end());
    std::vector<std::string> arguments_device;
    for (auto& argument : arguments) {
        if (argument.starts_with(CUDA_DEVICE_PREFIX)) {
            arguments_device.push_back(argument);
        } else if (sdfg_.type(argument).type_id() == types::TypeID::Scalar) {
            arguments_device.push_back(argument);
        } else {
            throw InvalidSDFGException("Argument " + argument + " is not a scalar or device pointer");
        }
    }

    std::vector<std::string> scope_variables;

    auto x_vars = gpu::get_gpu_indvars<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::X);
    auto y_vars = gpu::get_gpu_indvars<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::Y);
    auto z_vars = gpu::get_gpu_indvars<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::Z);

    for (auto& var : scope_variables_unfiltered) {
        if (x_vars.find(symbolic::symbol(var)) == x_vars.end() && y_vars.find(symbolic::symbol(var)) == y_vars.end() &&
            z_vars.find(symbolic::symbol(var)) == z_vars.end()) {
            scope_variables.push_back(var);
        }
    }

    std::sort(scope_variables.begin(), scope_variables.end());

    // Arguments Declaration
    std::vector<std::string> arguments_declaration;
    for (auto& container : arguments) {
        arguments_declaration.push_back(this->language_extension_.declaration(container, sdfg_.type(container)));
    }

    auto block_size_x = gpu::find_nested_gpu_blocksize<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::X);
    auto block_size_y = gpu::find_nested_gpu_blocksize<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::Y);
    auto block_size_z = gpu::find_nested_gpu_blocksize<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::Z);
    auto num_iters_x = gpu::find_nested_gpu_iterations<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::X);
    auto num_iters_y = gpu::find_nested_gpu_iterations<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::Y);
    auto num_iters_z = gpu::find_nested_gpu_iterations<ScheduleType_CUDA>(node_, analysis_manager, CUDADimension::Z);

    symbolic::Expression num_iters;
    if (CUDADimension::X == ScheduleType_CUDA::dimension(node_.schedule_type())) {
        num_iters = num_iters_x;
    } else if (CUDADimension::Y == ScheduleType_CUDA::dimension(node_.schedule_type())) {
        num_iters = num_iters_y;
    } else if (CUDADimension::Z == ScheduleType_CUDA::dimension(node_.schedule_type())) {
        num_iters = num_iters_z;
    } else {
        throw InvalidSDFGException("Invalid CUDA dimension");
    }

    // Block sizes
    symbolic::Expression num_blocks_x =
        symbolic::max(symbolic::divide_ceil(num_iters_x, block_size_x), symbolic::one());
    symbolic::Expression num_blocks_y =
        symbolic::max(symbolic::divide_ceil(num_iters_y, block_size_y), symbolic::one());
    symbolic::Expression num_blocks_z =
        symbolic::max(symbolic::divide_ceil(num_iters_z, block_size_z), symbolic::one());

    std::string kernel_name = "kernel_" + sdfg_.name() + "_" + std::to_string(node_.element_id());

    if (gpu::is_outermost_gpu_map<ScheduleType_CUDA>(node_, analysis_manager)) {
        this->dispatch_kernel_call(
            main_stream,
            kernel_name,
            num_blocks_x,
            num_blocks_y,
            num_blocks_z,
            block_size_x,
            block_size_y,
            block_size_z,
            arguments_device
        );

        globals_stream << "#include <cstdio>" << std::endl;
        // Kernel Declaration
        this->dispatch_header(globals_stream, kernel_name, arguments_declaration);
        globals_stream << ";" << std::endl;

        auto& library_stream = library_snippet_factory.require(kernel_name, "cu", true).stream();

        library_stream << "#include " << library_snippet_factory.header_path().filename() << std::endl
                       << std::endl; // we expect the compiler-call to do this instead

        this->dispatch_kernel_preamble(
            library_stream, analysis_manager, kernel_name, x_vars, y_vars, z_vars, arguments_declaration
        );

        this->dispatch_kernel_body(library_snippet_factory, library_stream, node_.indvar(), scope_variables, num_iters);

        library_stream.setIndent(library_stream.indent() - 4);
        library_stream << "}" << std::endl;
    } else {
        this->dispatch_kernel_body(library_snippet_factory, main_stream, node_.indvar(), scope_variables, num_iters);
    }
};

void CUDAMapDispatcher::dispatch_header(
    codegen::PrettyPrinter& globals_stream,
    const std::string& kernel_name,
    std::vector<std::string>& arguments_declaration
) {
    globals_stream << "__global__ void " << kernel_name << "(";
    globals_stream << helpers::join(arguments_declaration, ", ");
    globals_stream << ")";
}

void CUDAMapDispatcher::dispatch_kernel_body(
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& library_stream,
    symbolic::Symbol indvar,
    std::vector<std::string>& scope_variables,
    symbolic::Expression& num_iterations
) {
    codegen::CUDALanguageExtension cuda_language_extension(sdfg_);
    if (gpu::is_outermost_gpu_map<ScheduleType_CUDA>(node_, analysis_manager_)) {
        // Declare and optionally allocate scope variables
        for (auto& local : scope_variables) {
            if (local.starts_with("__daisy_cuda")) {
                continue;
            }
            std::string val = cuda_language_extension.declaration(local, sdfg_.type(local), false, true);
            if (!val.empty()) {
                library_stream << val;
                library_stream << ";" << std::endl;
            }
            auto& type = sdfg_.type(local);
            if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed) {
                library_stream << local << " = ";
                library_stream << "malloc(" << cuda_language_extension.expression(type.storage_type().allocation_size())
                               << ")";
                library_stream << ";" << std::endl;
            }
        }
    }
    // Boundary Conditions
    if (!ScheduleType_CUDA::nested_sync(node_.schedule_type())) {
        library_stream << "if (" << indvar->get_name() << " < " << cuda_language_extension.expression(num_iterations)
                       << ") {" << std::endl;
        library_stream.setIndent(library_stream.indent() + 4);
    }

    // Body
    codegen::SequenceDispatcher dispatcher(
        cuda_language_extension, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_
    );
    dispatcher.dispatch(library_stream, library_stream, library_snippet_factory);

    // Free managed scope variables
    for (auto& local : scope_variables) {
        auto& type = sdfg_.type(local);
        if (type.storage_type().deallocation() == types::StorageType::AllocationType::Managed) {
            library_stream << "free(" << local << ")";
            library_stream << ";" << std::endl;
        }
    }

    if (!ScheduleType_CUDA::nested_sync(node_.schedule_type())) {
        library_stream.setIndent(library_stream.indent() - 4);
        library_stream << "}" << std::endl;
    }
}

void CUDAMapDispatcher::dispatch_kernel_call(
    codegen::PrettyPrinter& main_stream,
    const std::string& kernel_name,
    symbolic::Expression& num_blocks_x,
    symbolic::Expression& num_blocks_y,
    symbolic::Expression& num_blocks_z,
    symbolic::Expression& block_size_x,
    symbolic::Expression& block_size_y,
    symbolic::Expression& block_size_z,
    std::vector<std::string>& arguments_device
) {
    main_stream << "{" << std::endl;
    main_stream.setIndent(main_stream.indent() + 4);

    // Kernel launch
    main_stream << kernel_name << "<<<";
    main_stream << "dim3((int)(" << this->language_extension_.expression(num_blocks_x) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(num_blocks_y) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(num_blocks_z) << ")), ";
    main_stream << "dim3((int)(" << this->language_extension_.expression(block_size_x) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(block_size_y) << "), ";
    main_stream << "(int)(" << this->language_extension_.expression(block_size_z) << "))";
    main_stream << ">>>";
    main_stream << "(";
    main_stream << helpers::join(arguments_device, ", ");
    main_stream << ")";
    main_stream << ";" << std::endl;

    // Synchronize
    check_cuda_kernel_launch_errors(main_stream, this->language_extension_);

    main_stream.setIndent(main_stream.indent() - 4);
    main_stream << "}" << std::endl;
}

void CUDAMapDispatcher::dispatch_kernel_preamble(
    codegen::PrettyPrinter& library_stream,
    analysis::AnalysisManager& analysis_manager,
    const std::string& kernel_name,
    symbolic::SymbolSet& x_vars,
    symbolic::SymbolSet& y_vars,
    symbolic::SymbolSet& z_vars,
    std::vector<std::string>& arguments_declaration
) {
    // Kernel Header
    dispatch_header(library_stream, kernel_name, arguments_declaration);

    // Kernel Body
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    std::string indvar_x = "__daisy_cuda_indvar_x";
    std::string indvar_y = "__daisy_cuda_indvar_y";
    std::string indvar_z = "__daisy_cuda_indvar_z";

    std::string thread_idx_x = "__daisy_cuda_thread_idx_x";
    std::string thread_idx_y = "__daisy_cuda_thread_idx_y";
    std::string thread_idx_z = "__daisy_cuda_thread_idx_z";

    // Declare all indvars in the kernel
    symbolic::Expression gpu_thread_idx_x = symbolic::threadIdx_x();
    library_stream << "int " << thread_idx_x << " = " << this->language_extension_.expression(gpu_thread_idx_x) << ";"
                   << std::endl;
    symbolic::Expression gpu_indvar_x =
        symbolic::add(symbolic::threadIdx_x(), symbolic::mul(symbolic::blockIdx_x(), symbolic::blockDim_x()));
    library_stream << "int " << indvar_x << " = " << this->language_extension_.expression(gpu_indvar_x) << ";"
                   << std::endl;

    symbolic::Expression gpu_thread_idx_y = symbolic::threadIdx_y();
    library_stream << "int " << thread_idx_y << " = " << this->language_extension_.expression(gpu_thread_idx_y) << ";"
                   << std::endl;
    symbolic::Expression gpu_indvar_y =
        symbolic::add(symbolic::threadIdx_y(), symbolic::mul(symbolic::blockIdx_y(), symbolic::blockDim_y()));
    library_stream << "int " << indvar_y << " = " << this->language_extension_.expression(gpu_indvar_y) << ";"
                   << std::endl;

    symbolic::Expression gpu_thread_idx_z = symbolic::threadIdx_z();
    library_stream << "int " << thread_idx_z << " = " << this->language_extension_.expression(gpu_thread_idx_z) << ";"
                   << std::endl;
    symbolic::Expression gpu_indvar_z =
        symbolic::add(symbolic::threadIdx_z(), symbolic::mul(symbolic::blockIdx_z(), symbolic::blockDim_z()));
    library_stream << "int " << indvar_z << " = " << this->language_extension_.expression(gpu_indvar_z) << ";"
                   << std::endl;

    // Declare all other indvars in the kernel
    for (auto& var : x_vars) {
        library_stream << "int " << var->get_name() << " = " << indvar_x << ";" << std::endl;
    }

    for (auto& var : y_vars) {
        library_stream << "int " << var->get_name() << " = " << indvar_y << ";" << std::endl;
    }

    for (auto& var : z_vars) {
        library_stream << "int " << var->get_name() << " = " << indvar_z << ";" << std::endl;
    }
}

codegen::InstrumentationInfo CUDAMapDispatcher::instrumentation_info() const {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    analysis::LoopInfo loop_info = loop_analysis.loop_info(&node_);

    // Perform FlopAnalysis
    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    auto flop = flop_analysis.get_if_available_for_codegen(&node_);
    if (!flop.is_null()) {
        std::string flop_str = language_extension_.expression(flop);
        metrics.insert({"flop", flop_str});
    }

    return codegen::InstrumentationInfo(node_.element_id(), codegen::ElementType_Map, TargetType_CUDA, loop_info, metrics);
};

} // namespace cuda
} // namespace sdfg
