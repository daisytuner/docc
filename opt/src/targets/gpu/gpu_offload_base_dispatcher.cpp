#include "sdfg/targets/gpu/gpu_offload_base_dispatcher.h"

#include <string>
#include <unordered_map>
#include <vector>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/analysis/users.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/structured_control_flow/control_flow_node.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/type.h>

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"

namespace sdfg {
namespace gpu {

GPUOffloadBaseDispatcher::GPUOffloadBaseDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::StructuredLoop& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan,
    std::unique_ptr<GPUOffloadDispatcherStrategy> strategy
)
    : codegen::NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node), strategy_(std::move(strategy)) {}

bool GPUOffloadBaseDispatcher::is_outermost_map(analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ancestors = loop_analysis.ancestors(&node_);
    for (auto ancestor : ancestors) {
        if (auto loop = dyn_cast<structured_control_flow::StructuredLoop*>(ancestor)) {
            if (loop->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                return false;
            }
        }
    }
    return true;
}

void GPUOffloadBaseDispatcher::emit_lib_dependency_includes(
    codegen::PrettyPrinter& kernel_header_stream, codegen::NestedCodeSnippetFactory& nested_snippet_factory
) {
    std::vector<std::string> includes;
    for (auto* dep : nested_snippet_factory.get_used_lib_dependencies()) {
        dep->enumerate_includes(includes);
    }
    for (auto& include : includes) {
        kernel_header_stream << "#include <" << include << ">" << std::endl;
    }
}

void GPUOffloadBaseDispatcher::dispatch_node(
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

    this->validate_before_dispatch(analysis_manager);

    // filter indvar
    auto indvar = node_.indvar();

    std::vector<std::string> scope_variables_unfiltered(locals.begin(), locals.end());
    scope_variables_unfiltered.erase(
        std::remove(scope_variables_unfiltered.begin(), scope_variables_unfiltered.end(), indvar->get_name()),
        scope_variables_unfiltered.end()
    );
    std::vector<std::string> arguments;

    for (auto& argument : used_arguments) {
        auto& type = sdfg_.type(argument.first);
        auto storage = type.storage_type();
        if (type.type_id() == types::TypeID::Array) {
            // Shared memory or register arrays of the kernel.
            // These arrays are not passed as kernel arguments.
            assert((storage.is_nv_shared() || storage.is_cpu_stack()) && "Array must be in shared memory or registers");
            continue;
        } else if (storage.is_nv_symbol()) {
            // Thread-index symbols are declared inside the kernel, never passed as kernel arguments.
            continue;
        }
        arguments.push_back(argument.first);
    }

    std::sort(arguments.begin(), arguments.end());
    std::vector<std::string> arguments_device;
    for (auto& argument : arguments) {
        auto& arg_type = sdfg_.type(argument);
        if (strategy_->is_device_pointer_storage(arg_type.storage_type())) {
            arguments_device.push_back(argument);
        } else if (arg_type.type_id() == types::TypeID::Scalar) {
            arguments_device.push_back(argument);
        } else {
            throw InvalidSDFGException("Argument " + argument + " is not a scalar or device pointer");
        }
    }

    std::vector<std::string> scope_variables;

    auto x_grids = target_level_indvars(node_, analysis_manager, TargetLevel::X_GRID);
    auto y_grids = target_level_indvars(node_, analysis_manager, TargetLevel::Y_GRID);
    auto z_grids = target_level_indvars(node_, analysis_manager, TargetLevel::Z_GRID);

    auto x_blocks = target_level_indvars(node_, analysis_manager, TargetLevel::X_BLOCK);
    auto y_blocks = target_level_indvars(node_, analysis_manager, TargetLevel::Y_BLOCK);
    auto z_blocks = target_level_indvars(node_, analysis_manager, TargetLevel::Z_BLOCK);

    auto warps = target_level_indvars(node_, analysis_manager, TargetLevel::WARP);

    for (auto& var : scope_variables_unfiltered) {
        if (x_grids.find(symbolic::symbol(var)) == x_grids.end() &&
            y_grids.find(symbolic::symbol(var)) == y_grids.end() &&
            z_grids.find(symbolic::symbol(var)) == z_grids.end() &&
            x_blocks.find(symbolic::symbol(var)) == x_blocks.end() &&
            y_blocks.find(symbolic::symbol(var)) == y_blocks.end() &&
            z_blocks.find(symbolic::symbol(var)) == z_blocks.end() &&
            warps.find(symbolic::symbol(var)) == warps.end()) {
            scope_variables.push_back(var);
        }
    }

    std::sort(scope_variables.begin(), scope_variables.end());

    symbolic::Expression num_iters = node_.num_iterations();

    if (is_outermost_map(analysis_manager)) {
        // Arguments Declaration
        std::vector<std::string> arguments_declaration;
        for (auto& container : arguments) {
            const auto& arg_type = sdfg_.type(container);
            // Distinct device buffers never alias: mark pointer params __restrict__ so clang's
            // load-store vectorizer can widen contiguous copies (it bails on possible aliasing).
            const std::string decl_name = strategy_->is_device_pointer_storage(arg_type.storage_type())
                                              ? "__restrict__ " + container
                                              : container;
            arguments_declaration.push_back(this->language_extension_.declaration(decl_name, arg_type));
        }

        std::unordered_map<TargetLevel, ScheduleType> nested_schedule_types;
        get_nested_schedule_types(node_, analysis_manager, nested_schedule_types);

        symbolic::Expression block_size_x = symbolic::one();
        symbolic::Expression block_size_y = symbolic::one();
        symbolic::Expression block_size_z = symbolic::one();
        symbolic::Expression grid_size_x = symbolic::one();
        symbolic::Expression grid_size_y = symbolic::one();
        symbolic::Expression grid_size_z = symbolic::one();

        if (nested_schedule_types.find(TargetLevel::X_BLOCK) != nested_schedule_types.end()) {
            block_size_x = gpu::ScheduleType_GPU_Offload::parallel_size(nested_schedule_types.at(TargetLevel::X_BLOCK));
        }
        if (nested_schedule_types.find(TargetLevel::Y_BLOCK) != nested_schedule_types.end()) {
            block_size_y = gpu::ScheduleType_GPU_Offload::parallel_size(nested_schedule_types.at(TargetLevel::Y_BLOCK));
        }
        if (nested_schedule_types.find(TargetLevel::Z_BLOCK) != nested_schedule_types.end()) {
            block_size_z = gpu::ScheduleType_GPU_Offload::parallel_size(nested_schedule_types.at(TargetLevel::Z_BLOCK));
        }
        if (nested_schedule_types.find(TargetLevel::X_GRID) != nested_schedule_types.end()) {
            grid_size_x = gpu::ScheduleType_GPU_Offload::parallel_size(nested_schedule_types.at(TargetLevel::X_GRID));
        }
        if (nested_schedule_types.find(TargetLevel::Y_GRID) != nested_schedule_types.end()) {
            grid_size_y = gpu::ScheduleType_GPU_Offload::parallel_size(nested_schedule_types.at(TargetLevel::Y_GRID));
        }
        if (nested_schedule_types.find(TargetLevel::Z_GRID) != nested_schedule_types.end()) {
            grid_size_z = gpu::ScheduleType_GPU_Offload::parallel_size(nested_schedule_types.at(TargetLevel::Z_GRID));
        }


        std::string kernel_name = "kernel_" + sdfg_.name() + "_" + std::to_string(node_.element_id());


        strategy_->dispatch_kernel_call(
            main_stream,
            kernel_name,
            language_extension_,
            grid_size_x,
            grid_size_y,
            grid_size_z,
            block_size_x,
            block_size_y,
            block_size_z,
            arguments_device
        );

        library_snippet_factory.add_global("#include <cstdio>");
        // Kernel Declaration
        this->dispatch_header(globals_stream, kernel_name, arguments_declaration);
        globals_stream << ";" << std::endl;

        auto& kernel_stream =
            library_snippet_factory.require(kernel_name, strategy_->kernel_file_extension(), true).stream();
        auto& kernel_header_snippet =
            library_snippet_factory.require(kernel_name + "_inc", strategy_->kernel_header_file_extension(), true);

        auto& kernel_header_stream = kernel_header_snippet.stream();
        auto kernel_header_path = library_snippet_factory.output_path() / kernel_header_snippet.filename();
        kernel_stream << "#include " << kernel_header_path.filename() << std::endl << std::endl << std::endl;
        kernel_header_stream << "#include " << library_snippet_factory.header_path().filename()
                             << std::endl; // we
                                           // expect
                                           // the
                                           // compiler-call
                                           // to do
                                           // this
                                           // instead

        std::pair<std::filesystem::path, std::filesystem::path> nested_config{
            library_snippet_factory.output_path(), kernel_header_path
        };
        auto nested_snippet_factory = codegen::NestedCodeSnippetFactory(&nested_config);

        this->dispatch_kernel_preamble(kernel_stream, kernel_name, arguments_declaration);

        // Every device-pointer argument is a full cudaMalloc/hipMalloc allocation,
        // which is guaranteed >=256-byte aligned. Asserting 16-byte alignment lets
        // clang's load-store vectorizer widen contiguous copies to 128-bit
        // (LDG/STG.128); decltype keeps it agnostic to element type / constness.
        for (auto& container : arguments) {
            if (strategy_->is_device_pointer_storage(sdfg_.type(container).storage_type())) {
                kernel_stream << container << " = reinterpret_cast<decltype(" << container
                              << ")>(__builtin_assume_aligned(" << container << ", 16));" << std::endl;
            }
        }

        this->dispatch_kernel_body(
            nested_snippet_factory, kernel_stream, kernel_header_stream, node_.indvar(), scope_variables, num_iters
        );

        kernel_stream.setIndent(kernel_stream.indent() - 4);
        kernel_stream << "}" << std::endl;

        strategy_->emit_target_header_declarations(kernel_header_stream);
        this->emit_lib_dependency_includes(kernel_header_stream, nested_snippet_factory);
    } else {
        this->dispatch_kernel_body(
            dynamic_cast<codegen::NestedCodeSnippetFactory&>(library_snippet_factory),
            main_stream,
            globals_stream,
            node_.indvar(),
            scope_variables,
            num_iters
        );
    }
}

void GPUOffloadBaseDispatcher::dispatch_header(
    codegen::PrettyPrinter& globals_stream,
    const std::string& kernel_name,
    std::vector<std::string>& arguments_declaration
) {
    globals_stream << "__global__ void " << kernel_name << "(";
    globals_stream << helpers::join(arguments_declaration, ", ");
    globals_stream << ")";
}

void GPUOffloadBaseDispatcher::dispatch_kernel_preamble(
    codegen::PrettyPrinter& library_stream,
    const std::string& kernel_name,
    std::vector<std::string>& arguments_declaration
) {
    // Kernel Header
    dispatch_header(library_stream, kernel_name, arguments_declaration);

    // Kernel Body
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);
}

codegen::InstrumentationInfo GPUOffloadBaseDispatcher::instrumentation_info() const {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    analysis::LoopInfo loop_info = loop_analysis.loop_info(&node_);

    std::unordered_map<std::string, std::string> metrics;
    return codegen::InstrumentationInfo(
        node_.element_id(),
        node_.element_type(),
        strategy_->get_instrumentation_kernel_target_type(),
        codegen::InstrumentationEventType::CUDA,
        loop_info,
        metrics
    );
};

} // namespace gpu
} // namespace sdfg
