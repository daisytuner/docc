#include "sdfg/targets/gpu/gpu_offload_map_dispatcher.h"

#include <iostream>
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
#include <unordered_map>
#include <unordered_set>


#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/element.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace gpu {

GPUOffloadMapDispatcher::GPUOffloadMapDispatcher(
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

bool GPUOffloadMapDispatcher::is_outermost_map(analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto ancestors = loop_analysis.ancestors(&node_);
    for (auto ancestor : ancestors) {
        if (auto loop = dyn_cast<StructuredLoop*>(ancestor)) {
            if (loop->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                return false;
            }
        }
    }
    return true;
}

void GPUOffloadMapDispatcher::dispatch_node(
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

    // filter indvar
    auto indvar = node_.indvar();

    std::vector<std::string> scope_variables_unfiltered(locals.begin(), locals.end());
    scope_variables_unfiltered.erase(
        std::remove(scope_variables_unfiltered.begin(), scope_variables_unfiltered.end(), indvar->get_name()),
        scope_variables_unfiltered.end()
    );
    std::vector<std::string> arguments;

    for (auto& argument : used_arguments) {
        auto storage = sdfg_.type(argument.first).storage_type();
        // Thread-index symbols and shared-memory scratch are declared inside the
        // kernel (the latter as a local of the enclosing kernel map), never passed
        // as kernel arguments.
        if (!storage.is_nv_symbol() && !storage.is_nv_shared()) {
            arguments.push_back(argument.first);
        }
    }

    std::sort(arguments.begin(), arguments.end());
    std::vector<std::string> arguments_device;
    for (auto& argument : arguments) {
        auto& arg_type = sdfg_.type(argument);
        if (this->is_device_pointer_storage(arg_type.storage_type())) {
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
            const std::string decl_name = this->is_device_pointer_storage(arg_type.storage_type())
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


        this->dispatch_kernel_call(
            main_stream,
            kernel_name,
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

        auto& library_stream =
            library_snippet_factory.require(kernel_name, this->kernel_file_extension(), true).stream();

        library_stream << "#include " << library_snippet_factory.header_path().filename() << std::endl
                       << std::endl; // we expect the compiler-call to do this instead

        this->dispatch_kernel_preamble(library_stream, analysis_manager, kernel_name, arguments_declaration);

        // Every device-pointer argument is a full cudaMalloc/hipMalloc allocation,
        // which is guaranteed >=256-byte aligned. Asserting 16-byte alignment lets
        // clang's load-store vectorizer widen contiguous copies to 128-bit
        // (LDG/STG.128); decltype keeps it agnostic to element type / constness.
        for (auto& container : arguments) {
            if (this->is_device_pointer_storage(sdfg_.type(container).storage_type())) {
                library_stream << container << " = reinterpret_cast<decltype(" << container
                               << ")>(__builtin_assume_aligned(" << container << ", 16));" << std::endl;
            }
        }

        this->dispatch_kernel_body(library_snippet_factory, library_stream, node_.indvar(), scope_variables, num_iters);

        library_stream.setIndent(library_stream.indent() - 4);
        library_stream << "}" << std::endl;
    } else {
        this->dispatch_kernel_body(library_snippet_factory, main_stream, node_.indvar(), scope_variables, num_iters);
    }
};

void GPUOffloadMapDispatcher::dispatch_header(
    codegen::PrettyPrinter& globals_stream,
    const std::string& kernel_name,
    std::vector<std::string>& arguments_declaration
) {
    globals_stream << "__global__ void " << kernel_name << "(";
    globals_stream << helpers::join(arguments_declaration, ", ");
    globals_stream << ")";
}

void GPUOffloadMapDispatcher::dispatch_kernel_body(
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& library_stream,
    symbolic::Symbol indvar,
    std::vector<std::string>& scope_variables,
    symbolic::Expression& num_iterations
) {
    codegen::LanguageExtension& kernel_language_extension = create_kernel_language_extension();
    if (is_outermost_map(analysis_manager_)) {
        // Declare and optionally allocate scope variables
        for (auto& local : scope_variables) {
            if (local.starts_with("__daisy_gpu")) {
                continue;
            }
            std::string val = kernel_language_extension.declaration(local, sdfg_.type(local), false, true);
            if (!val.empty()) {
                library_stream << val;
                library_stream << ";" << std::endl;
            }
            auto& type = sdfg_.type(local);
            if (type.storage_type().allocation() == types::StorageType::AllocationType::Managed) {
                library_stream << local << " = ";
                library_stream << "malloc("
                               << kernel_language_extension.expression(type.storage_type().allocation_size()) << ")";
                library_stream << ";" << std::endl;
            }
        }
    }
    // generate coverage loop
    TargetLevel target_level = gpu::ScheduleType_GPU_Offload::target_level(node_.schedule_type());
    std::string coverage_loop_var = "__daisy_gpu_coverage_loop_" + gpu::to_string(target_level);
    std::string size = kernel_language_extension.expression(node_.num_iterations());
    if (target_level == TargetLevel::WARP) {
        std::string warp_dim = kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size())
        );
        library_stream << "uint32_t num_warps = (" << kernel_language_extension.expression(symbolic::blockDim_x())
                       << " + " << warp_dim << " - 1) / " << warp_dim << ";" << std::endl;
        library_stream << "uint32_t warp_id = " << kernel_language_extension.expression(symbolic::threadIdx_x())
                       << " / "
                       << kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size()))
                       << ";" << std::endl;
        library_stream << "uint32_t lane = " << kernel_language_extension.expression(symbolic::threadIdx_x()) << " & ("
                       << kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size()))
                       << " - 1);" << std::endl;
    }

    std::string coverage_dim = kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size())
    );
    // For the WARP level each thread iterates sequentially over the warp-level
    // iteration space (the cross-lane reduction is performed by the reduce
    // dispatcher via __shfl_xor_sync over the enclosing X_BLOCK lanes), so the
    // coverage loop must run once per iteration rather than once per warp_size.
    std::string coverage_count_dim = (target_level == TargetLevel::WARP) ? std::string("1") : coverage_dim;
    // Cast the ceil-div to int: blockDim/gridDim are unsigned, and CUDA 12.9's max()
    // overload set makes max(1, <unsigned>) ambiguous under clang-cuda.
    library_stream << "for (int " << coverage_loop_var << " = 0; " << coverage_loop_var << " < "
                   << "max(1, (int)((" << size << " + " << coverage_count_dim << " - 1) / " << coverage_count_dim
                   << ")); " << coverage_loop_var << "++) {" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    if (target_level == TargetLevel::WARP) {
        std::string indvar_name = indvar->get_name();
        auto x_block_parent = find_x_block_owning_warp_level(node_, analysis_manager_);
        if (!x_block_parent) {
            throw InvalidSDFGException("WARP level map must be nested within an X_BLOCK level map");
        }

        // Sequential per-thread iteration over the warp-level space.
        library_stream << "size_t " << indvar_name << " = " << kernel_language_extension.expression(node_.init())
                       << " + " << coverage_loop_var << " * " << kernel_language_extension.expression(node_.stride())
                       << ";" << std::endl;
    } else {
        // 0-based parallel index across this dimension: `coverage` sweeps of `dim`
        // units plus this thread/block's index. The map's induction variable is
        // then init + stride * parallel_index, so the stride applies to BOTH the
        // coverage and the index terms (tiled offload maps have stride != 1).
        std::string dim_expr = kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size())
        );
        std::string idx_expr = kernel_language_extension.expression(get_target_level_idx(target_level));
        std::string parallel_index = coverage_loop_var + " * " + dim_expr + " + " + idx_expr;

        std::string offset;
        if (target_level == TargetLevel::X_BLOCK && nested_warp_dim(node_, analysis_manager_)) {
            // Warp handles the sub-stride; the block index is used directly.
            offset = "(" + parallel_index + ")";
        } else {
            offset = kernel_language_extension.expression(node_.stride()) + " * (" + parallel_index + ")";
        }

        library_stream << "size_t " << indvar->get_name() << " = " << kernel_language_extension.expression(node_.init())
                       << " + " << offset << ";" << std::endl;
    }


    // Boundary Conditions
    if (!gpu::ScheduleType_GPU_Offload::nested_sync(node_.schedule_type())) {
        library_stream << "if (" << kernel_language_extension.expression(node_.condition()) << ") {" << std::endl;
        library_stream.setIndent(library_stream.indent() + 4);
    }

    // Body
    codegen::SequenceDispatcher dispatcher(
        kernel_language_extension, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_
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

    if (!gpu::ScheduleType_GPU_Offload::nested_sync(node_.schedule_type())) {
        library_stream.setIndent(library_stream.indent() - 4);
        library_stream << "}" << std::endl;
    }

    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;
}

void GPUOffloadMapDispatcher::dispatch_kernel_preamble(
    codegen::PrettyPrinter& library_stream,
    analysis::AnalysisManager& analysis_manager,
    const std::string& kernel_name,
    std::vector<std::string>& arguments_declaration
) {
    // Kernel Header
    dispatch_header(library_stream, kernel_name, arguments_declaration);

    // Kernel Body
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);
}


} // namespace gpu
} // namespace sdfg
