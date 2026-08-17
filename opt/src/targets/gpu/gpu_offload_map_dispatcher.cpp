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
#include "sdfg/targets/gpu/gpu_schedule_type.h"
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

symbolic::SymbolSet target_level_indvars(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, TargetLevel target_level
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);
    symbolic::SymbolSet indvars;
    for (const auto& loop : loops) {
        if (auto struc_loop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            if (struc_loop->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                if (cuda::ScheduleType_CUDA::target_level(struc_loop->schedule_type()) == target_level ||
                    rocm::ScheduleType_ROCM::target_level(struc_loop->schedule_type()) == target_level) {
                    indvars.insert(struc_loop->indvar());
                }
            }
        }
    }
    return indvars;
}

void get_nested_schedule_types(
    structured_control_flow::Map& node,
    analysis::AnalysisManager& analysis_manager,
    std::unordered_map<TargetLevel, ScheduleType>& output
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);
    for (const auto& loop : loops) {
        if (auto struc_loop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            if (struc_loop->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                output[ScheduleType_GPU::target_level(struc_loop->schedule_type())] = struc_loop->schedule_type();
            }
        }
    }
}

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
        if (!sdfg_.type(argument.first).storage_type().is_nv_symbol()) {
            arguments.push_back(argument.first);
        }
    }

    std::sort(arguments.begin(), arguments.end());
    std::vector<std::string> arguments_device;
    for (auto& argument : arguments) {
        auto& arg_type = sdfg_.type(argument);
        if (arg_type.storage_type().is_nv_generic()) {
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
            arguments_declaration.push_back(this->language_extension_.declaration(container, sdfg_.type(container)));
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
            block_size_x = ScheduleType_GPU::parallel_size(nested_schedule_types[TargetLevel::X_BLOCK]);
        }
        if (nested_schedule_types.find(TargetLevel::Y_BLOCK) != nested_schedule_types.end()) {
            block_size_y = ScheduleType_GPU::parallel_size(nested_schedule_types[TargetLevel::Y_BLOCK]);
        }
        if (nested_schedule_types.find(TargetLevel::Z_BLOCK) != nested_schedule_types.end()) {
            block_size_z = ScheduleType_GPU::parallel_size(nested_schedule_types[TargetLevel::Z_BLOCK]);
        }
        if (nested_schedule_types.find(TargetLevel::X_GRID) != nested_schedule_types.end()) {
            grid_size_x = ScheduleType_GPU::parallel_size(nested_schedule_types[TargetLevel::X_GRID]);
        }
        if (nested_schedule_types.find(TargetLevel::Y_GRID) != nested_schedule_types.end()) {
            grid_size_y = ScheduleType_GPU::parallel_size(nested_schedule_types[TargetLevel::Y_GRID]);
        }
        if (nested_schedule_types.find(TargetLevel::Z_GRID) != nested_schedule_types.end()) {
            grid_size_z = ScheduleType_GPU::parallel_size(nested_schedule_types[TargetLevel::Z_GRID]);
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

        auto& library_stream = library_snippet_factory.require(kernel_name, "cu", true).stream();

        library_stream << "#include " << library_snippet_factory.header_path().filename() << std::endl
                       << std::endl; // we expect the compiler-call to do this instead

        this->dispatch_kernel_preamble(library_stream, analysis_manager, kernel_name, arguments_declaration);

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
            if (local.starts_with("__daisy_cuda")) {
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
    // Boundary Conditions
    if (!ScheduleType_GPU::nested_sync(node_.schedule_type())) {
        // Guard on the flat thread id rather than the per-Map indvar so that
        // Maps with non-unit stride or non-zero init still get a correct OOB
        // check (the per-Map indvar = init + flat_id * stride and is
        // only well-defined when flat_id < num_iterations).
        std::string flat_id;
        switch (ScheduleType_GPU::dimension(node_.schedule_type())) {
            case CUDADimension::X:
                flat_id = "__daisy_cuda_indvar_x";
                break;
            case CUDADimension::Y:
                flat_id = "__daisy_cuda_indvar_y";
                break;
            case CUDADimension::Z:
                flat_id = "__daisy_cuda_indvar_z";
                break;
            default:
                flat_id = indvar->get_name();
                break;
        }
        library_stream << "if (" << flat_id << " < " << cuda_language_extension.expression(num_iterations) << ") {"
                       << std::endl;
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

    if (!ScheduleType_CUDA_deprecated::nested_sync(node_.schedule_type())) {
        library_stream.setIndent(library_stream.indent() - 4);
        library_stream << "}" << std::endl;
    }
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

    // Declare each per-Map indvar as a strided affine of the flat thread id:
    //   <map.indvar> = <map.init> + <thread_flat_id> * <map.stride>
    //
    // This lets the dispatcher consume Maps with arbitrary init / stride
    // (e.g. block-tiled outer loops produced by LoopTiling). The bound check
    // in dispatch_kernel_body() guards on the flat id against num_iterations,
    // so out-of-grid threads are skipped before any body access.
    auto x_maps = gpu::get_gpu_maps<ScheduleType_CUDA_deprecated>(node_, analysis_manager, CUDADimension::X);
    auto y_maps = gpu::get_gpu_maps<ScheduleType_CUDA_deprecated>(node_, analysis_manager, CUDADimension::Y);
    auto z_maps = gpu::get_gpu_maps<ScheduleType_CUDA_deprecated>(node_, analysis_manager, CUDADimension::Z);

    std::unordered_map<std::string, symbolic::Expression> indvars;

    auto emit_indvar = [&](structured_control_flow::Map* map, const std::string& flat_id_var) {
        symbolic::Expression value = symbolic::symbol(flat_id_var);
        auto stride = map->stride();
        if (!stride.is_null() && !symbolic::eq(stride, symbolic::one())) {
            value = symbolic::mul(value, stride);
        }
        auto init = map->init();
        if (!symbolic::eq(init, symbolic::zero())) {
            value = symbolic::add(init, value);
        }
        auto indvar_name = map->indvar()->get_name();
        auto it = indvars.find(indvar_name);
        if (it != indvars.end()) {
            if (!symbolic::eq(it->second, value)) {
                throw InvalidSDFGException(
                    "Conflicting expressions for Map #" + std::to_string(node_.element_id()) + " indvar " +
                    map->indvar()->get_name() + ": " + this->language_extension_.expression(it->second) + " vs " +
                    this->language_extension_.expression(value)
                );
            }
        } else {
            library_stream << "int " << indvar_name << " = " << this->language_extension_.expression(value) << ";"
                           << std::endl;
            indvars.emplace(indvar_name, value);
        }
    };

    for (auto* map : x_maps) {
        emit_indvar(map, indvar_x);
    }
    for (auto* map : y_maps) {
        emit_indvar(map, indvar_y);
    }
    for (auto* map : z_maps) {
        emit_indvar(map, indvar_z);
    }
}


} // namespace gpu
} // namespace sdfg
