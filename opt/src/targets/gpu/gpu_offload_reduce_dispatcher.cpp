#include "sdfg/targets/gpu/gpu_offload_reduce_dispatcher.h"

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

#include <algorithm>
#include <list>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/types/pointer.h>
#include <sdfg/types/scalar.h>

namespace sdfg {
namespace gpu {

namespace {

using structured_control_flow::ReductionOperation;

std::string op_tag(ReductionOperation op) {
    switch (op) {
        case ReductionOperation::Add:
            return "add";
        case ReductionOperation::Mul:
            return "mul";
        case ReductionOperation::Min:
            return "min";
        case ReductionOperation::Max:
            return "max";
    }
    throw InvalidSDFGException("GPUOffloadReduceDispatcher: unknown reduction operation");
}

// Identity element of the operator for the given primitive type, as a C literal.
std::string identity_literal(ReductionOperation op, types::PrimitiveType prim) {
    if (op == ReductionOperation::Add) {
        return "0";
    }
    if (op == ReductionOperation::Mul) {
        return "1";
    }
    if (types::is_floating_point(prim)) {
        return op == ReductionOperation::Min ? "INFINITY" : "-INFINITY";
    }

    const size_t width = types::bit_width(prim);
    const bool is_unsigned = types::is_unsigned(prim);
    if (op == ReductionOperation::Min) {
        if (is_unsigned) {
            if (width == 32) return "UINT32_MAX";
            if (width == 64) return "UINT64_MAX";
        } else {
            if (width == 32) return "INT32_MAX";
            if (width == 64) return "INT64_MAX";
        }
    } else {
        if (is_unsigned) {
            return "0";
        }
        if (width == 32) return "INT32_MIN";
        if (width == 64) return "INT64_MIN";
    }
    throw InvalidSDFGException("GPUOffloadReduceDispatcher: unsupported integer width for min/max reduction");
}

// `cur OP val` as a C expression string, header-free (ternaries for min/max).
std::string combine_expr(ReductionOperation op, const std::string& a, const std::string& b) {
    switch (op) {
        case ReductionOperation::Add:
            return "(" + a + ") + (" + b + ")";
        case ReductionOperation::Mul:
            return "(" + a + ") * (" + b + ")";
        case ReductionOperation::Min:
            return "((" + a + ") < (" + b + ") ? (" + a + ") : (" + b + "))";
        case ReductionOperation::Max:
            return "((" + a + ") < (" + b + ") ? (" + b + ") : (" + a + "))";
    }
    throw InvalidSDFGException("GPUOffloadReduceDispatcher: unknown reduction operation");
}

// Scalar element type of a reduction accumulator (device pointer to scalar, or scalar).
types::PrimitiveType accumulator_primitive(const StructuredSDFG& sdfg, const std::string& container) {
    auto& type = sdfg.type(container);
    if (auto* ptr = dynamic_cast<const types::Pointer*>(&type)) {
        if (!ptr->has_pointee_type()) {
            throw InvalidSDFGException(
                "GPUOffloadReduceDispatcher: reduction accumulator '" + container + "' has no pointee type"
            );
        }
        if (auto* scalar = dynamic_cast<const types::Scalar*>(&ptr->pointee_type())) {
            return scalar->primitive_type();
        }
        throw InvalidSDFGException(
            "GPUOffloadReduceDispatcher: reduction accumulator '" + container + "' must point to a scalar"
        );
    }
    if (auto* scalar = dynamic_cast<const types::Scalar*>(&type)) {
        return scalar->primitive_type();
    }
    throw InvalidSDFGException(
        "GPUOffloadReduceDispatcher: reduction accumulator '" + container + "' must be a device pointer or scalar"
    );
}

// Whether the runtime provides a native atomicAdd overload for this primitive.
bool has_native_atomic_add(types::PrimitiveType prim) {
    const size_t width = types::bit_width(prim);
    if (types::is_floating_point(prim)) {
        return width == 32 || width == 64;
    }
    if (width == 32) {
        return true;
    }
    if (width == 64 && types::is_unsigned(prim)) {
        return true;
    }
    return false;
}

// Single, indvar-invariant index with which `container` is accessed in the reduce body.
symbolic::Expression accumulator_index(
    structured_control_flow::Sequence& root, const std::string& container, const symbolic::Symbol& indvar
) {
    symbolic::Expression index = SymEngine::null;
    bool found = false;

    std::list<structured_control_flow::ControlFlowNode*> queue = {&root};
    while (!queue.empty()) {
        auto* current = queue.front();
        queue.pop_front();

        if (auto* block = dynamic_cast<structured_control_flow::Block*>(current)) {
            auto& dfg = block->dataflow();
            for (auto& memlet : dfg.edges()) {
                const auto* src = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
                const auto* dst = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst());
                const data_flow::AccessNode* access = nullptr;
                if (src != nullptr && src->data() == container) {
                    access = src;
                } else if (dst != nullptr && dst->data() == container) {
                    access = dst;
                }
                if (access == nullptr || memlet.subset().size() != 1) {
                    continue;
                }
                auto candidate = memlet.subset()[0];
                if (!found) {
                    index = candidate;
                    found = true;
                } else if (!symbolic::eq(index, candidate)) {
                    throw InvalidSDFGException(
                        "GPUOffloadReduceDispatcher: accumulator '" + container +
                        "' is accessed with inconsistent indices in the reduce body"
                    );
                }
            }
        } else if (auto* seq = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < seq->size(); ++i) {
                queue.push_back(&seq->at(i));
            }
        } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&loop->root());
        }
    }

    if (!found) {
        throw InvalidSDFGException(
            "GPUOffloadReduceDispatcher: accumulator '" + container + "' is not accessed in the reduce body"
        );
    }
    if (symbolic::uses(index, indvar)) {
        throw InvalidSDFGException(
            "GPUOffloadReduceDispatcher: accumulator '" + container + "' index depends on the reduction variable '" +
            indvar->get_name() + "'; this is a scatter, not a reduction into a single slot"
        );
    }
    return index;
}

} // namespace

GPUOffloadReduceDispatcher::GPUOffloadReduceDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : codegen::NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {

      };

bool GPUOffloadReduceDispatcher::is_outermost_map(analysis::AnalysisManager& analysis_manager) {
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

void GPUOffloadReduceDispatcher::dispatch_node(
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

        this->dispatch_kernel_body(library_snippet_factory, library_stream, node_.indvar(), scope_variables, num_iters);

        library_stream.setIndent(library_stream.indent() - 4);
        library_stream << "}" << std::endl;
    } else {
        this->dispatch_kernel_body(library_snippet_factory, main_stream, node_.indvar(), scope_variables, num_iters);
    }
};

void GPUOffloadReduceDispatcher::dispatch_header(
    codegen::PrettyPrinter& globals_stream,
    const std::string& kernel_name,
    std::vector<std::string>& arguments_declaration
) {
    globals_stream << "__global__ void " << kernel_name << "(";
    globals_stream << helpers::join(arguments_declaration, ", ");
    globals_stream << ")";
}

void GPUOffloadReduceDispatcher::dispatch_kernel_body(
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

    // Declare this level's reduction partials (registers for WARP/GRID, shared memory for
    // BLOCK) and initialize them to each operator's identity element.
    this->dispatch_reduction_declarations(kernel_language_extension, library_stream, library_snippet_factory, target_level);


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
    // iteration space and accumulates into its per-thread register; the
    // cross-lane reduction is performed afterwards via __shfl_xor_sync over the
    // enclosing X_BLOCK lanes. The coverage loop therefore runs once per
    // iteration rather than once per warp_size.
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
        std::string target_level_idx_access = kernel_language_extension.expression(node_.stride()) + " * " +
                                              kernel_language_extension.expression(get_target_level_idx(target_level));

        if (target_level == TargetLevel::X_BLOCK && nested_warp_dim(node_, analysis_manager_)) {
            target_level_idx_access = kernel_language_extension.expression(get_target_level_idx(target_level));
        }

        // compute the effective indvar for this coverage loop iteration
        library_stream << "size_t " << indvar->get_name() << " = " << kernel_language_extension.expression(node_.init())
                       << " + " << coverage_loop_var << " * "
                       << kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size()))
                       << " + " << target_level_idx_access << ";" << std::endl;
    }


    // Boundary Conditions
    if (!gpu::ScheduleType_GPU_Offload::nested_sync(node_.schedule_type())) {
        library_stream << "if (" << kernel_language_extension.expression(node_.condition()) << ") {" << std::endl;
        library_stream.setIndent(library_stream.indent() + 4);
    }


    // Redirect accumulator accesses in the body onto this level's private/shared partials.
    this->dispatch_reduction_shadow(kernel_language_extension, library_stream, target_level);

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

    // Combine the per-thread / per-warp partials for this level into the accumulator.
    this->dispatch_reduction_combine(kernel_language_extension, library_stream, library_snippet_factory, target_level);
}

void GPUOffloadReduceDispatcher::dispatch_kernel_preamble(
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

bool GPUOffloadReduceDispatcher::has_nested_warp_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.descendants(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()) != TargetLevel::WARP) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::has_enclosing_block_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.ancestors(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_block_level(gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()))) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::has_nested_block_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.descendants(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_block_level(gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()))) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::has_enclosing_grid_reduction(const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    for (auto* loop : loop_analysis.ancestors(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_grid_level(gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type()))) {
            continue;
        }
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                return true;
            }
        }
    }
    return false;
}

bool GPUOffloadReduceDispatcher::block_result_collides_across_grid(const symbolic::Expression& index) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();

    // Free symbols of the accumulator index.
    std::unordered_set<std::string> index_symbols;
    for (auto& atom : symbolic::atoms(index)) {
        index_symbols.insert(atom->get_name());
    }

    for (auto* loop : loop_analysis.ancestors(&node_)) {
        auto* struc_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(loop);
        if (struc_loop == nullptr) {
            continue;
        }
        if (struc_loop->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        if (!is_grid_level(gpu::ScheduleType_GPU_Offload::target_level(struc_loop->schedule_type()))) {
            continue;
        }
        // If this grid loop's induction variable does not index the accumulator, every
        // grid block (and coverage-loop iteration) writes the same global slot.
        if (index_symbols.find(struc_loop->indvar()->get_name()) == index_symbols.end()) {
            return true;
        }
    }
    return false;
}

std::string GPUOffloadReduceDispatcher::
    block_reduce_leader_condition(codegen::LanguageExtension& language_extension, const std::string& container) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    std::vector<std::string> conditions;

    // This level's own axis leader.
    TargetLevel my_level = gpu::ScheduleType_GPU_Offload::target_level(node_.schedule_type());
    conditions.push_back("(" + language_extension.expression(get_target_level_idx(my_level)) + " == 0)");

    // Plus every nested block-level reduce of the same container: their axes have been
    // folded into flat-index 0, so only that slot holds the fully combined result.
    for (auto* loop : loop_analysis.descendants(&node_)) {
        auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
        if (reduce == nullptr) {
            continue;
        }
        if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            continue;
        }
        TargetLevel level = gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type());
        if (!is_block_level(level)) {
            continue;
        }
        bool reduces_container = false;
        for (const auto& r : reduce->reductions()) {
            if (r.container == container) {
                reduces_container = true;
                break;
            }
        }
        if (!reduces_container) {
            continue;
        }
        conditions.push_back("(" + language_extension.expression(get_target_level_idx(level)) + " == 0)");
    }

    return helpers::join(conditions, " && ");
}

std::string GPUOffloadReduceDispatcher::reduce_linear_thread_index(codegen::LanguageExtension& language_extension) {
    // threadIdx.x + threadIdx.y * BX + threadIdx.z * BX * BY, where BX/BY are the static
    // block dimensions from the schedule (== launched blockDim.x/y), so the flat layout is
    // a constant expression consistent with the statically sized shared buffer.
    symbolic::Expression bx = reduce_block_dim(TargetLevel::X_BLOCK);
    symbolic::Expression by = reduce_block_dim(TargetLevel::Y_BLOCK);
    symbolic::Expression lin = symbolic::
        add(symbolic::threadIdx_x(),
            symbolic::
                add(symbolic::mul(symbolic::threadIdx_y(), bx),
                    symbolic::mul(symbolic::threadIdx_z(), symbolic::mul(bx, by))));
    return language_extension.expression(lin);
}

std::string GPUOffloadReduceDispatcher::
    reduce_axis_stride(codegen::LanguageExtension& language_extension, TargetLevel target_level) {
    symbolic::Expression bx = reduce_block_dim(TargetLevel::X_BLOCK);
    symbolic::Expression by = reduce_block_dim(TargetLevel::Y_BLOCK);
    switch (target_level) {
        case TargetLevel::Y_BLOCK:
        case TargetLevel::Y_GRID:
            return language_extension.expression(bx);
        case TargetLevel::Z_BLOCK:
        case TargetLevel::Z_GRID:
            return language_extension.expression(symbolic::mul(bx, by));
        default:
            return language_extension.expression(symbolic::one());
    }
}

symbolic::Expression GPUOffloadReduceDispatcher::reduce_block_dim(TargetLevel block_level) {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();

    symbolic::Expression dim = symbolic::one();
    auto collect = [&](structured_control_flow::StructuredLoop* loop) {
        if (loop->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            return;
        }
        if (gpu::ScheduleType_GPU_Offload::target_level(loop->schedule_type()) == block_level) {
            dim = gpu::ScheduleType_GPU_Offload::parallel_size(loop->schedule_type());
        }
    };

    collect(&node_);
    for (auto* loop : loop_analysis.ancestors(&node_)) {
        if (auto* struc_loop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            collect(struc_loop);
        }
    }
    for (auto* loop : loop_analysis.descendants(&node_)) {
        if (auto* struc_loop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            collect(struc_loop);
        }
    }
    return dim;
}

symbolic::Expression GPUOffloadReduceDispatcher::reduce_block_size_product() {
    return symbolic::
        mul(reduce_block_dim(TargetLevel::X_BLOCK),
            symbolic::mul(reduce_block_dim(TargetLevel::Y_BLOCK), reduce_block_dim(TargetLevel::Z_BLOCK)));
}

void GPUOffloadReduceDispatcher::dispatch_reduction_declarations(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    TargetLevel target_level
) {
    const bool grid = is_grid_level(target_level);
    const bool block = is_block_level(target_level);
    const bool warp = target_level == TargetLevel::WARP;

    // Every thread of a (possibly multi-dimensional) block owns a distinct shared slot,
    // addressed by its flat thread index; the buffer spans the whole block (x * y * z).
    // Sizing to the full flat thread count (rather than only this reduction's own block
    // axes) keeps every thread's slot distinct even when the reduction is nested inside a
    // map over another block dimension, which would otherwise alias slots and race.
    // A warp nested inside the block publishes its result into its lane-0 slot and leaves
    // the rest at identity, so the same per-thread layout serves warp and non-warp blocks.
    std::string lin_tid = reduce_linear_thread_index(language_extension);
    // Compile-time constant: __shared__ arrays may not be variable-length. The launched
    // blockDim equals the product of the block levels' parallel_size, so this constant
    // matches blockDim.x*blockDim.y*blockDim.z while remaining a constant expression.
    std::string block_size = language_extension.expression(reduce_block_size_product());

    bool needs_cstdint = false;
    bool needs_cmath = false;
    bool declared_shared = false;

    for (const auto& r : node_.reductions()) {
        auto prim = accumulator_primitive(sdfg_, r.container);
        std::string ctype = language_extension.primitive_type(prim);
        std::string identity = identity_literal(r.operation, prim);
        if (types::is_floating_point(prim)) {
            needs_cmath = true;
        } else {
            needs_cstdint = true;
        }

        std::string reg_name = "__daisy_reduce_reg_" + r.container;
        std::string smem_name = "__daisy_reduce_smem_" + r.container;

        if (grid || warp) {
            stream << ctype << " " << reg_name << " = " << identity << ";" << std::endl;
        } else if (block && !has_enclosing_block_reduction(r.container)) {
            // Only the outermost block level owning this container declares the single shared
            // buffer; every inner block level folds into that same buffer. Declaring one per
            // level would create same-named shadows of which only the innermost is populated,
            // while the outer identity-filled buffers clobber the result on write-back.
            declared_shared = true;
            stream << "__shared__ " << ctype << " " << smem_name << "[" << block_size << "];" << std::endl;
            stream << smem_name << "[" << lin_tid << "] = " << identity << ";" << std::endl;
        }
    }

    // Publishers and readers of the shared partials live in different threads.
    if (declared_shared) {
        stream << "__syncthreads();" << std::endl;
    }
    if (needs_cstdint) {
        library_snippet_factory.add_global("#include <cstdint>");
    }
    if (needs_cmath) {
        library_snippet_factory.add_global("#include <cmath>");
    }
}

void GPUOffloadReduceDispatcher::dispatch_reduction_shadow(
    codegen::LanguageExtension& language_extension, codegen::PrettyPrinter& stream, TargetLevel target_level
) {
    const bool block = is_block_level(target_level);
    std::string lin_tid = reduce_linear_thread_index(language_extension);

    for (const auto& r : node_.reductions()) {
        // For warp-nested block reductions the accumulation is emitted by the nested
        // warp level, so the block level only owns the shared buffer, not the body.
        if (block && has_nested_warp_reduction(r.container)) {
            continue;
        }

        auto prim = accumulator_primitive(sdfg_, r.container);
        std::string ctype = language_extension.primitive_type(prim);
        std::string reg_name = "__daisy_reduce_reg_" + r.container;
        std::string smem_name = "__daisy_reduce_smem_" + r.container;

        std::string storage = block ? ("&" + smem_name + "[" + lin_tid + "]") : ("&" + reg_name);

        auto index = accumulator_index(node_.root(), r.container, node_.indvar());
        if (symbolic::eq(index, symbolic::zero())) {
            stream << ctype << " *" << r.container << " = " << storage << ";" << std::endl;
        } else {
            stream << ctype << " *" << r.container << " = " << storage << " - (" << language_extension.expression(index)
                   << ");" << std::endl;
        }
    }
}

void GPUOffloadReduceDispatcher::dispatch_reduction_combine(
    codegen::LanguageExtension& language_extension,
    codegen::PrettyPrinter& stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    TargetLevel target_level
) {
    const bool grid = is_grid_level(target_level);
    const bool block = is_block_level(target_level);
    const bool warp = target_level == TargetLevel::WARP;

    // A reduce reduces only along its own axis; every other block dimension is an
    // independent "row". Shared slots are addressed by the flat thread index, so the
    // halving tree walks the reduce axis with the stride that separates its neighbours
    // in the flat layout, while the axis-local index bounds the loop and selects writers.
    std::string lin_tid = reduce_linear_thread_index(language_extension);
    std::string warp_size = language_extension.expression(get_target_level_dim(TargetLevel::WARP, get_warp_size()));

    for (const auto& r : node_.reductions()) {
        auto prim = accumulator_primitive(sdfg_, r.container);
        std::string ctype = language_extension.primitive_type(prim);
        std::string reg_name = "__daisy_reduce_reg_" + r.container;
        std::string smem_name = "__daisy_reduce_smem_" + r.container;
        auto index = accumulator_index(node_.root(), r.container, node_.indvar());
        std::string target = "reinterpret_cast<" + ctype + " *>(" + r.container + ")[" +
                             language_extension.expression(index) + "]";

        if (warp) {
            // Reduce the per-lane partials into every lane's register. The shuffle is read
            // once into a temporary before combining: emitting the __shfl_xor_sync inside a
            // combine expression (e.g. a min/max ternary) would duplicate it into divergent
            // branches, and on Volta+ a masked shuffle stalls until every named lane reaches
            // that exact instruction, deadlocking the warp.
            std::string other = "__daisy_reduce_shfl_" + r.container;
            stream << "for (int __daisy_reduce_mask = " << warp_size << " / 2; __daisy_reduce_mask > 0; "
                   << "__daisy_reduce_mask >>= 1) {" << std::endl;
            stream.setIndent(stream.indent() + 4);
            stream << ctype << " " << other << " = " << warp_shuffle_xor(reg_name, "__daisy_reduce_mask") << ";"
                   << std::endl;
            stream << reg_name << " = " << combine_expr(r.operation, reg_name, other) << ";" << std::endl;
            stream.setIndent(stream.indent() - 4);
            stream << "}" << std::endl;

            if (has_enclosing_block_reduction(r.container)) {
                // Publish this warp's result into the enclosing block's per-thread shared
                // buffer at the lane-0 flat slot; every other slot keeps the operator
                // identity set at declaration time. The block-level per-thread reduction
                // tree then folds these partials (across all remaining block dimensions)
                // exactly as it folds ordinary per-thread partials.
                //
                // Combine (rather than overwrite) into the slot: when the enclosing block
                // owner's count exceeds its parallel_size its coverage loop runs multiple
                // tiles, re-executing this publish once per tile, and each tile's warp
                // result must fold into the identity-initialised slot instead of clobbering
                // the previous tiles. For a single tile combine(identity, reg) == reg.
                std::string slot = smem_name + "[" + lin_tid + "]";
                stream << "if (lane == 0) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << slot << " = " << combine_expr(r.operation, slot, reg_name) << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            } else {
                // No block level owns this container, so the per-warp result must reach
                // the global accumulator directly. The warp leader atomically merges its
                // register (cross-warp and cross-block combine) into acc[index].
                stream << "if (lane == 0) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                if (r.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
                    stream << "atomicAdd(&" << target << ", " << reg_name << ");" << std::endl;
                } else {
                    std::string type_tag = ctype;
                    std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
                    std::string helper = "__daisy_reduce_combine_" + op_tag(r.operation) + "_" + type_tag;
                    stream << helper << "(&" << target << ", " << reg_name << ");" << std::endl;
                }
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            }
        } else if (block) {
            // Inner block levels only accumulate their per-thread partials into the single
            // shared buffer via the body; they emit no fold here. A block coverage loop of
            // an enclosing level iterates multiple times when count > parallel_size, so
            // folding a nested axis at the inner level would collapse the buffer between
            // coverage passes and corrupt slots that later passes still accumulate into.
            // Instead, the outermost block level folds every reduced block axis exactly
            // once, after all coverage-loop iterations have finished accumulating.
            if (has_enclosing_block_reduction(r.container)) {
                continue;
            }

            // Emit one halving tree over a block axis. Neighbours are `half * stride` flat
            // slots apart (stride 1/bx/bx*by for x/y/z); ceil-half + bound guard handles
            // non-power-of-two sizes. A nested warp publishes its per-warp result into the
            // lane-0 slot and leaves every other slot at the operator identity, so the same
            // per-thread tree passes those partials through unchanged.
            auto emit_block_tree = [&](TargetLevel lvl, ReductionOperation op) {
                std::string a_dim = language_extension.expression(reduce_block_dim(lvl));
                std::string a_idx = language_extension.expression(get_target_level_idx(lvl));
                std::string a_stride = reduce_axis_stride(language_extension, lvl);
                std::string tag = gpu::to_string(lvl);
                std::string mvar = "__daisy_reduce_m_" + r.container + "_" + tag;
                std::string hvar = "__daisy_reduce_half_" + r.container + "_" + tag;
                std::string a = smem_name + "[" + lin_tid + "]";
                std::string b = smem_name + "[" + lin_tid + " + " + hvar + " * " + a_stride + "]";
                stream << "__syncthreads();" << std::endl;
                stream << "for (int " << mvar << " = " << a_dim << "; " << mvar << " > 1; ) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << "int " << hvar << " = (" << mvar << " + 1) / 2;" << std::endl;
                stream << "if (" << a_idx << " < " << mvar << " - " << hvar << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << a << " = " << combine_expr(op, a, b) << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
                stream << "__syncthreads();" << std::endl;
                stream << mvar << " = " << hvar << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            };

            // Fold every nested block axis of this container first, then this (outermost)
            // level's axis, so all block dimensions collapse into flat slot 0.
            auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
            for (auto* loop : loop_analysis.descendants(&node_)) {
                auto* reduce = dynamic_cast<structured_control_flow::Reduce*>(loop);
                if (reduce == nullptr) {
                    continue;
                }
                if (reduce->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
                    continue;
                }
                TargetLevel nested = gpu::ScheduleType_GPU_Offload::target_level(reduce->schedule_type());
                if (!is_block_level(nested)) {
                    continue;
                }
                for (const auto& rr : reduce->reductions()) {
                    if (rr.container == r.container) {
                        emit_block_tree(nested, rr.operation);
                        break;
                    }
                }
            }
            emit_block_tree(target_level, r.operation);

            // The outermost block level commits to the global accumulator. The writer is the
            // leader across every reduced block axis (this level plus all nested block
            // reduces), so exactly one thread per remaining (mapped) slot commits
            // smem[lin_tid]. When the block result is folded into an enclosing grid register
            // / persists across grid coverage tiles it is combined into the target rather
            // than overwritten, otherwise the last tile would win.
            {
                std::string block_src = smem_name + "[" + lin_tid + "]";
                std::string leader = block_reduce_leader_condition(language_extension, r.container);
                bool enclosed_by_reduction = has_enclosing_grid_reduction(r.container);
                bool collides = !enclosed_by_reduction && block_result_collides_across_grid(index);
                stream << "if (" << leader << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                if (collides) {
                    if (r.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
                        stream << "atomicAdd(&" << target << ", " << block_src << ");" << std::endl;
                    } else {
                        std::string type_tag = ctype;
                        std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
                        std::string helper = "__daisy_reduce_combine_" + op_tag(r.operation) + "_" + type_tag;
                        stream << helper << "(&" << target << ", " << block_src << ");" << std::endl;
                    }
                } else {
                    std::string block_write = enclosed_by_reduction ? combine_expr(r.operation, target, block_src)
                                                                    : block_src;
                    stream << target << " = " << block_write << ";" << std::endl;
                }
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            }
        } else if (grid) {
            // Atomics are exclusive to grid-level reductions. When the grid register is
            // fed by a nested block/warp reduction of the same container, only the block's
            // axis leaders hold a non-identity value (every other thread holds the operator
            // identity), so committing every thread is correct. Otherwise the grid body is
            // replicated verbatim across all block threads and each holds an identical copy
            // of the partial; committing all of them would multiply the result by blockDim,
            // so only one thread per block may commit. (For a pure grid reduction blockDim
            // == 1, making the guard a no-op.)
            bool fed_by_nested_reduction = has_nested_block_reduction(r.container) ||
                                           has_nested_warp_reduction(r.container);
            if (!fed_by_nested_reduction) {
                stream << "if (" << lin_tid << " == 0) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
            }
            if (r.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
                stream << "atomicAdd(&" << target << ", " << reg_name << ");" << std::endl;
            } else {
                std::string type_tag = ctype;
                std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
                std::string helper = "__daisy_reduce_combine_" + op_tag(r.operation) + "_" + type_tag;
                stream << helper << "(&" << target << ", " << reg_name << ");" << std::endl;
            }
            if (!fed_by_nested_reduction) {
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            }
        }
    }
}


} // namespace gpu
} // namespace sdfg
