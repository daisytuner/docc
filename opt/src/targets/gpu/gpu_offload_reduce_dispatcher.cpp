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
#include "sdfg/targets/gpu/gpu_schedule_type.h"
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

bool is_grid_level(TargetLevel level) {
    return level == TargetLevel::X_GRID || level == TargetLevel::Y_GRID || level == TargetLevel::Z_GRID;
}

bool is_block_level(TargetLevel level) {
    return level == TargetLevel::X_BLOCK || level == TargetLevel::Y_BLOCK || level == TargetLevel::Z_BLOCK;
}

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

symbolic::SymbolSet target_level_indvars(
    structured_control_flow::StructuredLoop& node, analysis::AnalysisManager& analysis_manager, TargetLevel target_level
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
    structured_control_flow::StructuredLoop& node,
    analysis::AnalysisManager& analysis_manager,
    std::unordered_map<TargetLevel, ScheduleType>& output
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);
    for (const auto& loop : loops) {
        if (auto struc_loop = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            if (struc_loop->schedule_type().category() == structured_control_flow::ScheduleTypeCategory::Offloader) {
                output.insert_or_assign(
                    ScheduleType_GPU::target_level(struc_loop->schedule_type()), struc_loop->schedule_type()
                );
            }
        }
    }
}

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
            block_size_x = ScheduleType_GPU::parallel_size(nested_schedule_types.at(TargetLevel::X_BLOCK));
        }
        if (nested_schedule_types.find(TargetLevel::Y_BLOCK) != nested_schedule_types.end()) {
            block_size_y = ScheduleType_GPU::parallel_size(nested_schedule_types.at(TargetLevel::Y_BLOCK));
        }
        if (nested_schedule_types.find(TargetLevel::Z_BLOCK) != nested_schedule_types.end()) {
            block_size_z = ScheduleType_GPU::parallel_size(nested_schedule_types.at(TargetLevel::Z_BLOCK));
        }
        if (nested_schedule_types.find(TargetLevel::X_GRID) != nested_schedule_types.end()) {
            grid_size_x = ScheduleType_GPU::parallel_size(nested_schedule_types.at(TargetLevel::X_GRID));
        }
        if (nested_schedule_types.find(TargetLevel::Y_GRID) != nested_schedule_types.end()) {
            grid_size_y = ScheduleType_GPU::parallel_size(nested_schedule_types.at(TargetLevel::Y_GRID));
        }
        if (nested_schedule_types.find(TargetLevel::Z_GRID) != nested_schedule_types.end()) {
            grid_size_z = ScheduleType_GPU::parallel_size(nested_schedule_types.at(TargetLevel::Z_GRID));
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
    TargetLevel target_level = ScheduleType_GPU::target_level(node_.schedule_type());
    std::string coverage_loop_var = "__daisy_gpu_coverage_loop_" + gpu::to_string(target_level);
    std::string size = kernel_language_extension.expression(node_.num_iterations());

    // Declare this level's reduction partials (registers for WARP/GRID, shared memory for
    // BLOCK) and initialize them to each operator's identity element.
    this->dispatch_reduction_declarations(kernel_language_extension, library_stream, library_snippet_factory, target_level);


    if (target_level == TargetLevel::WARP) {
        library_stream << "uint32_t num_warps = ceildiv("
                       << kernel_language_extension.expression(symbolic::blockDim_x()) << ", "
                       << kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size()))
                       << ");" << std::endl;
        library_stream << "uint32_t warp_id = " << kernel_language_extension.expression(symbolic::threadIdx_x())
                       << " / "
                       << kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size()))
                       << ";" << std::endl;
        library_stream << "uint32_t lane = " << kernel_language_extension.expression(symbolic::threadIdx_x()) << " & ("
                       << kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size()))
                       << " - 1);" << std::endl;
    }

    library_stream << "for (int " << coverage_loop_var << " = 0; " << coverage_loop_var << " < "
                   << "max(1, " << size << "/"
                   << kernel_language_extension.expression(get_target_level_dim(target_level, get_warp_size())) << "); "
                   << coverage_loop_var << "++) {" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    if (target_level == TargetLevel::WARP) {
        std::string indvar_name = indvar->get_name();
        std::string x_block_coverage_loop_var = "__daisy_gpu_coverage_loop_" + gpu::to_string(TargetLevel::X_BLOCK);
        auto x_block_parent = find_x_block_owning_warp_level(node_, analysis_manager_);
        if (!x_block_parent) {
            throw InvalidSDFGException("WARP level map must be nested within an X_BLOCK level map");
        }
        std::string x_block_indvar_name = x_block_parent->indvar()->get_name();
        std::string x_block_num_iterations = kernel_language_extension.expression(x_block_parent->num_iterations());
        std::string x_block_init = kernel_language_extension.expression(x_block_parent->init());

        std::string num_iterations = kernel_language_extension.expression(node_.num_iterations());

        library_stream << "size_t " << indvar_name << " = " << x_block_init << " + num_warps * " << num_iterations
                       << " * warp_id * " << num_iterations << " + " << coverage_loop_var << " * " << size << " + lane;"
                       << std::endl;
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
    if (!ScheduleType_GPU::nested_sync(node_.schedule_type())) {
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

    if (!ScheduleType_GPU::nested_sync(node_.schedule_type())) {
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
        if (ScheduleType_GPU::target_level(reduce->schedule_type()) != TargetLevel::WARP) {
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

std::string GPUOffloadReduceDispatcher::reduce_linear_thread_index(codegen::LanguageExtension& language_extension) {
    // threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y
    symbolic::Expression lin = symbolic::add(
        symbolic::threadIdx_x(),
        symbolic::
            add(symbolic::mul(symbolic::threadIdx_y(), symbolic::blockDim_x()),
                symbolic::mul(symbolic::threadIdx_z(), symbolic::mul(symbolic::blockDim_x(), symbolic::blockDim_y())))
    );
    return language_extension.expression(lin);
}

std::string GPUOffloadReduceDispatcher::
    reduce_axis_stride(codegen::LanguageExtension& language_extension, TargetLevel target_level) {
    switch (target_level) {
        case TargetLevel::Y_BLOCK:
        case TargetLevel::Y_GRID:
            return language_extension.expression(symbolic::blockDim_x());
        case TargetLevel::Z_BLOCK:
        case TargetLevel::Z_GRID:
            return language_extension.expression(symbolic::mul(symbolic::blockDim_x(), symbolic::blockDim_y()));
        default:
            return language_extension.expression(symbolic::one());
    }
}

symbolic::Expression GPUOffloadReduceDispatcher::reduce_block_size_product() {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();

    std::unordered_map<TargetLevel, symbolic::Expression> dims;
    auto collect = [&](structured_control_flow::StructuredLoop* loop) {
        if (loop->schedule_type().category() != structured_control_flow::ScheduleTypeCategory::Offloader) {
            return;
        }
        auto level = ScheduleType_GPU::target_level(loop->schedule_type());
        if (is_block_level(level)) {
            dims.insert_or_assign(level, ScheduleType_GPU::parallel_size(loop->schedule_type()));
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

    symbolic::Expression product = symbolic::one();
    for (auto& entry : dims) {
        product = symbolic::mul(product, entry.second);
    }
    return product;
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
    std::string lin_tid = reduce_linear_thread_index(language_extension);
    std::string block_size = language_extension.expression(reduce_block_size_product());
    std::string warp_size = language_extension.expression(get_target_level_dim(TargetLevel::WARP, get_warp_size()));
    std::string num_warps = "(" + block_size + " / " + warp_size + ")";

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
        } else if (block) {
            declared_shared = true;
            if (has_nested_warp_reduction(r.container)) {
                stream << "__shared__ " << ctype << " " << smem_name << "[" << num_warps << "];" << std::endl;
                stream << "if (" << lin_tid << " < " << num_warps << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << smem_name << "[" << lin_tid << "] = " << identity << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            } else {
                stream << "__shared__ " << ctype << " " << smem_name << "[" << block_size << "];" << std::endl;
                stream << smem_name << "[" << lin_tid << "] = " << identity << ";" << std::endl;
            }
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
    std::string axis_dim = language_extension.expression(ScheduleType_GPU::parallel_size(node_.schedule_type()));
    std::string block_size = language_extension.expression(reduce_block_size_product());
    std::string warp_size = language_extension.expression(get_target_level_dim(TargetLevel::WARP, get_warp_size()));
    std::string warps_per_axis = "(" + axis_dim + " / " + warp_size + ")";
    std::string global_warp = "(" + lin_tid + " / " + warp_size + ")";
    std::string stride = reduce_axis_stride(language_extension, target_level);

    for (const auto& r : node_.reductions()) {
        auto prim = accumulator_primitive(sdfg_, r.container);
        std::string ctype = language_extension.primitive_type(prim);
        std::string reg_name = "__daisy_reduce_reg_" + r.container;
        std::string smem_name = "__daisy_reduce_smem_" + r.container;
        auto index = accumulator_index(node_.root(), r.container, node_.indvar());
        std::string target = "reinterpret_cast<" + ctype + " *>(" + r.container + ")[" +
                             language_extension.expression(index) + "]";

        if (warp) {
            // Reduce the per-lane partials, then publish into the enclosing block's shared
            // buffer at this warp's flat slot (distinct per row of a multi-dimensional block).
            std::string shfl = "__shfl_xor_sync(0xffffffff, " + reg_name + ", __daisy_reduce_mask)";
            stream << "for (int __daisy_reduce_mask = " << warp_size << " / 2; __daisy_reduce_mask > 0; "
                   << "__daisy_reduce_mask >>= 1) {" << std::endl;
            stream.setIndent(stream.indent() + 4);
            stream << reg_name << " = " << combine_expr(r.operation, reg_name, shfl) << ";" << std::endl;
            stream.setIndent(stream.indent() - 4);
            stream << "}" << std::endl;
            stream << smem_name << "[" << global_warp << "] = " << reg_name << ";" << std::endl;
        } else if (block) {
            std::string axis_idx = language_extension.expression(get_target_level_idx(target_level));
            std::string mvar = "__daisy_reduce_m_" + r.container;
            std::string hvar = "__daisy_reduce_half_" + r.container;

            if (has_nested_warp_reduction(r.container)) {
                // Combine the per-warp partials of each axis-row. Warps along the reduce
                // axis are contiguous in the flat warp index, so the tree strides by one in
                // warp space; the lane-0 thread of each axis-warp drives it.
                std::string warp_in_axis = "(" + axis_idx + " / " + warp_size + ")";
                std::string a = smem_name + "[" + global_warp + "]";
                std::string b = smem_name + "[" + global_warp + " + " + hvar + "]";

                stream << "__syncthreads();" << std::endl;
                stream << "for (int " << mvar << " = " << warps_per_axis << "; " << mvar << " > 1; ) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << "int " << hvar << " = (" << mvar << " + 1) / 2;" << std::endl;
                stream << "if (" << axis_idx << " % " << warp_size << " == 0 && " << warp_in_axis << " < " << mvar
                       << " - " << hvar << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << a << " = " << combine_expr(r.operation, a, b) << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
                stream << "__syncthreads();" << std::endl;
                stream << mvar << " = " << hvar << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;

                // One writer per axis-row: the axis leader (its flat slot holds the result).
                stream << "if (" << axis_idx << " == 0) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << target << " = " << smem_name << "[" << global_warp << "];" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            } else {
                // Combine the per-thread partials along the reduce axis. Neighbours are
                // `hvar * stride` slots apart in the flat layout (stride 1/bx/bx*by for x/y/z).
                std::string a = smem_name + "[" + lin_tid + "]";
                std::string b = smem_name + "[" + lin_tid + " + " + hvar + " * " + stride + "]";

                // Halving tree over `axis_dim` slots; ceil-half + bound guard handles non-power-of-two sizes.
                stream << "__syncthreads();" << std::endl;
                stream << "for (int " << mvar << " = " << axis_dim << "; " << mvar << " > 1; ) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << "int " << hvar << " = (" << mvar << " + 1) / 2;" << std::endl;
                stream << "if (" << axis_idx << " < " << mvar << " - " << hvar << ") {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << a << " = " << combine_expr(r.operation, a, b) << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
                stream << "__syncthreads();" << std::endl;
                stream << mvar << " = " << hvar << ";" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;

                // One writer per axis-row: the axis leader (its flat slot holds the result).
                stream << "if (" << axis_idx << " == 0) {" << std::endl;
                stream.setIndent(stream.indent() + 4);
                stream << target << " = " << smem_name << "[" << lin_tid << "];" << std::endl;
                stream.setIndent(stream.indent() - 4);
                stream << "}" << std::endl;
            }
        } else if (grid) {
            // Atomics are exclusive to grid-level reductions.
            if (r.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
                stream << "atomicAdd(&" << target << ", " << reg_name << ");" << std::endl;
            } else {
                std::string type_tag = ctype;
                std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
                std::string helper = "__daisy_reduce_combine_" + op_tag(r.operation) + "_" + type_tag;
                stream << helper << "(&" << target << ", " << reg_name << ");" << std::endl;
            }
        }
    }
}


} // namespace gpu
} // namespace sdfg
