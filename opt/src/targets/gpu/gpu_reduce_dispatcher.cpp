#include "sdfg/targets/gpu/gpu_reduce_dispatcher.h"

#include <algorithm>
#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <vector>

#include <sdfg/analysis/analysis.h>
#include <sdfg/analysis/arguments_analysis.h>
#include <sdfg/analysis/loop_analysis.h>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <sdfg/data_flow/access_node.h>
#include <sdfg/data_flow/memlet.h>
#include <sdfg/exceptions.h>
#include <sdfg/helpers/helpers.h>
#include <sdfg/structured_control_flow/block.h>
#include <sdfg/structured_control_flow/map.h>
#include <sdfg/structured_control_flow/reduce.h>
#include <sdfg/structured_control_flow/structured_loop.h>
#include <sdfg/symbolic/symbolic.h>
#include <sdfg/types/scalar.h>
#include <sdfg/types/type.h>

#include "sdfg/targets/gpu/gpu_schedule_type.h"

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
    throw InvalidSDFGException("GPUReduceDispatcher: unknown reduction operation");
}

// Identity element of the operator for the given primitive type, as a C literal.
std::string identity_literal(ReductionOperation op, types::PrimitiveType prim) {
    if (op == ReductionOperation::Add) {
        return "0";
    }
    if (op == ReductionOperation::Mul) {
        return "1";
    }

    // Min / Max: need the most-extreme element so the first real value wins.
    if (types::is_floating_point(prim)) {
        // INFINITY / -INFINITY are provided by the CUDA/HIP math headers.
        return op == ReductionOperation::Min ? "INFINITY" : "-INFINITY";
    }

    // Integers: use the <cstdint> limit macros instead of hand-written literals.
    const size_t width = types::bit_width(prim);
    const bool is_unsigned = types::is_unsigned(prim);

    if (op == ReductionOperation::Min) {
        // identity for Min is the type maximum
        if (is_unsigned) {
            if (width == 32) return "UINT32_MAX";
            if (width == 64) return "UINT64_MAX";
        } else {
            if (width == 32) return "INT32_MAX";
            if (width == 64) return "INT64_MAX";
        }
    } else { // Max -> identity is the type minimum
        if (is_unsigned) {
            return "0";
        }
        if (width == 32) return "INT32_MIN";
        if (width == 64) return "INT64_MIN";
    }

    throw InvalidSDFGException("GPUReduceDispatcher: unsupported integer width for min/max reduction");
}

// Resolve the scalar element type of a reduction accumulator container.
// The accumulator must be a device-resident pointer to a scalar (the offloaded
// form of the host accumulator).
types::PrimitiveType accumulator_primitive(const StructuredSDFG& sdfg, const std::string& container) {
    auto& type = sdfg.type(container);
    if (auto* ptr = dynamic_cast<const types::Pointer*>(&type)) {
        if (!ptr->has_pointee_type()) {
            throw InvalidSDFGException(
                "GPUReduceDispatcher: reduction accumulator '" + container + "' has no pointee type"
            );
        }
        if (auto* scalar = dynamic_cast<const types::Scalar*>(&ptr->pointee_type())) {
            return scalar->primitive_type();
        }
        throw InvalidSDFGException(
            "GPUReduceDispatcher: reduction accumulator '" + container + "' must point to a scalar"
        );
    }
    throw InvalidSDFGException(
        "GPUReduceDispatcher: reduction accumulator '" + container +
        "' must be a device-resident pointer (offload it before scheduling)"
    );
}

// Find the single index expression with which `container` is accessed in the
// reduce body. The accumulator is a flat pointer (Pointer->Scalar), so its
// memlet subset has exactly one element; e.g. `acc[0]` for a scalar reduction
// or `acc[i]` for a reduction whose output slot is selected by an enclosing
// (data-parallel) map's induction variable.
//
// All accesses to the accumulator must use the same index, and that index must
// be invariant in the reduction induction variable `indvar` -- otherwise the
// body scatters across distinct slots per iteration and is not a reduction into
// a single accumulator element (that requires OutLocalStorage / privatization
// analysis, which is out of scope for this baseline).
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
                        "GPUReduceDispatcher: accumulator '" + container +
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
            "GPUReduceDispatcher: accumulator '" + container + "' is not accessed in the reduce body"
        );
    }
    if (symbolic::uses(index, indvar)) {
        throw InvalidSDFGException(
            "GPUReduceDispatcher: accumulator '" + container + "' index depends on the reduction variable '" +
            indvar->get_name() + "'; this is a scatter, not a reduction into a single slot"
        );
    }
    return index;
}

// Per-dimension GPU built-in symbols (identical intrinsics on CUDA and HIP).
struct DimSymbols {
    symbolic::Symbol thread_idx;
    symbolic::Symbol block_idx;
    symbolic::Symbol block_dim;
    symbolic::Symbol grid_dim;
};

DimSymbols dim_symbols(GPUDimension dim) {
    switch (dim) {
        case GPUDimension::X:
            return {symbolic::threadIdx_x(), symbolic::blockIdx_x(), symbolic::blockDim_x(), symbolic::gridDim_x()};
        case GPUDimension::Y:
            return {symbolic::threadIdx_y(), symbolic::blockIdx_y(), symbolic::blockDim_y(), symbolic::gridDim_y()};
        case GPUDimension::Z:
            return {symbolic::threadIdx_z(), symbolic::blockIdx_z(), symbolic::blockDim_z(), symbolic::gridDim_z()};
    }
    throw InvalidSDFGException("GPUReduceDispatcher: invalid GPU dimension");
}

// Whether the runtime provides a native atomicAdd overload for this primitive.
bool has_native_atomic_add(types::PrimitiveType prim) {
    const size_t width = types::bit_width(prim);
    if (types::is_floating_point(prim)) {
        return width == 32 || width == 64; // float, double (double needs sm_60+ on CUDA)
    }
    if (width == 32) {
        return true; // int / unsigned int
    }
    if (width == 64 && types::is_unsigned(prim)) {
        return true; // unsigned long long
    }
    return false; // signed 64-bit -> CAS fallback
}

} // namespace

GPUReduceDispatcher::GPUReduceDispatcher(
    codegen::LanguageExtension& language_extension,
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::Reduce& node,
    codegen::InstrumentationPlan& instrumentation_plan,
    codegen::ArgCapturePlan& arg_capture_plan
)
    : codegen::NodeDispatcher(language_extension, sdfg, analysis_manager, node, instrumentation_plan, arg_capture_plan),
      node_(node) {};

bool GPUReduceDispatcher::is_nested_in_gpu_kernel() {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    auto& loop_tree = loop_analysis.loop_tree();
    structured_control_flow::ControlFlowNode* ancestor = loop_tree.at(&node_);
    while (ancestor != nullptr) {
        if (auto* map = dynamic_cast<structured_control_flow::Map*>(ancestor)) {
            if (map->schedule_type().value() == this->schedule_value()) {
                return true;
            }
        }
        ancestor = loop_tree.at(ancestor);
    }
    return false;
}

void GPUReduceDispatcher::dispatch_node(
    codegen::PrettyPrinter& main_stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    if (node_.reductions().empty()) {
        throw InvalidSDFGException("GPUReduceDispatcher: Reduce node carries no reductions");
    }

    // Nested inside an enclosing GPU kernel: inline the privatize + grid-stride
    // reduce + atomic merge on this Reduce's own GPU dimension. No kernel launch.
    if (this->is_nested_in_gpu_kernel()) {
        GPUDimension reduce_dim = gpu_dimension(node_.schedule_type());
        std::vector<std::string> no_scope_variables;
        this->dispatch_reduce_core(
            library_snippet_factory, main_stream, reduce_dim, /*declare_scope_variables=*/false, no_scope_variables
        );
        return;
    }

    // Top-level: emit a standalone kernel parallelized on X.
    analysis::ArgumentsAnalysis& arguments_analysis = analysis_manager_.get<analysis::ArgumentsAnalysis>();
    auto& used_arguments = arguments_analysis.arguments(analysis_manager_, node_);
    auto& locals = arguments_analysis.locals(analysis_manager_, node_);

    auto indvar = node_.indvar();

    // Scope variables: loop-local temporaries except the induction variable.
    std::vector<std::string> scope_variables;
    for (auto& local : locals) {
        if (local == indvar->get_name()) {
            continue;
        }
        scope_variables.push_back(local);
    }
    std::sort(scope_variables.begin(), scope_variables.end());

    // Kernel arguments (device pointers and scalars), excluding NV symbols.
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

    // Grid geometry: flat X-dimension mapping, one thread per iteration.
    symbolic::Integer block_size = gpu_block_size(node_.schedule_type());
    symbolic::Expression num_iters = node_.num_iterations();
    symbolic::Expression block_size_expr = block_size;
    symbolic::Expression num_blocks = symbolic::max(symbolic::divide_ceil(num_iters, block_size_expr), symbolic::one());

    std::string kernel_name = "kernel_" + sdfg_.name() + "_" + std::to_string(node_.element_id());

    this->emit_kernel_call(main_stream, kernel_name, num_blocks, block_size_expr, arguments_device);

    this->emit_kernel_includes(library_snippet_factory);

    this->dispatch_header(globals_stream, kernel_name, arguments_declaration);
    globals_stream << ";" << std::endl;

    auto& library_stream = library_snippet_factory.require(kernel_name, this->kernel_file_extension(), true).stream();
    library_stream << "#include " << library_snippet_factory.header_path().filename() << std::endl << std::endl;
    this->emit_library_preamble(library_stream);

    this->dispatch_header(library_stream, kernel_name, arguments_declaration);
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    // Every device-pointer argument is a full cudaMalloc/hipMalloc allocation,
    // which is guaranteed >=256-byte aligned. Asserting 16-byte alignment lets
    // clang's load-store vectorizer widen contiguous copies to 128-bit; decltype
    // keeps it agnostic to element type / constness.
    for (auto& container : arguments) {
        if (this->is_device_pointer_storage(sdfg_.type(container).storage_type())) {
            library_stream << container << " = reinterpret_cast<decltype(" << container
                           << ")>(__builtin_assume_aligned(" << container << ", 16));" << std::endl;
        }
    }

    this->dispatch_reduce_core(
        library_snippet_factory, library_stream, GPUDimension::X, /*declare_scope_variables=*/true, scope_variables
    );

    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl;
}

void GPUReduceDispatcher::dispatch_header(
    codegen::PrettyPrinter& stream, const std::string& kernel_name, std::vector<std::string>& arguments_declaration
) {
    stream << "__global__ void " << kernel_name << "(";
    stream << helpers::join(arguments_declaration, ", ");
    stream << ")";
}

void GPUReduceDispatcher::dispatch_reduce_core(
    codegen::CodeSnippetFactory& library_snippet_factory,
    codegen::PrettyPrinter& library_stream,
    GPUDimension dimension,
    bool declare_scope_variables,
    std::vector<std::string>& scope_variables
) {
    std::unique_ptr<codegen::LanguageExtension> device_language_extension = this->create_device_language_extension();
    codegen::LanguageExtension& language_extension = *device_language_extension;
    DimSymbols dim = dim_symbols(dimension);

    // <cstdint> supplies the integer identity macros (INT32_MAX, ...).
    library_snippet_factory.add_global("#include <cstdint>");

    // Grid-stride parameters on the chosen dimension.
    symbolic::Expression flat_id = symbolic::add(dim.thread_idx, symbolic::mul(dim.block_idx, dim.block_dim));
    symbolic::Expression num_threads = symbolic::mul(dim.block_dim, dim.grid_dim);

    std::string start_var = "__daisy_reduce_tid";
    std::string step_var = "__daisy_reduce_nthreads";

    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    library_stream << "int " << start_var << " = " << language_extension.expression(flat_id) << ";" << std::endl;
    library_stream << "int " << step_var << " = " << language_extension.expression(num_threads) << ";" << std::endl;

    // Thread-private partials, one per reduction.
    for (auto& reduction : node_.reductions()) {
        auto prim = accumulator_primitive(sdfg_, reduction.container);
        std::string ctype = language_extension.primitive_type(prim);
        library_stream << ctype << " __daisy_reduce_" << reduction.container << " = "
                       << identity_literal(reduction.operation, prim) << ";" << std::endl;
    }

    // Body scope: shadow each accumulator pointer so the in-body combine writes
    // the thread-private partial instead of the shared device accumulator. For
    // an indexed accumulator `acc[index]` the shadow base is offset by `-index`
    // so the body's `acc[index]` resolves to the single private register; the
    // body re-adds `index`, so the net access is the register itself.
    library_stream << "{" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    for (auto& reduction : node_.reductions()) {
        auto prim = accumulator_primitive(sdfg_, reduction.container);
        std::string ctype = language_extension.primitive_type(prim);
        auto index = accumulator_index(node_.root(), reduction.container, node_.indvar());
        if (symbolic::eq(index, symbolic::zero())) {
            library_stream << ctype << " *" << reduction.container << " = &__daisy_reduce_" << reduction.container
                           << ";" << std::endl;
        } else {
            library_stream << ctype << " *" << reduction.container << " = &__daisy_reduce_" << reduction.container
                           << " - (" << language_extension.expression(index) << ");" << std::endl;
        }
    }

    // Declare loop-local scope variables (top-level only; nested kernels have
    // them declared by the enclosing map dispatcher).
    if (declare_scope_variables) {
        for (auto& local : scope_variables) {
            std::string val = language_extension.declaration(local, sdfg_.type(local), false, true);
            if (!val.empty()) {
                library_stream << val << ";" << std::endl;
            }
        }
    }

    // Grid-stride reduce loop on the reduction dimension:
    //   for (j = init + tid*stride; <cond>; j += nthreads*stride)
    auto indvar = node_.indvar();
    auto stride = node_.stride();
    symbolic::Expression strided_start = symbolic::symbol(start_var);
    symbolic::Expression strided_step = symbolic::symbol(step_var);
    if (!stride.is_null() && !symbolic::eq(stride, symbolic::one())) {
        strided_start = symbolic::mul(strided_start, stride);
        strided_step = symbolic::mul(strided_step, stride);
    }
    symbolic::Expression loop_init = strided_start;
    auto init = node_.init();
    if (!symbolic::eq(init, symbolic::zero())) {
        loop_init = symbolic::add(init, strided_start);
    }
    symbolic::Expression loop_update = symbolic::add(indvar, strided_step);

    library_stream << "for (int " << indvar->get_name() << " = " << language_extension.expression(loop_init) << "; "
                   << language_extension.expression(node_.condition()) << "; " << indvar->get_name() << " = "
                   << language_extension.expression(loop_update) << ") {" << std::endl;
    library_stream.setIndent(library_stream.indent() + 4);

    codegen::SequenceDispatcher
        dispatcher(language_extension, sdfg_, analysis_manager_, node_.root(), instrumentation_plan_, arg_capture_plan_);
    dispatcher.dispatch(library_stream, library_stream, library_snippet_factory);

    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl; // for

    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl; // body scope

    // Atomic merge of each private partial into the shared device accumulator.
    for (auto& reduction : node_.reductions()) {
        this->dispatch_atomic_merge(library_stream, library_snippet_factory, reduction);
    }

    library_stream.setIndent(library_stream.indent() - 4);
    library_stream << "}" << std::endl; // outer scope
}

void GPUReduceDispatcher::dispatch_atomic_merge(
    codegen::PrettyPrinter& library_stream,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const structured_control_flow::ReductionInfo& reduction
) {
    std::unique_ptr<codegen::LanguageExtension> device_language_extension = this->create_device_language_extension();
    codegen::LanguageExtension& language_extension = *device_language_extension;
    auto prim = accumulator_primitive(sdfg_, reduction.container);
    std::string ctype = language_extension.primitive_type(prim);

    // Merge into the accumulator's real slot. Outside the shadow scope the
    // container name again refers to the device pointer parameter, so the
    // address is `&acc[index]` -- distinct per enclosing-map thread, shared
    // across the parallel reduction threads (hence the atomic).
    auto index = accumulator_index(node_.root(), reduction.container, node_.indvar());
    std::string index_str = language_extension.expression(index);
    std::string target = "&(reinterpret_cast<" + ctype + " *>(" + reduction.container + "))[" + index_str + "]";
    std::string value = "__daisy_reduce_" + reduction.container;

    // Fast path: native atomicAdd where the runtime provides an overload.
    if (reduction.operation == ReductionOperation::Add && has_native_atomic_add(prim)) {
        library_stream << "atomicAdd(" << target << ", " << value << ");" << std::endl;
        return;
    }

    std::string type_tag = ctype;
    std::replace(type_tag.begin(), type_tag.end(), ' ', '_');
    std::string helper_name = "__daisy_reduce_combine_" + op_tag(reduction.operation) + "_" + type_tag;

    // Every other operator/type uses a device-side compare-and-swap combine
    // helper. These live in daisy_rtl.h (guarded by __CUDACC__/__HIPCC__) so
    // they are visible to every kernel translation unit; we only emit the call.
    // The CAS baseline supports 32/64-bit accumulators only.
    const size_t width = types::bit_width(prim);
    if (width != 32 && width != 64) {
        throw InvalidSDFGException(
            "GPUReduceDispatcher: only 32/64-bit reduction accumulators are supported in the atomics baseline"
        );
    }

    // The accumulator container is the real device pointer again outside the
    // shadow scope; merge into its (loop-invariant) element 0.
    library_stream << helper_name << "(" << target << ", " << value << ");" << std::endl;
}

codegen::InstrumentationInfo GPUReduceDispatcher::instrumentation_info() const {
    auto& loop_analysis = analysis_manager_.get<analysis::LoopAnalysis>();
    analysis::LoopInfo loop_info = loop_analysis.loop_info(&node_);

    std::unordered_map<std::string, std::string> metrics;
    auto& flop_analysis = analysis_manager_.get<analysis::FlopAnalysis>();
    auto flop = flop_analysis.get_if_available_for_codegen(&node_);
    if (!flop.is_null()) {
        metrics.insert({"flop", language_extension_.expression(flop)});
    }

    return codegen::InstrumentationInfo(
        node_.element_id(),
        node_.element_type(),
        this->target_type(),
        codegen::InstrumentationEventType::CUDA,
        loop_info,
        metrics
    );
}

} // namespace gpu
} // namespace sdfg
