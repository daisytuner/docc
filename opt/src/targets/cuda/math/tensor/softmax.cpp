#include "sdfg/targets/cuda/math/tensor/softmax.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"

#include <algorithm>

namespace sdfg::cuda::tensor {

static constexpr int SOFTMAX_BLOCK_SIZE = 256;

// Contiguous softmax kernel: one block per row, warp-shuffle + shared-memory reductions.
// Used when the reduced axis is innermost (inner == 1), so a row's elements are contiguous
// in memory and threads within a warp access consecutive addresses (coalesced).
static void
emit_softmax_kernel_contiguous(codegen::PrettyPrinter& ks, const std::string& kernel_name, const std::string& type) {
    ks << "__global__ void " << kernel_name << "(const " << type << "* __restrict__ input, " << type
       << "* __restrict__ output, int num_rows, int row_size) {" << std::endl;
    ks.setIndent(ks.indent() + 4);

    ks << "int row = blockIdx.x;" << std::endl;
    ks << "if (row >= num_rows) return;" << std::endl;
    ks << std::endl;
    ks << "const " << type << "* row_in = input + (size_t)row * row_size;" << std::endl;
    ks << type << "* row_out = output + (size_t)row * row_size;" << std::endl;
    ks << std::endl;

    // Shared memory for cross-warp reduction
    ks << "extern __shared__ " << type << " sdata[];" << std::endl;
    ks << "int lane_id = threadIdx.x & 31;" << std::endl;
    ks << "int warp_id = threadIdx.x >> 5;" << std::endl;
    ks << "int num_warps = (blockDim.x + 31) >> 5;" << std::endl;
    ks << std::endl;

    // Phase 1: find row max
    ks << "// Phase 1: row max" << std::endl;
    ks << type << " thread_max = -INFINITY;" << std::endl;
    ks << "for (int i = threadIdx.x; i < row_size; i += blockDim.x) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "thread_max = fmax" << (type == "float" ? "f" : "") << "(thread_max, row_in[i]);" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << std::endl;

    // Warp-level max reduction
    ks << "// Warp-level max reduction" << std::endl;
    ks << "for (int mask = 16; mask > 0; mask >>= 1) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "thread_max = fmax" << (type == "float" ? "f" : "")
       << "(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask));" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << std::endl;

    // Cross-warp max reduction
    ks << "// Cross-warp max reduction" << std::endl;
    ks << "if (lane_id == 0) sdata[warp_id] = thread_max;" << std::endl;
    ks << "__syncthreads();" << std::endl;
    ks << type << " row_max = (threadIdx.x < num_warps) ? sdata[threadIdx.x] : (" << type << ")(-INFINITY);"
       << std::endl;
    ks << "for (int mask = 16; mask > 0; mask >>= 1) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "row_max = fmax" << (type == "float" ? "f" : "") << "(row_max, __shfl_xor_sync(0xFFFFFFFF, row_max, mask));"
       << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << "if (threadIdx.x == 0) sdata[0] = row_max;" << std::endl;
    ks << "__syncthreads();" << std::endl;
    ks << "row_max = sdata[0];" << std::endl;
    ks << std::endl;

    // Phase 2: exp and sum
    ks << "// Phase 2: exp(x - max) and sum" << std::endl;
    ks << type << " thread_sum = 0;" << std::endl;
    ks << "for (int i = threadIdx.x; i < row_size; i += blockDim.x) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << type << " val = exp" << (type == "float" ? "f" : "") << "(row_in[i] - row_max);" << std::endl;
    ks << "row_out[i] = val;" << std::endl;
    ks << "thread_sum += val;" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << std::endl;

    // Warp-level sum reduction
    ks << "// Warp-level sum reduction" << std::endl;
    ks << "for (int mask = 16; mask > 0; mask >>= 1) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "thread_sum += __shfl_xor_sync(0xFFFFFFFF, thread_sum, mask);" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << std::endl;

    // Cross-warp sum reduction
    ks << "// Cross-warp sum reduction" << std::endl;
    ks << "if (lane_id == 0) sdata[warp_id] = thread_sum;" << std::endl;
    ks << "__syncthreads();" << std::endl;
    ks << type << " row_sum = (threadIdx.x < num_warps) ? sdata[threadIdx.x] : 0;" << std::endl;
    ks << "for (int mask = 16; mask > 0; mask >>= 1) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask);" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << "if (threadIdx.x == 0) sdata[0] = row_sum;" << std::endl;
    ks << "__syncthreads();" << std::endl;
    ks << "row_sum = sdata[0];" << std::endl;
    ks << std::endl;

    // Phase 3: normalize
    ks << "// Phase 3: normalize" << std::endl;
    ks << "for (int i = threadIdx.x; i < row_size; i += blockDim.x) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "row_out[i] /= row_sum;" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;

    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
}

// Strided softmax kernel: reduced axis is NOT innermost, so a softmax group's elements are
// `inner` apart in memory (layout [outer, reduce, inner]). Each thread owns one entire softmax
// group (one (outer, inner_idx) column). Consecutive threads map to consecutive `inner_idx`
// values, so for a fixed reduction index r all lanes of a warp touch consecutive addresses
// (outer*row_size*inner + r*inner + inner_idx) => fully coalesced global-memory accesses.
static void
emit_softmax_kernel_strided(codegen::PrettyPrinter& ks, const std::string& kernel_name, const std::string& type) {
    std::string fsuf = (type == "float" ? "f" : "");

    ks << "__global__ void " << kernel_name << "(const " << type << "* __restrict__ input, " << type
       << "* __restrict__ output, int num_groups, int row_size, int inner) {" << std::endl;
    ks.setIndent(ks.indent() + 4);

    // Grid-stride loop over softmax groups so any grid size is valid.
    ks << "for (int g = blockIdx.x * blockDim.x + threadIdx.x; g < num_groups; g += gridDim.x * blockDim.x) {"
       << std::endl;
    ks.setIndent(ks.indent() + 4);

    ks << "int __outer = g / inner;" << std::endl;
    ks << "int __inner_idx = g % inner;" << std::endl;
    ks << "const " << type << "* col_in = input + (size_t)__outer * row_size * inner + __inner_idx;" << std::endl;
    ks << type << "* col_out = output + (size_t)__outer * row_size * inner + __inner_idx;" << std::endl;
    ks << std::endl;

    // Phase 1: max over the reduced axis (coalesced across the warp for each r)
    ks << type << " m = -INFINITY;" << std::endl;
    ks << "for (int r = 0; r < row_size; ++r) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "m = fmax" << fsuf << "(m, col_in[(size_t)r * inner]);" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << std::endl;

    // Phase 2: exp(x - max), write to output, accumulate sum
    ks << type << " s = 0;" << std::endl;
    ks << "for (int r = 0; r < row_size; ++r) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << type << " v = exp" << fsuf << "(col_in[(size_t)r * inner] - m);" << std::endl;
    ks << "col_out[(size_t)r * inner] = v;" << std::endl;
    ks << "s += v;" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
    ks << std::endl;

    // Phase 3: normalize
    ks << type << " inv = (" << type << ")1 / s;" << std::endl;
    ks << "for (int r = 0; r < row_size; ++r) {" << std::endl;
    ks.setIndent(ks.indent() + 4);
    ks << "col_out[(size_t)r * inner] *= inv;" << std::endl;
    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;

    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl; // grid-stride loop

    ks.setIndent(ks.indent() - 4);
    ks << "}" << std::endl;
}

static void compute_row_dims(
    const sdfg::math::tensor::SoftmaxNode& node,
    codegen::LanguageExtension& lang,
    std::string& num_rows_str,
    std::string& row_size_str,
    std::string& inner_str
) {
    auto& shape = node.shape();
    auto& axes = node.axes();
    int64_t ndim = static_cast<int64_t>(shape.size());

    // Normalize axes to positive
    std::set<int64_t> reduce_axes;
    for (auto a : axes) {
        reduce_axes.insert(a < 0 ? a + ndim : a);
    }

    // Decompose the (row-major) tensor into (outer, reduce, inner):
    //   - outer:  product of dims before the first reduced axis
    //   - reduce: product of dims spanning the reduced axes (row_size of each softmax group)
    //   - inner:  product of dims after the last reduced axis (memory stride between
    //             consecutive reduced elements)
    // When the reduced axes are trailing (inner == 1) the softmax groups are contiguous in
    // memory; otherwise they are strided by `inner`.
    int64_t reduce_min = ndim;
    int64_t reduce_max = -1;
    for (auto a : reduce_axes) {
        reduce_min = std::min(reduce_min, a);
        reduce_max = std::max(reduce_max, a);
    }

    symbolic::Expression outer = symbolic::one();
    symbolic::Expression reduce = symbolic::one();
    symbolic::Expression inner = symbolic::one();
    for (int64_t i = 0; i < ndim; ++i) {
        if (i < reduce_min) {
            outer = symbolic::mul(outer, shape[i]);
        } else if (i > reduce_max) {
            inner = symbolic::mul(inner, shape[i]);
        } else {
            reduce = symbolic::mul(reduce, shape[i]);
        }
    }

    // num_rows = number of independent softmax groups = outer * inner
    num_rows_str = lang.expression(symbolic::mul(outer, inner));
    row_size_str = lang.expression(reduce);
    inner_str = lang.expression(inner);
}

static std::string get_type_string(types::PrimitiveType prim_type) {
    switch (prim_type) {
        case types::PrimitiveType::Float:
            return "float";
        case types::PrimitiveType::Double:
            return "double";
        default:
            throw std::runtime_error("Unsupported primitive type for CUDA softmax dispatcher");
    }
}

static void dispatch_softmax_common(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    codegen::LanguageExtension& language_extension,
    const sdfg::math::tensor::SoftmaxNode& node,
    const data_flow::DataFlowGraph& data_flow_graph,
    const std::string& input_ptr,
    const std::string& output_ptr
) {
    auto prim_type = node.primitive_type(data_flow_graph);
    std::string type = get_type_string(prim_type);

    std::string num_rows_str, row_size_str, inner_str;
    compute_row_dims(node, language_extension, num_rows_str, row_size_str, inner_str);

    // When the reduced axes are trailing, the softmax groups are contiguous (inner == 1) and we
    // can use the fast contiguous kernel exclusively. Otherwise we also need the strided kernel.
    bool inner_is_one = (inner_str == "1");

    std::string kernel_name = "softmax_kernel_" + std::to_string(node.element_id());
    std::string kernel_name_strided = kernel_name + "_strided";

    out.library_snippet_factory.add_global("#include <cuda.h>");
    out.library_snippet_factory.add_global("#include <math.h>");

    // Forward-declare kernel(s) in globals
    out.globals_stream << "__global__ void " << kernel_name << "(const " << type << "* __restrict__ input, " << type
                       << "* __restrict__ output, int num_rows, int row_size);" << std::endl;
    if (!inner_is_one) {
        out.globals_stream << "__global__ void " << kernel_name_strided << "(const " << type << "* __restrict__ input, "
                           << type << "* __restrict__ output, int num_rows, int row_size, int inner);" << std::endl;
    }

    // Emit kernel(s) to .cu file
    auto& kernel_stream = out.library_snippet_factory.require(kernel_name, "cu", true).stream();
    kernel_stream << "#include " << out.library_snippet_factory.header_path().filename() << std::endl << std::endl;
    emit_softmax_kernel_contiguous(kernel_stream, kernel_name, type);
    if (!inner_is_one) {
        kernel_stream << std::endl;
        emit_softmax_kernel_strided(kernel_stream, kernel_name_strided, type);
    }

    // Emit kernel call
    out.stream << "{" << std::endl;
    out.stream.setIndent(out.stream.indent() + 4);

    out.stream << "int __softmax_num_rows = (int)(" << num_rows_str << ");" << std::endl;
    out.stream << "int __softmax_row_size = (int)(" << row_size_str << ");" << std::endl;
    if (!inner_is_one) {
        out.stream << "int __softmax_inner = (int)(" << inner_str << ");" << std::endl;
    }

    // Launch config for the contiguous kernel: one block per row, block sized to the row.
    out.stream << "int __softmax_block_size = " << SOFTMAX_BLOCK_SIZE << ";" << std::endl;
    out.stream << "if (__softmax_row_size < __softmax_block_size) __softmax_block_size = __softmax_row_size;"
               << std::endl;
    // Round up to multiple of 32 (warp size)
    out.stream << "__softmax_block_size = ((__softmax_block_size + 31) / 32) * 32;" << std::endl;
    out.stream << "int __softmax_num_warps = __softmax_block_size / 32;" << std::endl;
    out.stream << "size_t __softmax_smem = __softmax_num_warps * sizeof(" << type << ");" << std::endl;

    auto emit_contiguous_launch = [&]() {
        out.stream << kernel_name << "<<<__softmax_num_rows, __softmax_block_size, __softmax_smem>>>(" << input_ptr
                   << ", " << output_ptr << ", __softmax_num_rows, __softmax_row_size);" << std::endl;
    };

    if (inner_is_one) {
        emit_contiguous_launch();
    } else {
        // Runtime dispatch: prefer the fast contiguous kernel whenever the stride collapses to 1.
        out.stream << "if (__softmax_inner == 1) {" << std::endl;
        out.stream.setIndent(out.stream.indent() + 4);
        emit_contiguous_launch();
        out.stream.setIndent(out.stream.indent() - 4);
        out.stream << "} else {" << std::endl;
        out.stream.setIndent(out.stream.indent() + 4);
        // Coalesced strided kernel: one thread per softmax group, grid-stride over groups.
        out.stream << "int __softmax_strided_block = " << SOFTMAX_BLOCK_SIZE << ";" << std::endl;
        out.stream << "int __softmax_strided_grid = (__softmax_num_rows + __softmax_strided_block - 1) / "
                      "__softmax_strided_block;"
                   << std::endl;
        out.stream << kernel_name_strided << "<<<__softmax_strided_grid, __softmax_strided_block>>>(" << input_ptr
                   << ", " << output_ptr << ", __softmax_num_rows, __softmax_row_size, __softmax_inner);" << std::endl;
        out.stream.setIndent(out.stream.indent() - 4);
        out.stream << "}" << std::endl;
    }

    check_cuda_kernel_launch_errors(out.stream, language_extension, false);

    out.stream.setIndent(out.stream.indent() - 4);
    out.stream << "}" << std::endl;
}

// WithTransfers

SoftmaxNodeDispatcher_CUDAWithTransfers::SoftmaxNodeDispatcher_CUDAWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::SoftmaxNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void SoftmaxNodeDispatcher_CUDAWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::math::tensor::SoftmaxNode&>(this->node_);
    auto prim_type = node.primitive_type(this->data_flow_graph_);
    std::string type = get_type_string(prim_type);

    // Connectors: inputs_={"Y", "X"} → inputs[0]=Y (output buffer), inputs[1]=X (input data)
    auto& y_expr = inputs.at(0).expr;
    auto& x_expr = inputs.at(1).expr;

    std::string num_rows_str, row_size_str, inner_str;
    compute_row_dims(node, this->language_extension_, num_rows_str, row_size_str, inner_str);

    std::string total_size = "((size_t)(" + num_rows_str + ") * (size_t)(" + row_size_str + ")) * sizeof(" + type + ")";

    out.stream << "{" << std::endl;
    out.stream.setIndent(out.stream.indent() + 4);

    out.stream << "cudaError_t err_cuda;" << std::endl;
    out.stream << type << " *d_input, *d_output;" << std::endl;
    out.stream << "size_t __softmax_total_bytes = " << total_size << ";" << std::endl;

    out.stream << "err_cuda = cudaMalloc((void**) &d_input, __softmax_total_bytes);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaMalloc((void**) &d_output, __softmax_total_bytes);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaMemcpy(d_input, " << x_expr << ", __softmax_total_bytes, cudaMemcpyHostToDevice);"
               << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    dispatch_softmax_common(out, inputs, this->language_extension_, node, this->data_flow_graph_, "d_input", "d_output");

    out.stream << "err_cuda = cudaMemcpy(" << y_expr << ", d_output, __softmax_total_bytes, cudaMemcpyDeviceToHost);"
               << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    out.stream << "err_cuda = cudaFree(d_input);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");
    out.stream << "err_cuda = cudaFree(d_output);" << std::endl;
    cuda_error_checking(out.stream, this->language_extension_, "err_cuda");

    out.stream.setIndent(out.stream.indent() - 4);
    out.stream << "}" << std::endl;
}

codegen::InstrumentationInfo SoftmaxNodeDispatcher_CUDAWithTransfers::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        TargetType_CUDA,
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

// WithoutTransfers

SoftmaxNodeDispatcher_CUDAWithoutTransfers::SoftmaxNodeDispatcher_CUDAWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::math::tensor::SoftmaxNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void SoftmaxNodeDispatcher_CUDAWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    auto& node = static_cast<const sdfg::math::tensor::SoftmaxNode&>(this->node_);

    // Connectors: inputs_={"Y", "X"} → inputs[0]=Y (output buffer), inputs[1]=X (input data)
    auto& y_expr = inputs.at(0).expr;
    auto& x_expr = inputs.at(1).expr;

    dispatch_softmax_common(out, inputs, this->language_extension_, node, this->data_flow_graph_, x_expr, y_expr);
}

codegen::InstrumentationInfo SoftmaxNodeDispatcher_CUDAWithoutTransfers::instrumentation_info() const {
    return {
        node_.element_id(),
        std::string(node_.element_type()) + ":::" + node_.code().value(),
        TargetType_CUDA,
        codegen::InstrumentationEventType::CUDA,
        analysis::LoopInfo{},
        {}
    };
}

} // namespace sdfg::cuda::tensor
