#include "sdfg/targets/cuda/math/tensor/softmax.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"

namespace sdfg::cuda::tensor {

static constexpr int SOFTMAX_BLOCK_SIZE = 256;

static void emit_softmax_kernel(codegen::PrettyPrinter& ks, const std::string& kernel_name, const std::string& type) {
    // Fused softmax kernel: one block per row, warp-shuffle reductions
    ks << "__global__ void " << kernel_name << "(const " << type << "* __restrict__ input, " << type
       << "* __restrict__ output, int num_rows, int row_size) {" << std::endl;
    ks.setIndent(ks.indent() + 4);

    ks << "int row = blockIdx.x;" << std::endl;
    ks << "if (row >= num_rows) return;" << std::endl;
    ks << std::endl;
    ks << "const " << type << "* row_in = input + row * row_size;" << std::endl;
    ks << type << "* row_out = output + row * row_size;" << std::endl;
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

static void compute_row_dims(
    const sdfg::math::tensor::SoftmaxNode& node,
    codegen::LanguageExtension& lang,
    std::string& num_rows_str,
    std::string& row_size_str
) {
    auto& shape = node.shape();
    auto& axes = node.axes();
    int64_t ndim = static_cast<int64_t>(shape.size());

    // Normalize axes to positive
    std::set<int64_t> reduce_axes;
    for (auto a : axes) {
        reduce_axes.insert(a < 0 ? a + ndim : a);
    }

    // num_rows = product of non-reduced dims, row_size = product of reduced dims
    symbolic::Expression num_rows = symbolic::one();
    symbolic::Expression row_size = symbolic::one();
    for (int64_t i = 0; i < ndim; ++i) {
        if (reduce_axes.count(i)) {
            row_size = symbolic::mul(row_size, shape[i]);
        } else {
            num_rows = symbolic::mul(num_rows, shape[i]);
        }
    }

    num_rows_str = lang.expression(num_rows);
    row_size_str = lang.expression(row_size);
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

    std::string num_rows_str, row_size_str;
    compute_row_dims(node, language_extension, num_rows_str, row_size_str);

    std::string kernel_name = "softmax_kernel_" + std::to_string(node.element_id());

    out.library_snippet_factory.add_global("#include <cuda.h>");
    out.library_snippet_factory.add_global("#include <math.h>");

    // Forward-declare kernel in globals
    out.globals_stream << "__global__ void " << kernel_name << "(const " << type << "* __restrict__ input, " << type
                       << "* __restrict__ output, int num_rows, int row_size);" << std::endl;

    // Emit kernel to .cu file
    auto& kernel_stream = out.library_snippet_factory.require(kernel_name, "cu", true).stream();
    kernel_stream << "#include " << out.library_snippet_factory.header_path().filename() << std::endl << std::endl;
    emit_softmax_kernel(kernel_stream, kernel_name, type);

    // Emit kernel call
    out.stream << "{" << std::endl;
    out.stream.setIndent(out.stream.indent() + 4);

    out.stream << "int __softmax_num_rows = (int)(" << num_rows_str << ");" << std::endl;
    out.stream << "int __softmax_row_size = (int)(" << row_size_str << ");" << std::endl;
    out.stream << "int __softmax_block_size = " << SOFTMAX_BLOCK_SIZE << ";" << std::endl;
    out.stream << "if (__softmax_row_size < __softmax_block_size) __softmax_block_size = __softmax_row_size;"
               << std::endl;
    // Round up to multiple of 32 (warp size)
    out.stream << "__softmax_block_size = ((__softmax_block_size + 31) / 32) * 32;" << std::endl;
    out.stream << "int __softmax_num_warps = __softmax_block_size / 32;" << std::endl;
    out.stream << "size_t __softmax_smem = __softmax_num_warps * sizeof(" << type << ");" << std::endl;
    out.stream << kernel_name << "<<<__softmax_num_rows, __softmax_block_size, __softmax_smem>>>(" << input_ptr << ", "
               << output_ptr << ", __softmax_num_rows, __softmax_row_size);" << std::endl;

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

    std::string num_rows_str, row_size_str;
    compute_row_dims(node, this->language_extension_, num_rows_str, row_size_str);

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

} // namespace sdfg::cuda::tensor
