#include "sdfg/targets/cuda/math/tensor/r2c_fft2d.h"

#include <string>
#include <vector>

#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/math/tensor/fft2d_common.h"

namespace sdfg::cuda::tensor {

using math::tensor::R2CFFT2DNode;

namespace {

// Emit the forward R2C 2D FFT launch sequence. When `with_transfers` is true the
// operands (X real, Y complex) are host buffers and matching device copies are made;
// otherwise they are already device-resident and used directly.
void emit_r2c_fft2d(
    codegen::CodegenOutput& out,
    codegen::LanguageExtension& lang,
    const R2CFFT2DNode& node,
    std::vector<codegen::DispatchInput>& inputs,
    bool with_transfers
) {
    if (node.real_primitive() != types::PrimitiveType::Float) {
        throw std::runtime_error("R2CFFT2D CUDA dispatcher supports single precision only");
    }

    const int64_t M = fft2d::const_int(node.shape()[0]);
    const int64_t fftH = fft2d::const_int(node.shape()[1]);
    const int64_t fftW = fft2d::const_int(node.shape()[2]);
    const auto radW = fft2d::factor_radices(fftW);
    const auto radH = fft2d::factor_radices(fftH);

    const std::string y_ptr = inputs.at(R2CFFT2DNode::Y_INPUT_IDX).expr; // complex half-spectrum output
    const std::string x_ptr = inputs.at(R2CFFT2DNode::X_INPUT_IDX).expr; // real padded input

    const std::string prefix = "__r2cfft_" + std::to_string(node.element_id());
    fft2d::emit_fft2d_kernels(out, prefix);

    const std::string dX = with_transfers ? std::string("__r2c_dX") : ("reinterpret_cast<const float*>(" + x_ptr + ")");
    const std::string dY = with_transfers ? std::string("__r2c_dY") : ("reinterpret_cast<float2*>(" + y_ptr + ")");

    auto& s = out.stream;
    s << "{" << std::endl;
    s.setIndent(s.indent() + 4);

    s << "const int __r2c_M = " << M << ", __r2c_fftH = " << fftH << ", __r2c_fftW = " << fftW << ";" << std::endl;
    s << "const int __r2c_halfW = __r2c_fftW / 2 + 1;" << std::endl;
    s << "const int __r2c_real_elems = __r2c_M * __r2c_fftH * __r2c_fftW;" << std::endl;
    s << "const int __r2c_spec_elems = __r2c_M * __r2c_fftH * __r2c_halfW;" << std::endl;

    auto emit_radix = [&](const std::string& name, const std::vector<int>& r) {
        s << "int " << name << "[" << r.size() << "] = {";
        for (size_t i = 0; i < r.size(); ++i) {
            s << r[i] << (i + 1 < r.size() ? ", " : "");
        }
        s << "};" << std::endl;
    };
    emit_radix("__r2c_hRadW", radW);
    emit_radix("__r2c_hRadH", radH);
    s << "const int __r2c_nRadW = " << radW.size() << ", __r2c_nRadH = " << radH.size() << ";" << std::endl;

    s << "cudaError_t err_cuda;" << std::endl;
    if (with_transfers) {
        s << "float *__r2c_dX; float2 *__r2c_dY;" << std::endl;
    }
    s << "int *__r2c_dRadW, *__r2c_dRadH;" << std::endl;

    auto malloc_check = [&](const std::string& var, const std::string& bytes) {
        s << "err_cuda = cudaMalloc((void**) &" << var << ", " << bytes << ");" << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    };
    if (with_transfers) {
        malloc_check("__r2c_dX", "__r2c_real_elems * sizeof(float)");
        malloc_check("__r2c_dY", "__r2c_spec_elems * sizeof(float2)");
    }
    malloc_check("__r2c_dRadW", "__r2c_nRadW * sizeof(int)");
    malloc_check("__r2c_dRadH", "__r2c_nRadH * sizeof(int)");

    auto h2d = [&](const std::string& dst, const std::string& src, const std::string& bytes) {
        s << "err_cuda = cudaMemcpy(" << dst << ", " << src << ", " << bytes << ", cudaMemcpyHostToDevice);"
          << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    };
    if (with_transfers) {
        h2d("__r2c_dX", x_ptr, "__r2c_real_elems * sizeof(float)");
    }
    h2d("__r2c_dRadW", "__r2c_hRadW", "__r2c_nRadW * sizeof(int)");
    h2d("__r2c_dRadH", "__r2c_hRadH", "__r2c_nRadH * sizeof(int)");

    // Launch geometry (mirrors the fused fft_conv heuristics).
    s << "int __r2c_threads = 256;" << std::endl;
    s << "int __r2c_rowLines = 16;" << std::endl;
    s << "while (__r2c_rowLines > 1 && (size_t)4 * __r2c_rowLines * __r2c_fftW * sizeof(float) > 32768) "
         "__r2c_rowLines >>= 1;"
      << std::endl;
    s << "size_t __r2c_rowSmem = (size_t)4 * __r2c_rowLines * __r2c_fftW * sizeof(float);" << std::endl;
    s << "int __r2c_colTile = 32;" << std::endl;
    s << "while (__r2c_colTile > 4 && (size_t)4 * __r2c_fftH * __r2c_colTile * sizeof(float) > 32768) "
         "__r2c_colTile >>= 1;"
      << std::endl;
    s << "int __r2c_colLdm = (__r2c_halfW < __r2c_colTile) ? __r2c_colTile + 1 : __r2c_colTile;" << std::endl;
    s << "size_t __r2c_colSmem = (size_t)4 * __r2c_fftH * __r2c_colLdm * sizeof(float);" << std::endl;
    s << "cudaFuncSetAttribute(" << prefix
      << "_fftRowsR2C, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)__r2c_rowSmem);" << std::endl;
    s << "cudaFuncSetAttribute(" << prefix
      << "_fftCols, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)__r2c_colSmem);" << std::endl;

    s << "int __r2c_rows = __r2c_M * __r2c_fftH;" << std::endl;
    s << "int __r2c_row_blocks = (__r2c_rows + __r2c_rowLines - 1) / __r2c_rowLines;" << std::endl;
    s << "int __r2c_colTilesHalf = (__r2c_halfW + __r2c_colTile - 1) / __r2c_colTile;" << std::endl;
    s << "int __r2c_col_blocks = __r2c_M * __r2c_colTilesHalf;" << std::endl;

    // Forward: R2C rows (real -> half-spectrum complex), then column FFT in place.
    s << prefix << "_fftRowsR2C<<<__r2c_row_blocks, __r2c_threads, __r2c_rowSmem>>>(" << dY << ", " << dX
      << ", __r2c_rows, __r2c_fftW, __r2c_halfW, __r2c_dRadW, __r2c_nRadW, __r2c_rowLines);" << std::endl;
    s << prefix << "_fftCols<<<__r2c_col_blocks, __r2c_threads, __r2c_colSmem>>>(" << dY
      << ", __r2c_M, __r2c_fftH, __r2c_halfW, __r2c_halfW, __r2c_dRadH, __r2c_nRadH, -1.0f, 0, __r2c_colTile, "
         "__r2c_colLdm);"
      << std::endl;
    check_cuda_kernel_launch_errors(s, lang, false);

    if (with_transfers) {
        s << "err_cuda = cudaMemcpy(" << y_ptr
          << ", __r2c_dY, __r2c_spec_elems * sizeof(float2), "
             "cudaMemcpyDeviceToHost);"
          << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    }

    for (const std::string& var : {"__r2c_dRadW", "__r2c_dRadH"}) {
        s << "err_cuda = cudaFree(" << var << ");" << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    }
    if (with_transfers) {
        for (const std::string& var : {"__r2c_dX", "__r2c_dY"}) {
            s << "err_cuda = cudaFree(" << var << ");" << std::endl;
            cuda_error_checking(s, lang, "err_cuda");
        }
    }

    s.setIndent(s.indent() - 4);
    s << "}" << std::endl;
}

} // namespace

R2CFFT2DNodeDispatcher_CUDAWithTransfers::R2CFFT2DNodeDispatcher_CUDAWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const R2CFFT2DNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void R2CFFT2DNodeDispatcher_CUDAWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    emit_r2c_fft2d(out, this->language_extension_, static_cast<const R2CFFT2DNode&>(this->node_), inputs, true);
}

R2CFFT2DNodeDispatcher_CUDAWithoutTransfers::R2CFFT2DNodeDispatcher_CUDAWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const R2CFFT2DNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void R2CFFT2DNodeDispatcher_CUDAWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    emit_r2c_fft2d(out, this->language_extension_, static_cast<const R2CFFT2DNode&>(this->node_), inputs, false);
}

} // namespace sdfg::cuda::tensor
