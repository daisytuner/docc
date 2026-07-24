#include "sdfg/targets/cuda/math/tensor/c2r_fft2d.h"

#include <string>
#include <vector>

#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/math/tensor/fft2d_common.h"

namespace sdfg::cuda::tensor {

using math::tensor::C2RFFT2DNode;

namespace {

// Emit the inverse C2R 2D FFT launch sequence. When `with_transfers` is true the
// operands (X complex, Y real) are host buffers and matching device copies are made;
// otherwise they are already device-resident and used directly. The column pass runs
// in place on the (transient) complex input.
void emit_c2r_fft2d(
    codegen::CodegenOutput& out,
    codegen::LanguageExtension& lang,
    const C2RFFT2DNode& node,
    std::vector<codegen::DispatchInput>& inputs,
    bool with_transfers
) {
    if (node.real_primitive() != types::PrimitiveType::Float) {
        throw std::runtime_error("C2RFFT2D CUDA dispatcher supports single precision only");
    }

    const int64_t M = fft2d::const_int(node.shape()[0]);
    const int64_t fftH = fft2d::const_int(node.shape()[1]);
    const int64_t fftW = fft2d::const_int(node.shape()[2]);
    const auto radW = fft2d::factor_radices(fftW);
    const auto radH = fft2d::factor_radices(fftH);

    const std::string y_ptr = inputs.at(C2RFFT2DNode::Y_INPUT_IDX).expr; // real output
    const std::string x_ptr = inputs.at(C2RFFT2DNode::X_INPUT_IDX).expr; // complex half-spectrum input

    const std::string prefix = "__c2rfft_" + std::to_string(node.element_id());
    fft2d::emit_fft2d_kernels(out, prefix);

    const std::string dX = with_transfers ? std::string("__c2r_dX") : ("reinterpret_cast<float2*>(" + x_ptr + ")");
    const std::string dY = with_transfers ? std::string("__c2r_dY") : ("reinterpret_cast<float*>(" + y_ptr + ")");

    auto& s = out.stream;
    s << "{" << std::endl;
    s.setIndent(s.indent() + 4);

    s << "const int __c2r_M = " << M << ", __c2r_fftH = " << fftH << ", __c2r_fftW = " << fftW << ";" << std::endl;
    s << "const int __c2r_halfW = __c2r_fftW / 2 + 1;" << std::endl;
    s << "const int __c2r_real_elems = __c2r_M * __c2r_fftH * __c2r_fftW;" << std::endl;
    s << "const int __c2r_spec_elems = __c2r_M * __c2r_fftH * __c2r_halfW;" << std::endl;

    auto emit_radix = [&](const std::string& name, const std::vector<int>& r) {
        s << "int " << name << "[" << r.size() << "] = {";
        for (size_t i = 0; i < r.size(); ++i) {
            s << r[i] << (i + 1 < r.size() ? ", " : "");
        }
        s << "};" << std::endl;
    };
    emit_radix("__c2r_hRadW", radW);
    emit_radix("__c2r_hRadH", radH);
    s << "const int __c2r_nRadW = " << radW.size() << ", __c2r_nRadH = " << radH.size() << ";" << std::endl;

    s << "cudaError_t err_cuda;" << std::endl;
    if (with_transfers) {
        s << "float2 *__c2r_dX; float *__c2r_dY;" << std::endl;
    }
    s << "int *__c2r_dRadW, *__c2r_dRadH;" << std::endl;

    auto malloc_check = [&](const std::string& var, const std::string& bytes) {
        s << "err_cuda = cudaMalloc((void**) &" << var << ", " << bytes << ");" << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    };
    if (with_transfers) {
        malloc_check("__c2r_dX", "__c2r_spec_elems * sizeof(float2)");
        malloc_check("__c2r_dY", "__c2r_real_elems * sizeof(float)");
    }
    malloc_check("__c2r_dRadW", "__c2r_nRadW * sizeof(int)");
    malloc_check("__c2r_dRadH", "__c2r_nRadH * sizeof(int)");

    auto h2d = [&](const std::string& dst, const std::string& src, const std::string& bytes) {
        s << "err_cuda = cudaMemcpy(" << dst << ", " << src << ", " << bytes << ", cudaMemcpyHostToDevice);"
          << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    };
    if (with_transfers) {
        h2d("__c2r_dX", x_ptr, "__c2r_spec_elems * sizeof(float2)");
    }
    h2d("__c2r_dRadW", "__c2r_hRadW", "__c2r_nRadW * sizeof(int)");
    h2d("__c2r_dRadH", "__c2r_hRadH", "__c2r_nRadH * sizeof(int)");

    // Launch geometry (mirrors the fused fft_conv heuristics).
    s << "int __c2r_threads = 256;" << std::endl;
    s << "int __c2r_rowLines = 16;" << std::endl;
    s << "while (__c2r_rowLines > 1 && (size_t)4 * __c2r_rowLines * __c2r_fftW * sizeof(float) > 32768) "
         "__c2r_rowLines >>= 1;"
      << std::endl;
    s << "size_t __c2r_rowSmem = (size_t)4 * __c2r_rowLines * __c2r_fftW * sizeof(float);" << std::endl;
    s << "int __c2r_colTile = 32;" << std::endl;
    s << "while (__c2r_colTile > 4 && (size_t)4 * __c2r_fftH * __c2r_colTile * sizeof(float) > 32768) "
         "__c2r_colTile >>= 1;"
      << std::endl;
    s << "int __c2r_colLdm = (__c2r_halfW < __c2r_colTile) ? __c2r_colTile + 1 : __c2r_colTile;" << std::endl;
    s << "size_t __c2r_colSmem = (size_t)4 * __c2r_fftH * __c2r_colLdm * sizeof(float);" << std::endl;
    s << "cudaFuncSetAttribute(" << prefix
      << "_fftRowsC2R, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)__c2r_rowSmem);" << std::endl;
    s << "cudaFuncSetAttribute(" << prefix
      << "_fftCols, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)__c2r_colSmem);" << std::endl;

    s << "int __c2r_rows = __c2r_M * __c2r_fftH;" << std::endl;
    s << "int __c2r_row_blocks = (__c2r_rows + __c2r_rowLines - 1) / __c2r_rowLines;" << std::endl;
    s << "int __c2r_colTilesHalf = (__c2r_halfW + __c2r_colTile - 1) / __c2r_colTile;" << std::endl;
    s << "int __c2r_col_blocks = __c2r_M * __c2r_colTilesHalf;" << std::endl;

    // Inverse: column FFT (in place, scaled 1/fftH), then C2R rows write the real output.
    s << prefix << "_fftCols<<<__c2r_col_blocks, __c2r_threads, __c2r_colSmem>>>(" << dX
      << ", __c2r_M, __c2r_fftH, __c2r_halfW, __c2r_halfW, __c2r_dRadH, __c2r_nRadH, 1.0f, 1, __c2r_colTile, "
         "__c2r_colLdm);"
      << std::endl;
    s << prefix << "_fftRowsC2R<<<__c2r_row_blocks, __c2r_threads, __c2r_rowSmem>>>(" << dY << ", " << dX
      << ", __c2r_rows, __c2r_fftW, __c2r_halfW, __c2r_dRadW, __c2r_nRadW, __c2r_rowLines);" << std::endl;
    check_cuda_kernel_launch_errors(s, lang, false);

    if (with_transfers) {
        s << "err_cuda = cudaMemcpy(" << y_ptr
          << ", __c2r_dY, __c2r_real_elems * sizeof(float), "
             "cudaMemcpyDeviceToHost);"
          << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    }

    for (const std::string& var : {"__c2r_dRadW", "__c2r_dRadH"}) {
        s << "err_cuda = cudaFree(" << var << ");" << std::endl;
        cuda_error_checking(s, lang, "err_cuda");
    }
    if (with_transfers) {
        for (const std::string& var : {"__c2r_dX", "__c2r_dY"}) {
            s << "err_cuda = cudaFree(" << var << ");" << std::endl;
            cuda_error_checking(s, lang, "err_cuda");
        }
    }

    s.setIndent(s.indent() - 4);
    s << "}" << std::endl;
}

} // namespace

C2RFFT2DNodeDispatcher_CUDAWithTransfers::C2RFFT2DNodeDispatcher_CUDAWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const C2RFFT2DNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void C2RFFT2DNodeDispatcher_CUDAWithTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    emit_c2r_fft2d(out, this->language_extension_, static_cast<const C2RFFT2DNode&>(this->node_), inputs, true);
}

C2RFFT2DNodeDispatcher_CUDAWithoutTransfers::C2RFFT2DNodeDispatcher_CUDAWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const C2RFFT2DNode& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void C2RFFT2DNodeDispatcher_CUDAWithoutTransfers::dispatch_code_with_edges(
    codegen::CodegenOutput& out,
    std::vector<codegen::DispatchInput>& inputs,
    std::vector<codegen::DispatchOutput>& outputs
) {
    emit_c2r_fft2d(out, this->language_extension_, static_cast<const C2RFFT2DNode&>(this->node_), inputs, false);
}

} // namespace sdfg::cuda::tensor
