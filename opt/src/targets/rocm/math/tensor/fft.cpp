#include "sdfg/targets/rocm/math/tensor/fft.h"

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/types/type.h"

namespace sdfg::rocm::tensor {

using math::tensor::FFTDirection;
using math::tensor::FFTNodeBase;

void hipfft_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_rocm_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != HIPFFT_SUCCESS) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix() << "fprintf(stderr, \"hipFFT error: %d File: %s, Line: %d\\n\", "
           << status_variable << ", __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

namespace {

// Per-transform real element count: prod(n_i).
symbolic::Expression per_transform_real(const FFTNodeBase& node) {
    symbolic::Expression extent = symbolic::one();
    for (const auto& dim : node.shape()) {
        extent = symbolic::mul(extent, dim);
    }
    return extent;
}

// Per-transform complex element count (Hermitian): prod_{i<d-1}(n_i) * (n_{d-1}/2 + 1).
symbolic::Expression per_transform_complex(const FFTNodeBase& node) {
    symbolic::Expression extent = symbolic::one();
    const auto& shape = node.shape();
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        extent = symbolic::mul(extent, shape[i]);
    }
    return symbolic::mul(extent, node.complex_last_dim());
}

} // namespace

void generate_fft(
    codegen::PrettyPrinter& stream,
    codegen::LanguageExtension& language_extension,
    codegen::CodeSnippetFactory& library_snippet_factory,
    const FFTNodeBase& node,
    bool with_transfers
) {
    library_snippet_factory.add_global("#include <hip/hip_runtime.h>");
    library_snippet_factory.add_global("#include <hipfft/hipfft.h>");

    const bool forward = node.direction() == FFTDirection::Forward;
    const bool is_double = node.real_primitive() == types::PrimitiveType::Double;

    const std::string real_t = is_double ? "double" : "float";
    const std::string cplx_t = is_double ? "hipfftDoubleComplex" : "hipfftComplex";
    const std::string real_cast = is_double ? "(hipfftDoubleReal*)" : "(hipfftReal*)";
    const std::string cplx_cast = is_double ? "(hipfftDoubleComplex*)" : "(hipfftComplex*)";
    const std::string plan_type = forward ? (is_double ? "HIPFFT_D2Z" : "HIPFFT_R2C")
                                          : (is_double ? "HIPFFT_Z2D" : "HIPFFT_C2R");
    const std::string exec_fn = forward ? (is_double ? "hipfftExecD2Z" : "hipfftExecR2C")
                                        : (is_double ? "hipfftExecZ2D" : "hipfftExecC2R");

    const auto per_real = per_transform_real(node);
    const auto per_cplx = per_transform_complex(node);
    const auto batch = node.batch();

    const auto idist = forward ? per_real : per_cplx;
    const auto odist = forward ? per_cplx : per_real;

    const auto total_real = symbolic::mul(batch, per_real);
    const auto total_cplx = symbolic::mul(batch, per_cplx);
    const std::string x_dev_t = forward ? real_t : cplx_t;
    const std::string y_dev_t = forward ? cplx_t : real_t;
    const std::string x_count = language_extension.expression(forward ? total_real : total_cplx);
    const std::string y_count = language_extension.expression(forward ? total_cplx : total_real);

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    stream << "int __fft_n[" << node.rank() << "] = {";
    for (size_t i = 0; i < node.shape().size(); ++i) {
        stream << language_extension.expression(node.shape()[i]);
        if (i + 1 < node.shape().size()) {
            stream << ", ";
        }
    }
    stream << "};" << std::endl;

    std::string in_ptr = "__X";
    std::string out_ptr = "__Y";

    if (with_transfers) {
        stream << "hipError_t err_hip;" << std::endl;
        stream << x_dev_t << " *__fft_dX;" << std::endl;
        stream << y_dev_t << " *__fft_dY;" << std::endl;
        stream << "err_hip = hipMalloc((void**) &__fft_dX, (" << x_count << ") * sizeof(" << x_dev_t << "));"
               << std::endl;
        rocm_error_checking(stream, language_extension, "err_hip");
        stream << "err_hip = hipMalloc((void**) &__fft_dY, (" << y_count << ") * sizeof(" << y_dev_t << "));"
               << std::endl;
        rocm_error_checking(stream, language_extension, "err_hip");
        stream << "err_hip = hipMemcpy(__fft_dX, __X, (" << x_count << ") * sizeof(" << x_dev_t
               << "), hipMemcpyHostToDevice);" << std::endl;
        rocm_error_checking(stream, language_extension, "err_hip");
        in_ptr = "__fft_dX";
        out_ptr = "__fft_dY";
    }

    stream << "hipfftHandle __fft_plan;" << std::endl;
    stream << "hipfftResult __fft_status;" << std::endl;
    stream << "__fft_status = hipfftPlanMany(&__fft_plan, " << node.rank() << ", __fft_n, NULL, 1, "
           << language_extension.expression(idist) << ", NULL, 1, " << language_extension.expression(odist) << ", "
           << plan_type << ", " << language_extension.expression(batch) << ");" << std::endl;
    hipfft_error_checking(stream, language_extension, "__fft_status");

    const std::string in_cast = forward ? real_cast : cplx_cast;
    const std::string out_cast = forward ? cplx_cast : real_cast;
    stream << "__fft_status = " << exec_fn << "(__fft_plan, " << in_cast << in_ptr << ", " << out_cast << out_ptr
           << ");" << std::endl;
    hipfft_error_checking(stream, language_extension, "__fft_status");

    stream << "hipfftDestroy(__fft_plan);" << std::endl;

    if (with_transfers) {
        stream << "err_hip = hipMemcpy(__Y, __fft_dY, (" << y_count << ") * sizeof(" << y_dev_t
               << "), hipMemcpyDeviceToHost);" << std::endl;
        rocm_error_checking(stream, language_extension, "err_hip");
        stream << "err_hip = hipFree(__fft_dX);" << std::endl;
        rocm_error_checking(stream, language_extension, "err_hip");
        stream << "err_hip = hipFree(__fft_dY);" << std::endl;
        rocm_error_checking(stream, language_extension, "err_hip");
    } else {
        stream << "hipError_t err_hip_sync = hipDeviceSynchronize();" << std::endl;
        rocm_error_checking(stream, language_extension, "err_hip_sync");
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

FFTNodeDispatcher_HIPFFTWithTransfers::FFTNodeDispatcher_HIPFFTWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FFTNodeBase& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FFTNodeDispatcher_HIPFFTWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const FFTNodeBase&>(this->node_);
    generate_fft(stream, this->language_extension_, library_snippet_factory, node, /*with_transfers=*/true);
}

FFTNodeDispatcher_HIPFFTWithoutTransfers::FFTNodeDispatcher_HIPFFTWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FFTNodeBase& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FFTNodeDispatcher_HIPFFTWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const FFTNodeBase&>(this->node_);
    generate_fft(stream, this->language_extension_, library_snippet_factory, node, /*with_transfers=*/false);
}

} // namespace sdfg::rocm::tensor
