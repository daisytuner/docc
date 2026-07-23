#include "sdfg/targets/cuda/math/tensor/fft.h"

#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/types/type.h"

namespace sdfg::cuda::tensor {

using math::tensor::FFTDirection;
using math::tensor::FFTNodeBase;

void cufft_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_cuda_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != CUFFT_SUCCESS) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix() << "fprintf(stderr, \"cuFFT error: %d File: %s, Line: %d\\n\", "
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
    library_snippet_factory.add_global("#include <cuda.h>");
    library_snippet_factory.add_global("#include <cufft.h>");

    const bool forward = node.direction() == FFTDirection::Forward;
    const bool is_double = node.real_primitive() == types::PrimitiveType::Double;

    const std::string real_t = is_double ? "double" : "float";
    const std::string cplx_t = is_double ? "cufftDoubleComplex" : "cufftComplex";
    const std::string real_cast = is_double ? "(cufftDoubleReal*)" : "(cufftReal*)";
    const std::string cplx_cast = is_double ? "(cufftDoubleComplex*)" : "(cufftComplex*)";
    const std::string plan_type = forward ? (is_double ? "CUFFT_D2Z" : "CUFFT_R2C")
                                          : (is_double ? "CUFFT_Z2D" : "CUFFT_C2R");
    const std::string exec_fn = forward ? (is_double ? "cufftExecD2Z" : "cufftExecR2C")
                                        : (is_double ? "cufftExecZ2D" : "cufftExecC2R");

    const auto per_real = per_transform_real(node);
    const auto per_cplx = per_transform_complex(node);
    const auto batch = node.batch();

    // Batch-strides for the advanced cuFFT layout.
    const auto idist = forward ? per_real : per_cplx;
    const auto odist = forward ? per_cplx : per_real;

    // Total element counts on the input (__X) and output (__Y) connectors.
    const auto total_real = symbolic::mul(batch, per_real);
    const auto total_cplx = symbolic::mul(batch, per_cplx);
    const std::string x_dev_t = forward ? real_t : cplx_t;
    const std::string y_dev_t = forward ? cplx_t : real_t;
    const std::string x_count = language_extension.expression(forward ? total_real : total_cplx);
    const std::string y_count = language_extension.expression(forward ? total_cplx : total_real);

    // Build the cuFFT plan in function setup (outside the per-call hot path) and destroy it in
    // teardown, mirroring the cuBLAS handle lifecycle: created once at function entry, released
    // at function exit, so only the cheap cufftExec* runs in the timed hot path.
    const std::string plan_var = "__fft_plan_" + std::to_string(node.element_id());
    {
        codegen::PrettyPrinter setup;
        setup << "cufftHandle " << plan_var << ";" << std::endl;
        setup << "{" << std::endl;
        setup.setIndent(setup.indent() + 4);
        setup << "int " << plan_var << "_n[" << node.rank() << "] = {";
        for (size_t i = 0; i < node.shape().size(); ++i) {
            setup << language_extension.expression(node.shape()[i]);
            if (i + 1 < node.shape().size()) {
                setup << ", ";
            }
        }
        setup << "};" << std::endl;
        setup << "cufftResult " << plan_var << "_st = cufftPlanMany(&" << plan_var << ", " << node.rank() << ", "
              << plan_var << "_n, NULL, 1, " << language_extension.expression(idist) << ", NULL, 1, "
              << language_extension.expression(odist) << ", " << plan_type << ", "
              << language_extension.expression(batch) << ");" << std::endl;
        cufft_error_checking(setup, language_extension, plan_var + "_st");
        setup.setIndent(setup.indent() - 4);
        setup << "}" << std::endl;
        library_snippet_factory.add_setup(setup.str());

        codegen::PrettyPrinter teardown;
        teardown << "{" << std::endl;
        teardown.setIndent(teardown.indent() + 4);
        teardown << "cufftResult " << plan_var << "_dt = cufftDestroy(" << plan_var << ");" << std::endl;
        cufft_error_checking(teardown, language_extension, plan_var + "_dt");
        teardown.setIndent(teardown.indent() - 4);
        teardown << "}" << std::endl;
        library_snippet_factory.add_teardown(teardown.str());
    }

    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);

    std::string in_ptr = "__X";
    std::string out_ptr = "__Y";

    if (with_transfers) {
        stream << "cudaError_t err_cuda;" << std::endl;
        stream << x_dev_t << " *__fft_dX;" << std::endl;
        stream << y_dev_t << " *__fft_dY;" << std::endl;
        stream << "err_cuda = cudaMalloc((void**) &__fft_dX, (" << x_count << ") * sizeof(" << x_dev_t << "));"
               << std::endl;
        cuda_error_checking(stream, language_extension, "err_cuda");
        stream << "err_cuda = cudaMalloc((void**) &__fft_dY, (" << y_count << ") * sizeof(" << y_dev_t << "));"
               << std::endl;
        cuda_error_checking(stream, language_extension, "err_cuda");
        stream << "err_cuda = cudaMemcpy(__fft_dX, __X, (" << x_count << ") * sizeof(" << x_dev_t
               << "), cudaMemcpyHostToDevice);" << std::endl;
        cuda_error_checking(stream, language_extension, "err_cuda");
        in_ptr = "__fft_dX";
        out_ptr = "__fft_dY";
    }

    const std::string in_cast = forward ? real_cast : cplx_cast;
    const std::string out_cast = forward ? cplx_cast : real_cast;
    stream << "cufftResult __fft_status = " << exec_fn << "(" << plan_var << ", " << in_cast << in_ptr << ", "
           << out_cast << out_ptr << ");" << std::endl;
    cufft_error_checking(stream, language_extension, "__fft_status");

    if (with_transfers) {
        stream << "err_cuda = cudaMemcpy(__Y, __fft_dY, (" << y_count << ") * sizeof(" << y_dev_t
               << "), cudaMemcpyDeviceToHost);" << std::endl;
        cuda_error_checking(stream, language_extension, "err_cuda");
        stream << "err_cuda = cudaFree(__fft_dX);" << std::endl;
        cuda_error_checking(stream, language_extension, "err_cuda");
        stream << "err_cuda = cudaFree(__fft_dY);" << std::endl;
        cuda_error_checking(stream, language_extension, "err_cuda");
    }

    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

FFTNodeDispatcher_CUFFTWithTransfers::FFTNodeDispatcher_CUFFTWithTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FFTNodeBase& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FFTNodeDispatcher_CUFFTWithTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const FFTNodeBase&>(this->node_);
    generate_fft(stream, this->language_extension_, library_snippet_factory, node, /*with_transfers=*/true);
}

FFTNodeDispatcher_CUFFTWithoutTransfers::FFTNodeDispatcher_CUFFTWithoutTransfers(
    codegen::LanguageExtension& language_extension,
    const Function& function,
    const data_flow::DataFlowGraph& data_flow_graph,
    const FFTNodeBase& node
)
    : codegen::LibraryNodeDispatcher(language_extension, function, data_flow_graph, node) {}

void FFTNodeDispatcher_CUFFTWithoutTransfers::dispatch_code(
    codegen::PrettyPrinter& stream,
    codegen::PrettyPrinter& globals_stream,
    codegen::CodeSnippetFactory& library_snippet_factory
) {
    auto& node = static_cast<const FFTNodeBase&>(this->node_);
    generate_fft(stream, this->language_extension_, library_snippet_factory, node, /*with_transfers=*/false);
}

} // namespace sdfg::cuda::tensor
