#include "sdfg/targets/rocm/blas/utils.h"

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace rocm {
namespace blas {

void create_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension, bool async) {
    stream << "hipblasHandle_t handle;" << std::endl;
    stream << "{" << std::endl;
    stream.changeIndent(+4);
    if (async) {
        stream << "if (__daisy_hipblas_handle == nullptr) {" << std::endl;
        stream.changeIndent(+4);
        stream << "hipblasStatus_t status_create = hipblasCreate(&__daisy_hipblas_handle);" << std::endl;
        rocmblas_error_checking(stream, language_extension, "status_create");
        stream.changeIndent(-4);
        stream << "}" << std::endl
               << "handle = __daisy_hipblas_handle;" << std::endl
               << "hipblasStatus_t status_set_stream = hipblasSetStream(handle, __daisy_stream);" << std::endl;
        rocmblas_error_checking(stream, language_extension, "status_set_stream");
    } else {
        stream << "hipblasStatus_t status_create = hipblasCreate(&handle);" << std::endl;
        rocmblas_error_checking(stream, language_extension, "status_create");
    }
    stream.changeIndent(-4);
    stream << "}" << std::endl;
}

void destroy_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    stream << "{" << std::endl;
    stream.changeIndent(+4);
    stream << "hipblasStatus_t status_destroy = hipblasDestroy(handle);" << std::endl;
    rocmblas_error_checking(stream, language_extension, "status_destroy");
    stream.changeIndent(-4);
    stream << "}" << std::endl;
}

void setup_blas_handle(
    codegen::CodeSnippetFactory& factory, const codegen::LanguageExtension& language_extension, bool async
) {
    codegen::PrettyPrinter setup_stream;
    create_blas_handle(setup_stream, language_extension, async);
    factory.add_setup(setup_stream.str());

    if (async) {
        factory.add_global("#include <hipblas/hipblas.h>\nstatic hipblasHandle_t __daisy_hipblas_handle = nullptr;");
    } else {
        codegen::PrettyPrinter teardown_stream;
        destroy_blas_handle(teardown_stream, language_extension);
        factory.add_teardown(teardown_stream.str());
    }
}

void rocmblas_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_rocm_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != HIPBLAS_STATUS_SUCCESS) {" << std::endl;
    stream.changeIndent(+4);
    stream << language_extension.external_prefix() << "fprintf(stderr, \"ROCMBLAS error: %d File: %s, Line: %d\\n\", "
           << status_variable << ", __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.changeIndent(-4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace rocm
} // namespace sdfg
