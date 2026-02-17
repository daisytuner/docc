#include "sdfg/targets/hip/blas/utils.h"

#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/targets/hip/hip.h"
namespace sdfg {
namespace hip {
namespace blas {

void create_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    stream << "hipblasHandle_t handle;" << std::endl;
    stream << "hipblasStatus_t status_create = hipblasCreate(&handle);" << std::endl;
    hipblas_error_checking(stream, language_extension, "status_create");
}

void destroy_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    stream << "hipblasStatus_t status_destroy = hipblasDestroy(handle);" << std::endl;
    hipblas_error_checking(stream, language_extension, "status_destroy");
}

void hipblas_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_hip_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != HIPBLAS_STATUS_SUCCESS) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix() << "fprintf(stderr, \"HIPBLAS error: %d File: %s, Line: %d\\n\", "
           << status_variable << ", __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace hip
} // namespace sdfg
