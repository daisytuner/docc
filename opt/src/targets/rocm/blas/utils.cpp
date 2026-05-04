#include "sdfg/targets/rocm/blas/utils.h"

#include "sdfg/codegen/code_snippet_factory.h"
#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
#include "sdfg/targets/rocm/rocm.h"
namespace sdfg {
namespace rocm {
namespace blas {

void create_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    stream << "hipblasHandle_t handle;" << std::endl;
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "hipblasStatus_t status_create = hipblasCreate(&handle);" << std::endl;
    rocmblas_error_checking(stream, language_extension, "status_create");
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void destroy_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    stream << "{" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << "hipblasStatus_t status_destroy = hipblasDestroy(handle);" << std::endl;
    rocmblas_error_checking(stream, language_extension, "status_destroy");
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

void setup_blas_handle(codegen::CodeSnippetFactory& factory, const codegen::LanguageExtension& language_extension) {
    {
        codegen::PrettyPrinter setup_stream;
        create_blas_handle(setup_stream, language_extension);
        factory.add_setup(setup_stream.str());
    }
    {
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
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix() << "fprintf(stderr, \"ROCMBLAS error: %d File: %s, Line: %d\\n\", "
           << status_variable << ", __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

} // namespace blas
} // namespace rocm
} // namespace sdfg
