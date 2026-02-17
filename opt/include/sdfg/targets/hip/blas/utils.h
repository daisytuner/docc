#pragma once

#include "sdfg/codegen/language_extension.h"
#include "sdfg/codegen/utils.h"
namespace sdfg {
namespace hip {
namespace blas {

void create_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension);

void destroy_blas_handle(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension);

void hipblas_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
);

} // namespace blas
} // namespace hip
} // namespace sdfg
