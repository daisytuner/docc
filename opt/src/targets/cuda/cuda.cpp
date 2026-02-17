#include "sdfg/targets/cuda/cuda.h"

#include <cstdlib>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <string>

namespace sdfg {
namespace cuda {

void cuda_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_cuda_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != cudaSuccess) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix()
           << "fprintf(stderr, \"CUDA error: %s File: %s, Line: %d\\n\", cudaGetErrorString(" << status_variable
           << "), __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

bool do_cuda_error_checking() {
    auto env = getenv("DOCC_CUDA_DEBUG");
    if (env == nullptr) {
        return false;
    }
    std::string env_str(env);
    std::transform(env_str.begin(), env_str.end(), env_str.begin(), ::tolower);
    if (env_str == "1" || env_str == "true") {
        return true;
    }
    return false;
}

void check_cuda_kernel_launch_errors(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    if (!do_cuda_error_checking()) {
        return;
    }
    stream << "cudaError_t launch_err = cudaDeviceSynchronize();" << std::endl;
    cuda_error_checking(stream, language_extension, "launch_err");
    stream << "launch_err = cudaGetLastError();" << std::endl;
    cuda_error_checking(stream, language_extension, "launch_err");
}

} // namespace cuda
} // namespace sdfg
