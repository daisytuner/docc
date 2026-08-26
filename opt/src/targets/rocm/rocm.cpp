#include "sdfg/targets/rocm/rocm.h"

#include <cstdio>
#include <cstdlib>
#include <sdfg/codegen/dispatchers/sequence_dispatcher.h>
#include <string>

namespace sdfg {
namespace rocm {

void rocm_error_checking(
    codegen::PrettyPrinter& stream,
    const codegen::LanguageExtension& language_extension,
    const std::string& status_variable
) {
    if (!do_rocm_error_checking()) {
        return;
    }
    stream << "if (" << status_variable << " != hipSuccess) {" << std::endl;
    stream.setIndent(stream.indent() + 4);
    stream << language_extension.external_prefix()
           << "fprintf(stderr, \"ROCM error: %s File: %s, Line: %d\\n\", hipGetErrorString(" << status_variable
           << "), __FILE__, __LINE__);" << std::endl;
    stream << language_extension.external_prefix() << "exit(EXIT_FAILURE);" << std::endl;
    stream.setIndent(stream.indent() - 4);
    stream << "}" << std::endl;
}

bool do_rocm_error_checking() {
    auto env = getenv("DOCC_ROCM_DEBUG");
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

void check_rocm_kernel_launch_errors(codegen::PrettyPrinter& stream, const codegen::LanguageExtension& language_extension) {
    if (!do_rocm_error_checking()) {
        return;
    }
    stream << "hipError_t launch_err = hipDeviceSynchronize();" << std::endl;
    rocm_error_checking(stream, language_extension, "launch_err");
    stream << "launch_err = hipGetLastError();" << std::endl;
    rocm_error_checking(stream, language_extension, "launch_err");
}

int query_rocm_wavefront_size() {
    if (const char* env = std::getenv("DOCC_ROCM_WAVEFRONT_SIZE")) {
        int value = std::atoi(env);
        if (value > 0) {
            return value;
        }
    }

    // Read the first GPU agent's wavefront size from rocminfo.
    if (FILE* pipe = popen("rocminfo 2>/dev/null", "r")) {
        char line[512];
        int size = 0;
        while (std::fgets(line, sizeof(line), pipe) != nullptr) {
            if (std::sscanf(line, " Wavefront Size: %d", &size) == 1 && size > 0) {
                break;
            }
            size = 0;
        }
        pclose(pipe);
        if (size > 0) {
            return size;
        }
    }

    return ROCM_WARP_SIZE;
}

int rocm_wavefront_size() {
    static const int cached = query_rocm_wavefront_size();
    return cached;
}

} // namespace rocm
} // namespace sdfg
