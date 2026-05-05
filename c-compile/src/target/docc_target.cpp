#include "docc/target/docc_target.h"
#include <filesystem>

#include "docc/compile/src_file_compiler_builder.h"

namespace docc::target {

static DoccTarget cuda_target = {
    .short_name = "cuda",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
        builder.add_compile_option("-x cuda");
        builder.add_link_option("-lcuda");
        builder.add_link_option("/usr/local/cuda/lib64/libcudart.so");
        builder.add_link_option("/usr/local/cuda/lib64/libcublas.so");

        compile::SrcFileCompilerBuilder b;
        b.inherit(builder, true);
        b.add_compile_option("--cuda-gpu-arch=sm_70");
        b.add_compile_option("--cuda-path=/usr/local/cuda");
        b.set_bin_extension("cu");
        builder.redirect_snippet("cu", std::move(b));
        return true;
    }
};

static DoccTarget rocm_target = {
    .short_name = "rocm",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
        builder.add_compile_option("-x hip");
        std::string rocm_dev = "gfx1201";
        builder.add_compile_option("--offload-arch=" + rocm_dev);
        std::filesystem::path rocm_path = "/opt/rocm";
        builder.add_compile_option("--offload-host-only");
        builder.add_compile_option("--rocm-path=" + rocm_path.string());
        builder.add_include_path(rocm_path / "include");

        auto lib_path = rocm_path / "lib";
        builder.add_link_option(lib_path / "libamdhip64.so");
        builder.add_link_option(lib_path / "libhiprtc.so");
        builder.add_link_option(lib_path / "libhipblas.so");

        compile::SrcFileCompilerBuilder b;
        b.inherit(builder, true);
        b.remove_compile_option("--offload-host-only");

        builder.redirect_snippet("rocm.cpp", std::move(b));

        return true;
    }
};

/**
 * Temporary workaround. Ideally, these should live with the plugins themselves.
 * But the plugins currently live in sdfgopt. We either need to split out the builder API
 * so that sdfgopt can depend on it, or the plugins must be moved to a new library
 */
void register_builtin_targets(sdfg::plugins::Context& context) {
    context.add_target(&cuda_target);
    context.add_target(&rocm_target);
}

} // namespace docc::target
