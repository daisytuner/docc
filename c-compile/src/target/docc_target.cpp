#include "docc/target/docc_target.h"
#include <filesystem>

#include "docc/compile/file_compiler.h"

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
        builder.redirect_snippet("cu", b.build());
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

        builder.redirect_snippet("rocm.cpp", b.build());

        return true;
    }
};

bool add_highway_build_support(compile::SrcFileCompilerBuilder& builder) {
    compile::SrcFileCompilerBuilder highway_builder;
    highway_builder.inherit(builder, true);
    highway_builder.contribute_parent_link_options({"-lhwy"});

    builder.redirect_snippet("highway.cpp", highway_builder.build());
    return true;
}

/**
 * This exists as a workaround for current dependency hell. The builder is only defined in c-compile.
 * But the plugins require at least the builder interface to operate on it.
 * The c-compile library currently depends on the sdfgopt that contains these to plugins. We would need to place the
 * compiler builder code into sdfglib to make this work
 */
void register_builtin_targets(sdfg::plugins::Context& context) {
    context.add_target(&cuda_target);
    context.add_target(&rocm_target);
}

} // namespace docc::target
