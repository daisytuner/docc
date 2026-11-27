#include "docc/target/docc_target.h"

#include <filesystem>
#include <memory>

#include "docc/compile/src_file_compiler_builder.h"
#include "docc/util/cuda_query_compute_capability.h"
#include "sdfg/data_flow/library_nodes/math/tensor/matmul_node.h"
#include "sdfg/passes/expansion/library_node_expansion_pass.h"
#include "sdfg/passes/expansion/switch_expander.h"
#include "sdfg/passes/offloading/cuda_library_node_expansion_pass.h"
#include "sdfg/passes/offloading/cuda_library_node_rewriter_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_expansion_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_rewriter_pass.h"
#include "sdfg/passes/scheduler/cuda_offload_scheduler.h"
#include "sdfg/passes/scheduler/omp_scheduler.h"
#include "sdfg/passes/scheduler/rocm_offload_scheduler.h"
#include "sdfg/scheduling/a_priori_loop_nest_normalization.h"
#include "sdfg/scheduling/passthrough_scheduler.h"
#include "sdfg/targets/cuda/cuda_offload_dispatcher_strategy.h"
#include "sdfg/targets/omp/math/tensor/conv_expander.h"
#include "sdfg/targets/rocm/rocm_offload_dispatcher_strategy.h"

namespace docc::target {

static DoccTarget cuda_target = {
    .api_ver = DoccTarget::NEWEST_API_VER,
    .short_name = "cuda",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
        builder.add_compile_option("-x cuda");
        builder.add_link_option("-lcuda");
        builder.add_link_option("/usr/local/cuda/lib64/libcudart.so");
        builder.add_link_option("/usr/local/cuda/lib64/libcublas.so");

        compile::SrcFileCompilerBuilder b;
        b.inherit(builder, true);

        const char* arch_env = std::getenv("DOCC_CUDA_ARCH");
        if (arch_env) { // build for a specific arch
            builder.add_compile_option("--cuda-gpu-arch=" + std::string(arch_env));
            b.add_compile_option("--cuda-gpu-arch=" + std::string(arch_env));
        } else {
            auto caps = sdfg::gpu::cuda::query_cuda_compute_capabilities();
            if (!caps.empty()) {
                auto& first = caps.front();
                std::cerr << "[DOCC] Compiling CUDA for sm_" << first.compute_cap << " of "
                          << ((first.device_names.empty()) ? "unidentified GPU" : first.device_names.front())
                          << std::endl;
                util::clang_21_set_cuda_specific_compute_cap(builder, b, caps.front().compute_cap);
            } else {
                // don't know the arch, so try to build for all with the driver handling the rest (may not work for all
                // cases)
                std::cerr << "[DOCC] No CUDA ARCH identified, trying to compile for all supported architectures"
                          << std::endl;
                util::clang_21_set_cuda_forward_compatible_options(builder, b);
            }
        }

        b.add_compile_option("--cuda-path=/usr/local/cuda");
        b.set_bin_extension("cu");
        b.set_compile_order(1, 1);
        builder.redirect_snippet(sdfg::cuda::CUDAOffloadDispatcherStrategy::KERNEL_SNIPPET_FILE_EXT, std::move(b));

        compile::SrcFileCompilerBuilder hb;
        hb.inherit(builder, false);
        hb.codegen_only();
        hb.set_compile_order(0);

        builder.redirect_snippet(sdfg::cuda::CUDAOffloadDispatcherStrategy::KERNEL_SNIPPET_HEADER_EXT, std::move(hb));
        return true;
    },
    .apply_expand_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                    sdfg::analysis::AnalysisManager& analysis_manager,
                                    const TargetOptions& options) -> bool {
        sdfg::passes::CudaExpansionPass cuda_expansion_pass(*options.options);
        bool changed = cuda_expansion_pass.run(builder, analysis_manager);

        sdfg::passes::LibraryNodeExpansionPass
            exp_pass(std::make_shared<sdfg::math::tensor::MatMulBlasExpander>(true), *options.options);
        changed |= exp_pass.run_pass(builder, analysis_manager);

        return changed;
    },
    .apply_sched_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                   sdfg::analysis::AnalysisManager& analysis_manager,
                                   const TargetOptions& options) -> bool {
        sdfg::cuda::CudaLibraryNodeRewriterPass cuda_pass;
        return cuda_pass.run(builder, analysis_manager);
    },
    .scheduling =
        {.create_sdfg_scheduler =
             [](const TargetOptions& options) -> std::unique_ptr<sdfg::passes::scheduler::SDFGScheduler> {
             return std::make_unique<sdfg::scheduling::PassthroughScheduler>();
         },
         .create_loop_schedulers =
             [](const TargetOptions& options) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
             return {std::make_shared<sdfg::passes::scheduler::CUDAOffloadScheduler>()};
         },
         .remote_normalization = true}
};

static DoccTarget rocm_target = {
    .api_ver = DoccTarget::NEWEST_API_VER,
    .short_name = "rocm",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
        builder.add_compile_option("-x hip");
        // Enable the synchronizing warp builtins (e.g. __shfl_xor_sync).
        builder.add_compile_option("-DHIP_ENABLE_WARP_SYNC_BUILTINS");

        std::string arch_str;
        const char* arch_name = std::getenv("DOCC_ROCM_ARCH");
        if (arch_name) {
            arch_str = arch_name;
        } else {
            auto devs = sdfg::gpu::rocm::query_rocm_devices();
            if (!devs.empty()) {
                auto& first = devs.front();
                std::cerr << "[DOCC] Compiling ROCM for " << first.gfx_name << " of "
                          << ((first.device_names.empty()) ? "unidentified GPU" : first.device_names.front())
                          << std::endl;
                arch_str = first.gfx_name;
            } else {
                throw std::runtime_error(
                    "Could not determine ROCm architecture. Please set the DOCC_ROCM_ARCH environment variable."
                );
            }
        }

        builder.add_compile_option("--offload-arch=" + arch_str);
        const char* rocm_env = std::getenv("ROCM_PATH");
        std::filesystem::path rocm_path = (rocm_env && *rocm_env) ? rocm_env : "/opt/rocm";
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
        b.set_compile_order(1, 1);

        builder.redirect_snippet(sdfg::rocm::ROCMOffloadDispatcherStrategy::KERNEL_SNIPPET_FILE_EXT, std::move(b));

        compile::SrcFileCompilerBuilder hb;
        hb.inherit(builder, false);
        hb.codegen_only();
        hb.set_compile_order(0);

        builder.redirect_snippet(sdfg::rocm::ROCMOffloadDispatcherStrategy::KERNEL_SNIPPET_HEADER_EXT, std::move(hb));

        return true;
    },
    .apply_expand_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                    sdfg::analysis::AnalysisManager& analysis_manager,
                                    const TargetOptions& options) -> bool {
        sdfg::passes::RocmExpansionPass rocm_expansion_pass(*options.options);
        bool changed = rocm_expansion_pass.run(builder, analysis_manager);

        sdfg::passes::LibraryNodeExpansionPass
            exp_pass(std::make_shared<sdfg::math::tensor::MatMulBlasExpander>(true, true), *options.options);
        changed |= exp_pass.run_pass(builder, analysis_manager);

        return changed;
    },
    .apply_sched_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                   sdfg::analysis::AnalysisManager& analysis_manager,
                                   const TargetOptions& options) -> bool {
        sdfg::rocm::RocmLibraryNodeRewriterPass rocm_pass;
        return rocm_pass.run(builder, analysis_manager);
    },
    .scheduling =
        {.create_sdfg_scheduler =
             [](const TargetOptions& options) -> std::unique_ptr<sdfg::passes::scheduler::SDFGScheduler> {
             return std::make_unique<sdfg::scheduling::PassthroughScheduler>();
         },
         .create_loop_schedulers =
             [](const TargetOptions& options) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
             return {std::make_shared<sdfg::passes::scheduler::ROCMOffloadScheduler>()};
         },
         .remote_normalization = true}
};

static DoccTarget sequential_target = {
    .short_name = "sequential",
    .apply_expand_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                    sdfg::analysis::AnalysisManager& analysis_manager,
                                    const TargetOptions& options) -> bool {
        auto expander = std::make_shared<sdfg::passes::SwitchExpander>(
            std::make_shared<sdfg::omp::OMPConvExpander>(), std::make_shared<sdfg::math::tensor::MatMulBlasExpander>()
        );
        sdfg::passes::LibraryNodeExpansionPass libnode_expansion_pass(expander, *options.options);
        return libnode_expansion_pass.run(builder, analysis_manager);
    },
    .scheduling = {
        .create_sdfg_scheduler = [](const TargetOptions&) -> std::unique_ptr<sdfg::passes::scheduler::SDFGScheduler> {
            return std::make_unique<sdfg::scheduling::APrioriLoopNestNormalization>();
        },
        .create_loop_schedulers =
            [](const TargetOptions& options) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
            return {};
        }
    }
};

static DoccTarget openmp_target = {
    .short_name = "openmp",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
#if defined(__APPLE__)
        builder.add_common_option("-Xpreprocessor -fopenmp");
        builder.add_library_path("/opt/homebrew/opt/libomp/lib");
        builder.add_library_path("/opt/homebrew/opt/libomp/include");
        builder.add_link_option("-lomp");
#else
        builder.add_common_option("-fopenmp");
#endif
        return true;
    },
    .apply_expand_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                    sdfg::analysis::AnalysisManager& analysis_manager,
                                    const TargetOptions& options) -> bool {
        auto expander = std::make_shared<sdfg::passes::SwitchExpander>(
            std::make_shared<sdfg::omp::OMPConvExpander>(), std::make_shared<sdfg::math::tensor::MatMulBlasExpander>()
        );
        sdfg::passes::LibraryNodeExpansionPass libnode_expansion_pass(expander, *options.options);
        return libnode_expansion_pass.run(builder, analysis_manager);
    },
    .scheduling =
        {.create_sdfg_scheduler = [](const TargetOptions&) -> std::unique_ptr<sdfg::passes::scheduler::SDFGScheduler> {
             return std::make_unique<sdfg::scheduling::APrioriLoopNestNormalization>();
         },
         .create_loop_schedulers =
             [](const TargetOptions& options) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
             return {std::make_shared<sdfg::passes::scheduler::OMPScheduler>()};
         }}
};

/**
 * Temporary workaround. Ideally, these should live with the plugins themselves.
 * But the plugins currently live in sdfgopt. We either need to split out the builder API
 * so that sdfgopt can depend on it, or the plugins must be moved to a new library
 */
void register_builtin_targets(sdfg::plugins::Context& context) {
    context.add_target(&cuda_target);
    context.add_target(&rocm_target);
    context.add_target(&sequential_target);
    context.add_target(&openmp_target);
}

} // namespace docc::target
