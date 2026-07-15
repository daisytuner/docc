#include "docc/target/docc_target.h"
#include <filesystem>

#include "docc/compile/src_file_compiler_builder.h"
#include "sdfg/passes/offloading/cuda_library_node_expansion_pass.h"
#include "sdfg/passes/offloading/cuda_library_node_rewriter_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_expansion_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_rewriter_pass.h"
#include "sdfg/passes/scheduler/cuda_scheduler.h"
#include "sdfg/passes/scheduler/omp_scheduler.h"
#include "sdfg/passes/scheduler/rocm_scheduler.h"
#include "sdfg/passes/scheduler/vectorize_scheduler.h"

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
        b.add_compile_option("--cuda-gpu-arch=sm_70");
        b.add_compile_option("--cuda-path=/usr/local/cuda");
        b.set_bin_extension("cu");
        builder.redirect_snippet("cu", std::move(b));
        return true;
    },
    .apply_expand_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                    sdfg::analysis::AnalysisManager& analysis_manager,
                                    const TargetOptions& options) -> bool {
        sdfg::passes::CudaExpansionPass cuda_expansion_pass;
        return cuda_expansion_pass.run(builder, analysis_manager);
    },
    .apply_sched_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                   sdfg::analysis::AnalysisManager& analysis_manager,
                                   const TargetOptions& options) -> bool {
        sdfg::cuda::CudaLibraryNodeRewriterPass cuda_pass;
        return cuda_pass.run(builder, analysis_manager);
    },
    .get_target_loop_schedulers = [](const TargetOptions& options
                                  ) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
        std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> schedulers;
        schedulers.push_back(std::make_shared<sdfg::passes::scheduler::CUDAScheduler>());
        return schedulers;
    }
};

static DoccTarget rocm_target = {
    .api_ver = DoccTarget::NEWEST_API_VER,
    .short_name = "rocm",
    .apply_additional_compile_options = [](compile::SrcFileCompilerBuilder& builder) -> bool {
        builder.add_compile_option("-x hip");
        const char* arch_env = std::getenv("DOCC_ROCM_ARCH");
        if (!arch_env) {
            arch_env = "gfx1201";
        }
        builder.add_compile_option("--offload-arch=" + std::string(arch_env));
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
    },
    .apply_expand_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                    sdfg::analysis::AnalysisManager& analysis_manager,
                                    const TargetOptions& options) -> bool {
        sdfg::passes::RocmExpansionPass rocm_expansion_pass;
        return rocm_expansion_pass.run(builder, analysis_manager);
    },
    .apply_sched_time_mapping = [](sdfg::builder::StructuredSDFGBuilder& builder,
                                   sdfg::analysis::AnalysisManager& analysis_manager,
                                   const TargetOptions& options) -> bool {
        sdfg::rocm::RocmLibraryNodeRewriterPass rocm_pass;
        return rocm_pass.run(builder, analysis_manager);
    },
    .get_target_loop_schedulers = [](const TargetOptions& options
                                  ) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
        std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> schedulers;
        schedulers.push_back(std::make_shared<sdfg::passes::scheduler::ROCMScheduler>());
        return schedulers;
    }
};

static DoccTarget sequential_target = {
    .short_name = "sequential",
    .get_target_loop_schedulers = [](const TargetOptions& options
                                  ) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
        std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> schedulers;
        schedulers.push_back(std::make_shared<sdfg::passes::scheduler::VectorizeScheduler>());
        return schedulers;
    }
};

static DoccTarget openmp_target = {
    .short_name = "openmp",
    .get_target_loop_schedulers = [](const TargetOptions& options
                                  ) -> std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> {
        std::vector<std::shared_ptr<sdfg::passes::scheduler::LoopScheduler>> schedulers;
        schedulers.push_back(std::make_shared<sdfg::passes::scheduler::OMPScheduler>());
        schedulers.push_back(std::make_shared<sdfg::passes::scheduler::VectorizeScheduler>());
        return schedulers;
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
    context.add_target(&sequential_target);
    context.add_target(&openmp_target);
}

} // namespace docc::target
