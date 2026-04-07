#include "target_mapping.h"

#include "sdfg/passes/offloading/cuda_library_node_rewriter_pass.h"
#include "sdfg/passes/offloading/rocm_library_node_rewriter_pass.h"
#include "sdfg/passes/targets/target_mapping_pass.h"
#include "sdfg/plugins/target_mapping.h"

#ifdef DOCC_HAS_TARGET_ET
#include "docc/target/et/target.h"
#endif

namespace docc::plugins {

void apply_lib_node_target_mapping(
    sdfg::builder::StructuredSDFGBuilder& builder,
    sdfg::analysis::AnalysisManager& analysis_manager,
    TargetOptions& options
) {
    if (options.target == "cuda") {
        sdfg::cuda::CudaLibraryNodeRewriterPass cuda_pass;
        cuda_pass.run(builder, analysis_manager);
    } else if (options.target == "etsoc") {
#ifdef DOCC_HAS_TARGET_ET
        docc::target::et::et_scheduling_passes(builder, analysis_manager, options.category);
#endif
    } else if (options.target == "rocm") {
        sdfg::rocm::RocmLibraryNodeRewriterPass rocm_pass;
        rocm_pass.run(builder, analysis_manager);
    }

    // Generic code. Find a way to declare TargetMappers with each plugin and then discover those from target and use
    // the generic pass std::vector<std::shared_ptr<sdfg::plugins::TargetMapper>> mappers{};
    // sdfg::passes::TargetMappingPass mappingPass(mappers);
    // mappingPass.run_pass(builder, analysis_manager);
}

} // namespace docc::plugins
