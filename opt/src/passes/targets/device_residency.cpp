#include <sdfg/passes/targets/target_mapping_pass.h>

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/dead_reference_elimination.h"
#include "sdfg/passes/dataflow/reference_propagation.h"
#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"
#include "sdfg/passes/offloading/device_buffer_reuse_pass.h"
#include "sdfg/passes/offloading/device_resident_arg_promotion_pass.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

namespace sdfg {
namespace passes {

bool promote_device_residency(StructuredSDFG& sdfg, bool is_rocm) {
    builder::StructuredSDFGBuilder builder(sdfg);
    analysis::AnalysisManager analysis_manager(sdfg);

    DeviceResidentArgPromotionPass promotion_pass(is_rocm);
    bool promoted = promotion_pass.run(builder, analysis_manager);
    if (promoted) {
        ReferencePropagation reference_propagation;
        DeadReferenceElimination dead_reference_elimination;
        DataTransferMinimizationPass data_transfer_minimization;
        DeadDataElimination dead_data_elimination;
        DeviceBufferReusePass device_buffer_reuse_pass;
        DeadCFGElimination dead_cfg_elimination;

        // 1st round
        reference_propagation.run(builder, analysis_manager);
        dead_reference_elimination.run(builder, analysis_manager);
        data_transfer_minimization.run(builder, analysis_manager);
        device_buffer_reuse_pass.run(builder, analysis_manager);
        dead_data_elimination.run(builder, analysis_manager);
        dead_cfg_elimination.run(builder, analysis_manager);

        // 2nd round
        reference_propagation.run(builder, analysis_manager);
        dead_reference_elimination.run(builder, analysis_manager);
        data_transfer_minimization.run(builder, analysis_manager);
        device_buffer_reuse_pass.run(builder, analysis_manager);
        dead_data_elimination.run(builder, analysis_manager);
        dead_cfg_elimination.run(builder, analysis_manager);
    }

    return promoted;
}

} // namespace passes
} // namespace sdfg
