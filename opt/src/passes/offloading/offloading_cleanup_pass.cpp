#include "sdfg/passes/offloading/offloading_cleanup_pass.h"

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/dataflow/dead_reference_elimination.h"
#include "sdfg/passes/dataflow/reference_propagation.h"
#include "sdfg/passes/offloading/data_transfer_minimization_pass.h"
#include "sdfg/passes/offloading/device_buffer_reuse_pass.h"
#include "sdfg/passes/redundant_zero_init_elimination_pass.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

namespace sdfg {
namespace passes {

bool OffloadingCleanupPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    bool changed = false;

    DataTransferMinimizationPass data_transfer_minimization_pass;
    changed |= data_transfer_minimization_pass.run(builder, analysis_manager);

    DeviceBufferReusePass device_buffer_reuse_pass;
    changed |= device_buffer_reuse_pass.run(builder, analysis_manager);
    DeadDataElimination dde(false);
    changed |= dde.run(builder, analysis_manager);
    DeadCFGElimination dead_cfg_elimination;
    changed |= dead_cfg_elimination.run(builder, analysis_manager);

    ReferencePropagation reference_propagation;
    changed |= reference_propagation.run(builder, analysis_manager);
    DeadReferenceElimination dead_reference_elimination;
    changed |= dead_reference_elimination.run(builder, analysis_manager);
    changed |= reference_propagation.run(builder, analysis_manager);
    changed |= dead_reference_elimination.run(builder, analysis_manager);

    // Drop redundant zero-init nests of matmuls lowered to MMA. Run after reference propagation and transfer cleanup
    // so the init map is the immediate sibling of the MMA nest and both name the same (aliases collapsed) output
    // container, which is what the transformation matches. Nests whose output still round-trips through the host here
    // only become eligible after device-residency promotion, so promote_device_residency re-runs the pass.
    RedundantZeroInitEliminationPass redundant_zero_init_elimination_pass;
    changed |= redundant_zero_init_elimination_pass.run(builder, analysis_manager);
    changed |= dde.run(builder, analysis_manager);
    changed |= dead_cfg_elimination.run(builder, analysis_manager);

    return changed;
}

} // namespace passes
} // namespace sdfg
