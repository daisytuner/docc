#include "sdfg/scheduling/a_priori_loop_nest_normalization.h"

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/pipeline.h"
#include "sdfg/passes/symbolic/symbol_propagation.h"
#include "sdfg/reordering/reordering.h"

namespace sdfg {
namespace scheduling {

APrioriLoopNestNormalization::APrioriLoopNestNormalization(bool enable_fusion) : enable_fusion_(enable_fusion) {
}

bool APrioriLoopNestNormalization::
    schedule(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    reordering::normalize(builder.subject(), enable_fusion_);
    // normalize runs on its own builder and analysis manager.
    analysis_manager.invalidate_all();

    passes::SymbolPropagation symbol_propagation;
    symbol_propagation.run(builder, analysis_manager);
    passes::DeadDataElimination dead_data_elimination;
    dead_data_elimination.run(builder, analysis_manager);
    auto dead_code_elimination = passes::Pipeline::dead_code_elimination();
    dead_code_elimination.run(builder, analysis_manager);
    return true;
}

} // namespace scheduling
} // namespace sdfg
