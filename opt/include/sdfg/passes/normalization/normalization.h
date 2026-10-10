#pragma once

#include <sdfg/passes/pipeline.h>

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/structured_control_flow/block_fusion.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"
#include "sdfg/reordering/fusion/passes/map_fusion_pass.h"
#include "sdfg/reordering/passes/perfect_loop_distribution.h"
#include "sdfg/reordering/passes/stride_minimization.h"

namespace sdfg {
namespace passes {
namespace normalization {

inline passes::Pipeline loop_normalization() {
    passes::Pipeline pipeline("Loop Normalization");

    // Register passes for loop normalization
    pipeline.register_pass<reordering::PerfectLoopDistributionPass>();
    pipeline.register_pass<reordering::StrideMinimization>();

    return pipeline;
}

inline passes::Pipeline stride_minimization() {
    passes::Pipeline pipeline("Stride Minimization");

    pipeline.register_pass<reordering::StrideMinimization>();

    return pipeline;
}

inline passes::Pipeline map_fusion(bool allow_init_hoist = true, bool allow_prod_into_cons = true) {
    passes::Pipeline p("MapFusion");

    p.register_pass<reordering::fusion::MapFusionPass>(allow_init_hoist, allow_prod_into_cons);
    p.register_pass<passes::BlockFusionPass>();
    p.register_pass<passes::DeadDataElimination>(true);
    p.register_pass<passes::DeadCFGElimination>(true);

    return p;
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
