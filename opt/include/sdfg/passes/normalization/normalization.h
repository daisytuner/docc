#pragma once

#include <sdfg/passes/pipeline.h>

#include "sdfg/passes/dataflow/dead_data_elimination.h"
#include "sdfg/passes/normalization/map_fusion.h"
#include "sdfg/passes/normalization/perfect_loop_distribution.h"
#include "sdfg/passes/normalization/stride_minimization.h"
#include "sdfg/passes/structured_control_flow/block_fusion.h"
#include "sdfg/passes/structured_control_flow/dead_cfg_elimination.h"

namespace sdfg {
namespace passes {
namespace normalization {

inline passes::Pipeline loop_normalization() {
    passes::Pipeline pipeline("Loop Normalization");

    // Register passes for loop normalization
    pipeline.register_pass<normalization::PerfectLoopDistributionPass>();
    pipeline.register_pass<normalization::StrideMinimization>();

    return pipeline;
}

inline passes::Pipeline map_fusion() {
    passes::Pipeline p("MapFusion");

    p.register_pass<normalization::MapFusionPass>();
    p.register_pass<passes::BlockFusionPass>();
    p.register_pass<passes::DeadDataElimination>(true);
    p.register_pass<passes::DeadCFGElimination>(true);

    return p;
}

} // namespace normalization
} // namespace passes
} // namespace sdfg
