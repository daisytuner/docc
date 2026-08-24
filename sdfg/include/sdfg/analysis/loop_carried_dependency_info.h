#pragma once

#include <unordered_map>
#include <vector>

#include "sdfg/analysis/analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/data_dependency_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_sdfg.h"
#include "sdfg/symbolic/maps.h"

namespace sdfg::analysis {

enum LoopCarriedDependency {
    LOOP_CARRIED_DEPENDENCY_READ_WRITE,
    LOOP_CARRIED_DEPENDENCY_WRITE_WRITE,
    LOOP_CARRIED_DEPENDENCY_UNDEFINED,
};

/**
 * @brief Extended loop-carried dependency information including distance vectors.
 *
 * Combines the dependency type (read-write or write-write) with the full
 * ISL delta set representing all possible iteration-distance vectors.
 */
struct LoopCarriedDependencyInfo {
    LoopCarriedDependency type;
    symbolic::maps::DependenceDeltas deltas;
};

} // namespace sdfg::analysis
