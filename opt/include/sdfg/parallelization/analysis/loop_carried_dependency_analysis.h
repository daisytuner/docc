#pragma once

// Deprecated: moved to sdfg/reordering/analysis/loop_carried_dependency_analysis.h
#include "sdfg/reordering/analysis/loop_carried_dependency_analysis.h"

namespace sdfg {
namespace parallelization {

using reordering::LOOP_CARRIED_DEPENDENCY_READ_WRITE;
using reordering::LOOP_CARRIED_DEPENDENCY_UNDEFINED;
using reordering::LOOP_CARRIED_DEPENDENCY_WRITE_WRITE;
using reordering::LoopCarriedDependency;
using reordering::LoopCarriedDependencyAnalysis;
using reordering::LoopCarriedDependencyInfo;
using reordering::LoopCarriedDependencyPair;

} // namespace parallelization
} // namespace sdfg
