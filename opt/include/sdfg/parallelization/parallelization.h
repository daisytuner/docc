#pragma once

#include <sdfg/passes/pipeline.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/parallelization/passes/auto_parallelization.h"
#include "sdfg/reordering/analysis/loop_carried_dependency_analysis.h"

namespace sdfg {
namespace parallelization {

inline void register_parallelization_plugin(plugins::Context& context) {
    // Register library nodes
}

/**
 * @deprecated use the variant with explicit context
 */
inline void register_parallelization_plugin() {
    auto ctx = sdfg::plugins::Context::global_context();
    register_parallelization_plugin(ctx);
}

inline passes::Pipeline data_parallelism() {
    passes::Pipeline p("DataParallelism");

    p.register_pass<AutoParallelization>();
    p.register_pass<passes::SymbolPropagation>();
    p.register_pass<passes::DeadDataElimination>();

    return p;
};

} // namespace parallelization
} // namespace sdfg
