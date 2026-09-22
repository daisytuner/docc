#pragma once

#include <sdfg/passes/pipeline.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/parallelization/analysis/loop_carried_dependency_analysis.h"
#include "sdfg/parallelization/passes/for_classification.h"

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

    p.register_pass<ForClassificationPass>();
    p.register_pass<passes::SymbolPropagation>();
    p.register_pass<passes::DeadDataElimination>();

    return p;
};

} // namespace parallelization
} // namespace sdfg
