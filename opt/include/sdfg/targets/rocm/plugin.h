#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/data_flow/library_nodes/stdlib/memset.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/passes/scheduler/rocm_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/rocm/blas/dot.h"
#include "sdfg/targets/rocm/blas/gemm.h"
#include "sdfg/targets/rocm/blas/gemm_handtuned.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/targets/rocm/rocm_data_offloading_node.h"
#include "sdfg/targets/rocm/rocm_map_dispatcher.h"
#include "sdfg/targets/rocm/stdlib/memset.h"

namespace sdfg {
namespace rocm {

void register_rocm_plugin(plugins::Context& context);

/**
 * @deprecated use the variant with explicit context
 */
inline void register_rocm_plugin() {
    auto ctx = sdfg::plugins::Context::global_context();
    register_rocm_plugin(ctx);
}

} // namespace rocm
} // namespace sdfg
