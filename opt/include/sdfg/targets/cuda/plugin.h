#pragma once

#include <sdfg/codegen/dispatchers/node_dispatcher_registry.h>
#include <sdfg/data_flow/library_nodes/math/blas/dot_node.h>
#include <sdfg/data_flow/library_nodes/math/blas/gemm_node.h>
#include <sdfg/data_flow/library_nodes/stdlib/memset.h>
#include <sdfg/plugins/plugins.h>
#include <sdfg/serializer/json_serializer.h>

#include "sdfg/codegen/language_extension.h"
#include "sdfg/passes/scheduler/cuda_scheduler.h"
#include "sdfg/passes/scheduler/scheduler_registry.h"
#include "sdfg/targets/cuda/blas/dot.h"
#include "sdfg/targets/cuda/blas/gemm.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/cuda/cuda_data_offloading_node.h"
#include "sdfg/targets/cuda/cuda_map_dispatcher.h"
#include "sdfg/targets/cuda/stdlib/memset.h"

namespace sdfg {
namespace cuda {


void register_cuda_plugin(plugins::Context& context);

/**
 * @deprecated use the variant with explicit context
 */
inline void register_cuda_plugin() {
    auto ctx = sdfg::plugins::Context::global_context();
    register_cuda_plugin(ctx);
}

} // namespace cuda
} // namespace sdfg
