#include "sdfg/passes/offloading/gpu_nested_offload_pass.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/transformations/offloading/gpu_offload_nested_loop.h"

namespace sdfg {
namespace passes {

namespace {

constexpr int64_t DEFAULT_Y_GRID = 1024;
constexpr int64_t DEFAULT_X_BLOCK = 128;
constexpr int64_t DEFAULT_Y_BLOCK = 4;
constexpr int64_t WARP_SIZE = 32;

// Chooses the parallel size for a target level from a loop's iteration count.
// Grid dimensions use the exact integer count when known; block dimensions cap it
// at the default. Symbolic counts fall back to the default.
symbolic::Integer parallel_size_for(structured_control_flow::StructuredLoop* loop, int64_t default_size, bool is_block) {
    auto num_iters = loop->num_iterations();
    if (!num_iters.is_null() && SymEngine::is_a<SymEngine::Integer>(*num_iters)) {
        int64_t n = SymEngine::rcp_static_cast<const SymEngine::Integer>(num_iters)->as_int();
        if (n > 0) {
            return symbolic::integer(is_block ? std::min(n, default_size) : n);
        }
    }
    return symbolic::integer(default_size);
}

// Applies GPUOffloadNestedLoop to a nested loop for the given target, if possible.
bool apply_offload(
    builder::StructuredSDFGBuilder& builder,
    analysis::AnalysisManager& analysis_manager,
    GPUTarget target,
    structured_control_flow::StructuredLoop& loop,
    gpu::TargetLevel target_level,
    symbolic::Integer parallel_size
) {
    bool applied = false;
    if (target == GPUTarget::CUDA) {
        transformations::GPUOffloadNestedLoop<cuda::ScheduleType_CUDA_Offload>
            transform(loop, target_level, parallel_size);
        if (transform.can_be_applied(builder, analysis_manager)) {
            transform.apply(builder, analysis_manager);
            applied = true;
        }
    } else {
        transformations::GPUOffloadNestedLoop<rocm::ScheduleType_ROCM_Offload>
            transform(loop, target_level, parallel_size);
        if (transform.can_be_applied(builder, analysis_manager)) {
            transform.apply(builder, analysis_manager);
            applied = true;
        }
    }
    return applied;
}

// Returns the single nested loop directly below `node`, or nullptr if there is not
// exactly one (i.e. the nest is not a perfect chain at this level).
structured_control_flow::StructuredLoop*
single_child_loop(analysis::LoopAnalysis& loop_analysis, structured_control_flow::ControlFlowNode* node) {
    structured_control_flow::StructuredLoop* found = nullptr;
    for (auto* child : loop_analysis.children(node)) {
        if (auto* child_loop = dynamic_cast<structured_control_flow::StructuredLoop*>(child)) {
            if (found != nullptr) {
                return nullptr;
            }
            found = child_loop;
        }
    }
    return found;
}

// A single offload to apply during the sweep phase.
struct PlannedOffload {
    structured_control_flow::StructuredLoop* loop;
    gpu::TargetLevel target_level;
    symbolic::Integer parallel_size;
};

} // namespace

GPUNestedOffloadPass::
    GPUNestedOffloadPass(const std::vector<structured_control_flow::StructuredLoop*>& loops, GPUTarget target)
    : loops_(loops), target_(target) {}

bool GPUNestedOffloadPass::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    if (loops_.empty()) {
        return false;
    }

    // Mark phase: plan all offloads against a single LoopAnalysis, without mutating
    // the SDFG. This avoids re-running the analysis for every loop nest.
    std::vector<PlannedOffload> plan;
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    for (auto* outer : loops_) {
        size_t depth = gpu::perfectly_nested_depth(outer);
        if (depth < 2) {
            continue;
        }

        // Collect the perfectly-nested chain of loops below the outer loop.
        std::vector<structured_control_flow::StructuredLoop*> chain;
        structured_control_flow::ControlFlowNode* current = outer;
        while (auto* child = single_child_loop(loop_analysis, current)) {
            chain.push_back(child);
            current = child;
        }

        if (depth == 2 && chain.size() >= 1) {
            plan.push_back({chain[0], gpu::TargetLevel::X_BLOCK, parallel_size_for(chain[0], DEFAULT_X_BLOCK, true)});
        } else if (depth == 3 && chain.size() >= 2) {
            plan.push_back({chain[0], gpu::TargetLevel::X_BLOCK, parallel_size_for(chain[0], DEFAULT_X_BLOCK, true)});
            plan.push_back({chain[1], gpu::TargetLevel::WARP, symbolic::integer(WARP_SIZE)});
        } else if (depth >= 4 && chain.size() >= 3) {
            plan.push_back({chain[0], gpu::TargetLevel::Y_GRID, parallel_size_for(chain[0], DEFAULT_Y_GRID, false)});
            plan.push_back({chain[1], gpu::TargetLevel::X_BLOCK, parallel_size_for(chain[1], DEFAULT_X_BLOCK, true)});
            plan.push_back({chain[2], gpu::TargetLevel::Y_BLOCK, parallel_size_for(chain[2], DEFAULT_Y_BLOCK, true)});
        }
    }

    // Sweep phase: apply the planned offloads.
    bool applied = false;
    for (const auto& p : plan) {
        applied |= apply_offload(builder, analysis_manager, target_, *p.loop, p.target_level, p.parallel_size);
    }

    if (applied) {
        analysis_manager.invalidate_all();
    }

    return applied;
}

} // namespace passes
} // namespace sdfg
