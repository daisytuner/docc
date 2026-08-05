#include "sdfg/targets/gpu/gpu_map_utils.h"

#include <string>
#include <unordered_set>

#include "sdfg/analysis/arguments_analysis.h"
#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/analysis/loop_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/rocm/rocm.h"

namespace sdfg {
namespace gpu {

template<typename ScheduleT>
symbolic::Expression find_nested_gpu_blocksize(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);

    // Check for repeated dimensions in loop tree paths
    auto loop_tree_paths = loop_analysis.loop_tree_paths(&node);
    for (auto& path : loop_tree_paths) {
        bool foundX = false;
        bool foundY = false;
        bool foundZ = false;
        for (auto& loop : path) {
            if (auto map = dyn_cast<structured_control_flow::Map*>(loop)) {
                if (map->schedule_type().value() == ScheduleT::value()) {
                    auto dim = ScheduleT::dimension(map->schedule_type());
                    if (dim == GPUDimension::X) {
                        if (foundX) {
                            throw InvalidSDFGException("Nested map in GPU kernel has repeated X dimension");
                        }
                        foundX = true;
                    } else if (dim == GPUDimension::Y) {
                        if (foundY) {
                            throw InvalidSDFGException("Nested map in GPU kernel has repeated Y dimension");
                        }
                        foundY = true;
                    } else if (dim == GPUDimension::Z) {
                        if (foundZ) {
                            throw InvalidSDFGException("Nested map in GPU kernel has repeated Z dimension");
                        }
                        foundZ = true;
                    }
                }
            }
        }
    }

    // Find block size for the requested dimension
    for (auto loop : loops) {
        if (auto map = dyn_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() != ScheduleT::value() &&
                map->schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value()) {
                throw InvalidSDFGException("Nested map in GPU kernel not GPU or Sequential");
            }

            if (map->schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value()) {
                continue;
            }

            if (ScheduleT::dimension(map->schedule_type()) == dimension) {
                return ScheduleT::block_size(map->schedule_type());
            }
        }
    }
    return symbolic::one();
}

template<typename ScheduleT>
symbolic::Expression find_nested_gpu_iterations(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);

    symbolic::Expression max_num_iterations = symbolic::one();

    for (auto loop : loops) {
        if (auto map = dyn_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() != ScheduleT::value() &&
                map->schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value()) {
                throw InvalidSDFGException("Nested map in GPU kernel not GPU or Sequential");
            }
            if (map->schedule_type().value() == structured_control_flow::ScheduleType_Sequential::value()) {
                continue;
            }
            if (ScheduleT::dimension(map->schedule_type()) != dimension) {
                continue;
            }

            // Note: arbitrary `init` and `stride` are permitted here; the
            // dispatcher emits `indvar = init + thread_flat_id * stride` so
            // the body sees the natural strided value. `num_iterations()`
            // already accounts for both.
            auto num_iterations = map->num_iterations();
            if (num_iterations.is_null()) {
                throw InvalidSDFGException("Cannot determine number of iterations for nested map in GPU kernel");
            }
            max_num_iterations = symbolic::max(max_num_iterations, num_iterations);
        }
    }
    return max_num_iterations;
}

template<typename ScheduleT>
bool is_outermost_gpu_map(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto& loop_tree = loop_analysis.loop_tree();
    structured_control_flow::ControlFlowNode* ancestor = loop_tree.at(&node);
    while (ancestor != nullptr) {
        if (auto map = dyn_cast<structured_control_flow::Map*>(ancestor)) {
            if (map->schedule_type().value() == ScheduleT::value()) {
                return false;
            }
        }
        ancestor = loop_tree.at(ancestor);
    }
    return true;
}

template<typename ScheduleT>
symbolic::SymbolSet get_gpu_indvars(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);
    symbolic::SymbolSet indvars;
    for (const auto& loop : loops) {
        if (auto map = dyn_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() == ScheduleT::value()) {
                if (ScheduleT::dimension(map->schedule_type()) == dimension) {
                    indvars.insert(map->indvar());
                }
            }
        }
    }
    return indvars;
}

template<typename ScheduleT>
std::vector<structured_control_flow::Map*>
get_gpu_maps(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();
    auto loops = loop_analysis.descendants(&node);
    loops.insert(&node);
    std::vector<structured_control_flow::Map*> maps;
    for (const auto& loop : loops) {
        if (auto map = dyn_cast<structured_control_flow::Map*>(loop)) {
            if (map->schedule_type().value() == ScheduleT::value()) {
                if (ScheduleT::dimension(map->schedule_type()) == dimension) {
                    maps.push_back(map);
                }
            }
        }
    }
    return maps;
}

bool nested_parallelization_replicates_accumulation(
    structured_control_flow::Map& loop, analysis::AnalysisManager& analysis_manager
) {
    auto& loop_analysis = analysis_manager.get<analysis::LoopAnalysis>();

    // The outermost enclosing GPU map is the kernel scope. Everything below it is
    // folded into a single flattened launch, so adding a dimension for `loop`
    // replicates its siblings (and the siblings of any ancestor up to this map)
    // across the new dimension's threads.
    auto is_parallelized = [](structured_control_flow::Map* map) {
        return map->schedule_type().value() != structured_control_flow::ScheduleType_Sequential::value();
    };

    structured_control_flow::Map* outermost = nullptr;
    for (auto* ancestor = loop_analysis.parent_loop(&loop); ancestor != nullptr;
         ancestor = loop_analysis.parent_loop(ancestor)) {
        if (auto* map = dyn_cast<structured_control_flow::Map*>(ancestor)) {
            if (is_parallelized(map)) {
                outermost = map;
            }
        }
    }
    if (outermost == nullptr) {
        // No enclosing GPU map: nothing is folded, hence nothing is replicated.
        return false;
    }

    auto& users = analysis_manager.get<analysis::Users>();
    auto& arguments_analysis = analysis_manager.get<analysis::ArgumentsAnalysis>();
    // Containers whose entire lifetime is confined to the kernel are privatized per
    // thread (registers/stack, per-thread allocation), so a read-modify-write on one
    // races nothing. Only an accumulation on a container that escapes the kernel
    // (function argument/external or a transient living outside) is a hazard. Loop
    // induction variables are locals by definition, so this subsumes loop-control
    // bookkeeping without any special-casing.
    const auto& locals = arguments_analysis.locals(analysis_manager, *outermost);
    auto is_local = [&locals](const std::string& container) { return locals.count(container) != 0; };

    // A subtree performs an unsafe accumulation if it reads and writes the same
    // non-local container (e.g. `acc[i] += x`). A plain store is idempotent under
    // replication and therefore allowed.
    auto accumulates_on_shared = [&](structured_control_flow::ControlFlowNode& node) {
        analysis::UsersView view(users, node);
        std::unordered_set<std::string> writes;
        std::unordered_set<std::string> reads;
        for (auto* u : view.writes()) {
            writes.insert(u->container());
        }
        for (auto* u : view.moves()) {
            writes.insert(u->container());
        }
        // Views alias memory; treat conservatively as both a read and a write.
        for (auto* u : view.views()) {
            writes.insert(u->container());
            reads.insert(u->container());
        }
        for (auto* u : view.reads()) {
            reads.insert(u->container());
        }
        for (const auto& container : writes) {
            if (reads.count(container) != 0 && !is_local(container)) {
                return true;
            }
        }
        return false;
    };

    // Walk from `loop` up to (but excluding) the outermost GPU map, inspecting the
    // siblings at each level. Siblings above the kernel are not replicated and are
    // therefore not considered.
    structured_control_flow::ControlFlowNode* node = &loop;
    while (node != outermost) {
        auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(node->get_parent());
        if (sequence == nullptr) {
            break;
        }
        for (size_t i = 0; i < sequence->size(); ++i) {
            auto& sibling = sequence->at(i);
            if (&sibling == node) {
                continue;
            }
            // A sibling that is itself a GPU map is already parallelized: codegen maps
            // it onto its own threads, so its accumulation is not replicated.
            if (auto* sibling_map = dyn_cast<structured_control_flow::Map*>(&sibling)) {
                if (is_parallelized(sibling_map)) {
                    continue;
                }
            }
            if (accumulates_on_shared(sibling)) {
                return true;
            }
        }
        node = sequence->get_parent();
        if (node == nullptr) {
            break;
        }
    }

    return false;
}


// Explicit template instantiations for CUDA
template symbolic::Expression find_nested_gpu_blocksize<cuda::ScheduleType_CUDA>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template symbolic::Expression find_nested_gpu_iterations<cuda::ScheduleType_CUDA>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template bool is_outermost_gpu_map<
    cuda::ScheduleType_CUDA>(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager);

template symbolic::SymbolSet get_gpu_indvars<cuda::ScheduleType_CUDA>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template std::vector<structured_control_flow::Map*> get_gpu_maps<cuda::ScheduleType_CUDA>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

// Explicit template instantiations for ROCM
template symbolic::Expression find_nested_gpu_blocksize<rocm::ScheduleType_ROCM>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template symbolic::Expression find_nested_gpu_iterations<rocm::ScheduleType_ROCM>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template bool is_outermost_gpu_map<
    rocm::ScheduleType_ROCM>(structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager);

template symbolic::SymbolSet get_gpu_indvars<rocm::ScheduleType_ROCM>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

template std::vector<structured_control_flow::Map*> get_gpu_maps<rocm::ScheduleType_ROCM>(
    structured_control_flow::Map& node, analysis::AnalysisManager& analysis_manager, GPUDimension dimension
);

} // namespace gpu
} // namespace sdfg
