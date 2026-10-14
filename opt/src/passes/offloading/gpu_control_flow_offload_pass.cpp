#include "sdfg/passes/offloading/gpu_control_flow_offload_pass.h"

#include <string>
#include <utility>
#include <vector>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/cuda/cuda.h"
#include "sdfg/targets/offloading/data_offloading_node.h"
#include "sdfg/targets/rocm/rocm.h"
#include "sdfg/types/scalar.h"
#include "sdfg/visitor/immutable_structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

// Scans a root child; visit() returns true if the child must stay on the host.
template<typename Traits>
class ControlFlowOffloadScanner : public visitor::ImmutableStructuredSDFGVisitor {
private:
    structured_control_flow::ControlFlowNode& node_;

    static bool is_scheduled(const structured_control_flow::StructuredLoop& loop) {
        return loop.schedule_type().category() != structured_control_flow::ScheduleTypeCategory::None;
    }

public:
    ControlFlowOffloadScanner(
        StructuredSDFG& sdfg,
        analysis::AnalysisManager& analysis_manager,
        structured_control_flow::ControlFlowNode& node
    )
        : visitor::ImmutableStructuredSDFGVisitor(sdfg, analysis_manager), node_(node) {
    }

    bool visit() override {
        if (auto* sequence = dyn_cast<structured_control_flow::Sequence*>(&node_)) {
            return this->visit_internal(*sequence);
        }
        if (auto* block = dyn_cast<structured_control_flow::Block*>(&node_)) {
            return this->accept(*block);
        }
        if (auto* if_else = dyn_cast<structured_control_flow::IfElse*>(&node_)) {
            for (size_t i = 0; i < if_else->size(); ++i) {
                if (this->visit_internal(if_else->at(i).first)) {
                    return true;
                }
            }
            return false;
        }
        if (auto* while_loop = dyn_cast<structured_control_flow::While*>(&node_)) {
            return this->visit_internal(while_loop->root());
        }
        return true;
    }

    bool accept(structured_control_flow::Block& block) override {
        for (auto* library_node : block.dataflow().library_nodes()) {
            if (dynamic_cast<offloading::DataOffloadingNode*>(library_node) != nullptr ||
                Traits::is_gpu_implementation(library_node->implementation_type())) {
                return true;
            }
        }
        return false;
    }

    bool accept(structured_control_flow::Return& node) override {
        return true;
    }

    bool accept(structured_control_flow::For& loop) override {
        return is_scheduled(loop);
    }

    bool accept(structured_control_flow::Map& loop) override {
        return is_scheduled(loop);
    }

    bool accept(structured_control_flow::Reduce& loop) override {
        return is_scheduled(loop);
    }
};

// Single-iteration map wrapping consecutive children of a sequence.
struct Wrapper {
    structured_control_flow::Map* map;
    std::string indvar;
};

// A run of root children. Multi-node runs nest one wrapper per node inside the run's wrapper, so the
// fused run and its single-node fallbacks are checked against the same analyses.
struct Candidate {
    Wrapper run;
    std::vector<Wrapper> nodes;
};

Wrapper wrap(
    builder::StructuredSDFGBuilder& builder,
    structured_control_flow::Sequence& parent,
    const std::vector<structured_control_flow::ControlFlowNode*>& nodes
) {
    auto& first = *nodes.front();

    auto indvar_name = builder.find_new_name("__daisy_cf_offload_");
    builder.add_container(indvar_name, types::Scalar(types::PrimitiveType::Int64));
    auto indvar = symbolic::symbol(indvar_name);
    auto& map = builder.add_map_before(
        parent,
        first,
        indvar,
        symbolic::Lt(indvar, symbolic::one()),
        symbolic::zero(),
        symbolic::add(indvar, symbolic::one()),
        structured_control_flow::ScheduleType_Sequential::create(),
        first.debug_info()
    );
    for (auto* node : nodes) {
        builder.move_child(parent, parent.index(*node), map.root());
    }
    return {&map, indvar_name};
}

// Moves the map's children into its parent and removes the map; its indvar container is kept.
void dissolve(builder::StructuredSDFGBuilder& builder, structured_control_flow::Map& map) {
    auto& parent = *static_cast<structured_control_flow::Sequence*>(map.get_parent());
    builder.move_children(map.root(), parent, parent.index(map));
    builder.remove_child(parent, parent.index(map));
}

} // namespace

bool CUDAControlFlowOffloadTraits::is_gpu_implementation(const data_flow::ImplementationType& implementation_type) {
    return implementation_type == cuda::ImplementationType_CUDA ||
           implementation_type == cuda::ImplementationType_CUDAWithTransfers ||
           implementation_type == cuda::ImplementationType_CUDAWithoutTransfers;
}

bool ROCMControlFlowOffloadTraits::is_gpu_implementation(const data_flow::ImplementationType& implementation_type) {
    return implementation_type == rocm::ImplementationType_ROCM ||
           implementation_type == rocm::ImplementationType_ROCMWithTransfers ||
           implementation_type == rocm::ImplementationType_ROCMWithoutTransfers;
}

template<typename Traits>
GPUControlFlowOffloadPass<Traits>::GPUControlFlowOffloadPass(bool allow_dynamic_sizes)
    : allow_dynamic_sizes_(allow_dynamic_sizes) {
}

template<typename Traits>
bool GPUControlFlowOffloadPass<
    Traits>::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    auto& root = builder.subject().root();

    std::vector<std::vector<structured_control_flow::ControlFlowNode*>> runs(1);
    for (size_t i = 0; i < root.size(); ++i) {
        auto& child = root.at(i);
        ControlFlowOffloadScanner<Traits> scanner(builder.subject(), analysis_manager, child);
        if (!scanner.visit()) {
            runs.back().push_back(&child);
        } else if (!runs.back().empty()) {
            runs.emplace_back();
        }
    }
    if (runs.back().empty()) {
        runs.pop_back();
    }
    if (runs.empty()) {
        return false;
    }

    std::vector<Candidate> candidates;
    for (auto& nodes : runs) {
        Candidate candidate{wrap(builder, root, nodes), {}};
        if (nodes.size() > 1) {
            for (auto* node : nodes) {
                candidate.nodes.push_back(wrap(builder, candidate.run.map->root(), {node}));
            }
        }
        candidates.push_back(std::move(candidate));
    }
    analysis_manager.invalidate_all();

    // Collect: decide every wrapper before the SDFG is mutated again.
    auto can_offload = [&](const Wrapper& wrapper) {
        typename Traits::OffloadTransform
            transform(*wrapper.map, symbolic::one(), gpu::TargetLevel::X_GRID, allow_dynamic_sizes_);
        return transform.can_be_applied(builder, analysis_manager);
    };
    std::vector<structured_control_flow::Map*> offloaded;
    std::vector<structured_control_flow::Map*> dissolved;
    std::vector<std::string> unused_indvars;
    auto drop = [&](const Wrapper& wrapper) {
        dissolved.push_back(wrapper.map);
        unused_indvars.push_back(wrapper.indvar);
    };
    for (auto& candidate : candidates) {
        if (can_offload(candidate.run)) {
            offloaded.push_back(candidate.run.map);
            for (auto& node : candidate.nodes) {
                drop(node);
            }
            continue;
        }
        drop(candidate.run);
        for (auto& node : candidate.nodes) {
            if (can_offload(node)) {
                offloaded.push_back(node.map);
            } else {
                drop(node);
            }
        }
    }

    // Sweep: offloading only reads the per-map analyses cached above, so nothing is invalidated in between.
    for (auto* map : dissolved) {
        dissolve(builder, *map);
    }
    for (auto* map : offloaded) {
        typename Traits::OffloadTransform
            transform(*map, symbolic::one(), gpu::TargetLevel::X_GRID, allow_dynamic_sizes_);
        transform.apply(builder, analysis_manager);
    }
    for (auto& indvar : unused_indvars) {
        builder.remove_container(indvar);
    }

    analysis_manager.invalidate_all();
    return !offloaded.empty();
}

template class GPUControlFlowOffloadPass<CUDAControlFlowOffloadTraits>;
template class GPUControlFlowOffloadPass<ROCMControlFlowOffloadTraits>;

} // namespace passes
} // namespace sdfg
