#include "sdfg/passes/offloading/lanes_map_predication.h"

#include <map>
#include <set>
#include <string>
#include <vector>

#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/data_flow/library_nodes/barrier_local_node.h"
#include "sdfg/exceptions.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/symbolic/symbolic.h"
#include "sdfg/targets/gpu/gpu_map_utils.h"
#include "sdfg/targets/gpu/gpu_offload_schedule_type.h"
#include "sdfg/targets/gpu/gpu_wave_barrier_node.h"
#include "sdfg/tiles/library_nodes/pipeline_node.h"
#include "sdfg/tiles/library_nodes/tile_copy_node.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace passes {

namespace {

using structured_control_flow::Block;
using structured_control_flow::IfElse;
using structured_control_flow::Sequence;

// Nodes every lane of the wave (or block) must execute together.
bool is_collective(const data_flow::LibraryNode& node) {
    const auto& code = node.code().value();
    return code.starts_with("gpu::") || code == data_flow::LibraryNodeType_BarrierLocal.value() ||
           code == tiles::LibraryNodeType_TileCopy.value() || code.starts_with("pipeline_");
}

// What a block does, as seen by the lanes of a group.
struct BlockEffects {
    bool collective = false;
    bool lane_indexed = false;
    // A write whose target is not a named container (side-effecting node, written pointer operand).
    bool opaque_write = false;
    std::set<std::string> shared_writes;
    std::set<std::string> private_writes;
    std::set<std::string> reads;

    bool has_shared_writes() const {
        return opaque_write || !shared_writes.empty();
    }
};

// Memory every lane of the group can observe: device memory or anything written through a pointer.
bool lane_shared(const StructuredSDFG& sdfg, const std::string& container) {
    const auto& type = sdfg.type(container);
    return type.type_id() == types::TypeID::Pointer || type.storage_type().is_device();
}

bool is_leader_guard(const IfElse& if_else, const symbolic::Condition& leader) {
    return if_else.size() == 1 && symbolic::eq(if_else.at(0).second, leader);
}

BlockEffects analyze(const StructuredSDFG& sdfg, const Block& block) {
    BlockEffects effects;
    const auto& graph = block.dataflow();
    for (const auto* library_node : graph.library_nodes()) {
        if (is_collective(*library_node)) {
            effects.collective = true;
            continue;
        }
        for (const auto& symbol : library_node->symbols()) {
            effects.lane_indexed |= symbolic::eq(symbol, symbolic::threadIdx_x());
        }
        if (library_node->side_effect()) {
            effects.opaque_write = true;
        }
        for (const auto& memlet : graph.in_edges(*library_node)) {
            if (memlet.result_type(sdfg)->type_id() != types::TypeID::Pointer) {
                continue;
            }
            auto access = library_node->pointer_access_type(memlet);
            if (!access || access->may_contain_writes()) {
                effects.opaque_write = true;
            }
        }
    }
    for (const auto& memlet : graph.edges()) {
        for (const auto& index : memlet.subset()) {
            effects.lane_indexed |= symbolic::uses(index, symbolic::threadIdx_x());
        }
        if (auto* src = dynamic_cast<const data_flow::AccessNode*>(&memlet.src())) {
            effects.reads.insert(src->data());
        }
        auto* dst = dynamic_cast<const data_flow::AccessNode*>(&memlet.dst());
        if (dst == nullptr) {
            continue;
        }
        // Writes of collective nodes are distributed over the lanes by the node itself.
        auto* writer = dynamic_cast<const data_flow::LibraryNode*>(&memlet.src());
        if (writer != nullptr && is_collective(*writer)) {
            continue;
        }
        // A reference memlet assigns the pointer variable itself, not the memory it points to.
        if (memlet.type() != data_flow::MemletType::Reference && lane_shared(sdfg, dst->data())) {
            effects.shared_writes.insert(dst->data());
        } else {
            effects.private_writes.insert(dst->data());
        }
    }
    return effects;
}

// Records the effects of every block of a lanes map body, and which blocks already sit under a
// leader-lane guard (from an earlier run of the pass).
class EffectCollector : public visitor::ActualStructuredSDFGVisitor {
    const StructuredSDFG& sdfg_;
    const symbolic::Condition& leader_;
    std::map<const Block*, BlockEffects>& effects_;
    std::set<const Block*>& guarded_;
    int guard_depth_ = 0;

public:
    EffectCollector(
        const StructuredSDFG& sdfg,
        const symbolic::Condition& leader,
        std::map<const Block*, BlockEffects>& effects,
        std::set<const Block*>& guarded
    )
        : sdfg_(sdfg), leader_(leader), effects_(effects), guarded_(guarded) {
    }

    using visitor::ActualStructuredSDFGVisitor::visit;

    bool visit(Block& block) override {
        effects_[&block] = analyze(sdfg_, block);
        if (guard_depth_ > 0) {
            guarded_.insert(&block);
        }
        return false;
    }

    bool visit(IfElse& if_else) override {
        const bool guard = is_leader_guard(if_else, leader_);
        guard_depth_ += guard;
        visitor::ActualStructuredSDFGVisitor::visit(if_else);
        guard_depth_ -= guard;
        return true;
    }
};

// Wraps the runs of blocks with lane-shared writes in each sequence of a lanes map body into a
// leader-lane guard. A sequence is rewritten after its children were visited, so the rewrite never
// invalidates a traversal still in progress.
class Predicator : public visitor::ActualStructuredSDFGVisitor {
    builder::StructuredSDFGBuilder& builder_;
    const symbolic::Condition& leader_;
    const std::map<const Block*, BlockEffects>& effects_;
    std::set<const Block*>& guarded_;
    bool changed_ = false;

    bool needs_guard(const Block& block) const {
        const auto& effects = effects_.at(&block);
        if (effects.lane_indexed || !effects.has_shared_writes()) {
            return false;
        }
        if (effects.collective) {
            throw InvalidSDFGException(
                "LanesMapPredication: block " + std::to_string(block.element_id()) +
                " mixes wave-collective nodes with lane-shared writes"
            );
        }
        return true;
    }

    // Whether a block outside the guarded set reads what @p writes wrote (anything, for opaque writes).
    bool read_unguarded(const BlockEffects& writes) const {
        for (const auto& [block, effects] : effects_) {
            if (guarded_.contains(block)) {
                continue;
            }
            if (writes.opaque_write && !effects.reads.empty()) {
                return true;
            }
            for (const auto& container : writes.shared_writes) {
                if (effects.reads.contains(container)) {
                    return true;
                }
            }
        }
        return false;
    }

    void check_private_results(const Block& block) const {
        for (const auto& container : effects_.at(&block).private_writes) {
            for (const auto& [other, effects] : effects_) {
                if (other != &block && !guarded_.contains(other) && effects.reads.contains(container)) {
                    throw InvalidSDFGException(
                        "LanesMapPredication: predicated block " + std::to_string(block.element_id()) + " produces '" +
                        container + "', which other lanes read"
                    );
                }
            }
        }
    }

    void guard_run(Sequence& sequence, size_t begin, size_t end) {
        BlockEffects writes;
        for (size_t i = begin; i < end; ++i) {
            auto& block = static_cast<Block&>(sequence.at(i));
            check_private_results(block);
            const auto& effects = effects_.at(&block);
            writes.opaque_write |= effects.opaque_write;
            writes.shared_writes.insert(effects.shared_writes.begin(), effects.shared_writes.end());
        }
        auto& if_else = builder_.add_if_else_before(sequence, sequence.at(begin));
        auto& branch = builder_.add_case(if_else, leader_);
        for (size_t i = begin; i < end; ++i) {
            builder_.move_child(sequence, begin + 1, branch, branch.size());
        }
        if (read_unguarded(writes)) {
            auto& barrier = builder_.add_block_after(sequence, if_else);
            builder_.add_library_node<gpu::GpuWaveBarrierNode>(barrier, DebugInfo());
        }
        changed_ = true;
    }

public:
    Predicator(
        builder::StructuredSDFGBuilder& builder,
        const symbolic::Condition& leader,
        const std::map<const Block*, BlockEffects>& effects,
        std::set<const Block*>& guarded
    )
        : builder_(builder), leader_(leader), effects_(effects), guarded_(guarded) {
    }

    bool changed() const {
        return changed_;
    }

    using visitor::ActualStructuredSDFGVisitor::visit;

    bool visit(IfElse& if_else) override {
        // Already predicated: everything below runs on the leader lane only.
        if (is_leader_guard(if_else, leader_)) {
            return false;
        }
        return visitor::ActualStructuredSDFGVisitor::visit(if_else);
    }

    bool visit(Sequence& sequence) override {
        visitor::ActualStructuredSDFGVisitor::visit(sequence);

        // Maximal runs of consecutive blocks to predicate, wrapped back to front to keep indices valid.
        std::vector<std::pair<size_t, size_t>> runs;
        for (size_t i = 0; i < sequence.size();) {
            auto* block = dynamic_cast<Block*>(&sequence.at(i));
            if (block == nullptr || !needs_guard(*block)) {
                ++i;
                continue;
            }
            size_t end = i + 1;
            while (end < sequence.size()) {
                auto* next = dynamic_cast<Block*>(&sequence.at(end));
                if (next == nullptr || !needs_guard(*next)) {
                    break;
                }
                ++end;
            }
            runs.emplace_back(i, end);
            i = end;
        }
        for (const auto& [begin, end] : runs) {
            for (size_t i = begin; i < end; ++i) {
                guarded_.insert(&static_cast<Block&>(sequence.at(i)));
            }
        }
        for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
            guard_run(sequence, it->first, it->second);
        }
        return true;
    }
};

// Finds the X_BLOCK maps with lanes > 1 and predicates each body.
class LanesMapFinder : public visitor::ActualStructuredSDFGVisitor {
    builder::StructuredSDFGBuilder& builder_;
    bool changed_ = false;

public:
    explicit LanesMapFinder(builder::StructuredSDFGBuilder& builder) : builder_(builder) {
    }

    bool changed() const {
        return changed_;
    }

    using visitor::ActualStructuredSDFGVisitor::visit;

    bool visit(structured_control_flow::Map& map) override {
        const auto& schedule = map.schedule_type();
        if (schedule.category() != structured_control_flow::ScheduleTypeCategory::Offloader ||
            !gpu::ScheduleType_GPU_Offload::target_level_if_exists(schedule) ||
            gpu::ScheduleType_GPU_Offload::lanes(schedule)->as_int() <= 1) {
            return visitor::ActualStructuredSDFGVisitor::visit(map);
        }
        // Lanes are X_BLOCK only, so a lanes map never nests in another one.
        const auto leader = gpu::lane_leader_condition(gpu::ScheduleType_GPU_Offload::lanes(schedule)->as_int());
        std::map<const Block*, BlockEffects> effects;
        std::set<const Block*> guarded;
        EffectCollector collector(builder_.subject(), leader, effects, guarded);
        collector.visit(map.root());
        Predicator predicator(builder_, leader, effects, guarded);
        predicator.visit(map.root());
        changed_ |= predicator.changed();
        return true;
    }
};

} // namespace

std::string LanesMapPredication::name() {
    return "LanesMapPredication";
}

bool LanesMapPredication::run_pass(builder::StructuredSDFGBuilder& builder, analysis::AnalysisManager& analysis_manager) {
    LanesMapFinder finder(builder);
    finder.visit(builder.subject().root());
    if (finder.changed()) {
        analysis_manager.invalidate_all();
    }
    return finder.changed();
}

} // namespace passes
} // namespace sdfg
