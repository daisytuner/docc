#include "sdfg/code_motion/conflicts.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "sdfg/analysis/memory_layout_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/access_node.h"
#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/if_else.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/maps.h"
#include "sdfg/tiles/analysis/tile_analysis.h"
#include "sdfg/types/scalar.h"

namespace sdfg {
namespace code_motion {

namespace {

void collect_blocks(structured_control_flow::ControlFlowNode& node, std::vector<structured_control_flow::Block*>& blocks) {
    if (auto* block = dynamic_cast<structured_control_flow::Block*>(&node)) {
        blocks.push_back(block);
    } else if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            collect_blocks(sequence->at(i), blocks);
        }
    } else if (auto* branches = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < branches->size(); ++i) {
            collect_blocks(branches->at(i).first, blocks);
        }
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        collect_blocks(loop->root(), blocks);
    } else if (auto* loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        collect_blocks(loop->root(), blocks);
    }
}

/// Whether @p node transfers control out of itself: a return leaves the function, a break or continue the loop around,
/// unless a while loop within @p node holds it.
bool transfers_control(structured_control_flow::ControlFlowNode& node, bool in_while = false) {
    if (dynamic_cast<structured_control_flow::Return*>(&node)) {
        return true;
    }
    if (dynamic_cast<structured_control_flow::Break*>(&node) ||
        dynamic_cast<structured_control_flow::Continue*>(&node)) {
        return !in_while;
    }
    if (auto* sequence = dynamic_cast<structured_control_flow::Sequence*>(&node)) {
        for (size_t i = 0; i < sequence->size(); ++i) {
            if (transfers_control(sequence->at(i), in_while)) {
                return true;
            }
        }
    } else if (auto* branches = dynamic_cast<structured_control_flow::IfElse*>(&node)) {
        for (size_t i = 0; i < branches->size(); ++i) {
            if (transfers_control(branches->at(i).first, in_while)) {
                return true;
            }
        }
    } else if (auto* loop = dynamic_cast<structured_control_flow::StructuredLoop*>(&node)) {
        return transfers_control(loop->root(), in_while);
    } else if (auto* loop = dynamic_cast<structured_control_flow::While*>(&node)) {
        return transfers_control(loop->root(), true);
    }
    return false;
}

std::vector<Box> boxes_of(const analysis::MemoryTile& tile) {
    return {Box{tile.layout, tile.min_subset, tile.upper_subset()}};
}

/// The boxes MemoryLayoutAnalysis bounds @p container's accesses in @p node, through @p memlets, by; none if a memlet
/// has no tile.
std::optional<std::vector<Box>> boxes(
    analysis::MemoryLayoutAnalysis& layouts,
    structured_control_flow::ControlFlowNode& node,
    const std::string& container,
    const std::vector<const data_flow::Memlet*>& memlets
) {
    std::vector<Box> result;
    if (dynamic_cast<structured_control_flow::Block*>(&node)) {
        for (auto* memlet : memlets) {
            auto* tile = layouts.access(*memlet);
            if (!tile) {
                return std::nullopt;
            }
            auto box = boxes_of(*tile);
            result.insert(result.end(), box.begin(), box.end());
        }
        return result;
    }
    auto* groups = layouts.tile_groups(node, container);
    if (!groups) {
        return memlets.empty() ? std::optional<std::vector<Box>>(result) : std::nullopt;
    }
    std::set<const data_flow::Memlet*> covered;
    for (auto& group : *groups) {
        covered.insert(group.memlets.begin(), group.memlets.end());
        auto box = boxes_of(group.tile);
        result.insert(result.end(), box.begin(), box.end());
    }
    for (auto* memlet : memlets) {
        if (!covered.contains(memlet)) {
            return std::nullopt;
        }
    }
    return result;
}

/// @p box's elements, as an expression over fresh counters bounded in @p assumptions, in @p linear addresses or
/// indices.
symbolic::MultiExpression elements(const Box& box, bool linear, symbolic::Assumptions& assumptions, size_t& counters) {
    symbolic::MultiExpression indices;
    for (size_t d = 0; d < box.lower.size(); ++d) {
        auto counter = symbolic::symbol("__code_motion_element" + std::to_string(counters++));
        auto assumption = symbolic::Assumption::create(counter, types::Scalar(types::PrimitiveType::Int64));
        assumption.add_lower_bound(box.lower.at(d));
        assumption.add_upper_bound(box.upper.at(d));
        assumption.tight_lower_bound(box.lower.at(d));
        assumption.tight_upper_bound(box.upper.at(d));
        assumption.constant(false);
        assumptions.insert_or_assign(counter, std::move(assumption));
        indices.push_back(counter);
    }
    if (!linear) {
        return indices;
    }
    auto address = box.layout.offset();
    for (size_t d = 0; d < indices.size(); ++d) {
        address = symbolic::add(address, symbolic::mul(box.layout.strides().at(d), indices.at(d)));
    }
    return {address};
}

/// @p assumptions with an unbounded constant assumption on each symbol of @p expressions it lacks.
void complete(symbolic::Assumptions& assumptions, const symbolic::MultiExpression& expressions) {
    std::vector<symbolic::Expression> pending(expressions.begin(), expressions.end());
    while (!pending.empty()) {
        auto expression = pending.back();
        pending.pop_back();
        for (auto& atom : symbolic::atoms(expression)) {
            if (assumptions.contains(atom)) {
                continue;
            }
            auto assumption = symbolic::Assumption::create(atom, types::Scalar(types::PrimitiveType::Int64));
            assumption.constant(true);
            assumptions.insert_or_assign(atom, std::move(assumption));
        }
    }
    // The bounds of the fresh counters use symbols too.
    std::vector<symbolic::Expression> bounds;
    for (auto& [symbol, assumption] : assumptions) {
        for (auto& bound : assumption.lower_bounds()) {
            bounds.push_back(bound);
        }
        for (auto& bound : assumption.upper_bounds()) {
            bounds.push_back(bound);
        }
    }
    for (auto& bound : bounds) {
        for (auto& atom : symbolic::atoms(bound)) {
            if (!assumptions.contains(atom)) {
                auto assumption = symbolic::Assumption::create(atom, types::Scalar(types::PrimitiveType::Int64));
                assumption.constant(true);
                assumptions.insert_or_assign(atom, std::move(assumption));
            }
        }
    }
}

/// Whether @p first and @p second may share an element, with the symbols @p scope holds constant shared.
bool may_overlap(const Box& first, const Box& second, const symbolic::Assumptions& scope) {
    // Boxes of one index space compare per index; others by linear address.
    bool same_space = first.lower.size() == second.lower.size() &&
                      first.layout.shape().size() == second.layout.shape().size() &&
                      symbolic::eq(first.layout.offset(), second.layout.offset());
    for (size_t d = 0; same_space && d < first.layout.shape().size(); ++d) {
        same_space = symbolic::eq(first.layout.shape().at(d), second.layout.shape().at(d)) &&
                     symbolic::eq(first.layout.strides().at(d), second.layout.strides().at(d));
    }
    size_t counters = 0;
    auto first_assumptions = scope;
    auto second_assumptions = scope;
    auto first_elements = elements(first, !same_space, first_assumptions, counters);
    auto second_elements = elements(second, !same_space, second_assumptions, counters);
    // The dependence query compares instances at two values of an ordering symbol neither uses: with the other
    // symbols, its answer is whether the sets intersect.
    auto order = symbolic::symbol("__code_motion_order");
    auto ordering = symbolic::Assumption::create(order, types::Scalar(types::PrimitiveType::Int64));
    ordering.add_lower_bound(symbolic::zero());
    ordering.add_upper_bound(symbolic::one());
    ordering.tight_lower_bound(symbolic::zero());
    ordering.tight_upper_bound(symbolic::one());
    ordering.map(symbolic::add(order, symbolic::one()));
    ordering.constant(false);
    for (auto* assumptions : {&first_assumptions, &second_assumptions}) {
        assumptions->insert_or_assign(order, ordering);
    }
    complete(first_assumptions, first_elements);
    complete(first_assumptions, second_elements);
    complete(second_assumptions, first_elements);
    complete(second_assumptions, second_elements);
    for (auto& [symbol, assumption] : first_assumptions) {
        if (!second_assumptions.contains(symbol)) {
            second_assumptions.insert_or_assign(symbol, assumption);
        }
    }
    for (auto& [symbol, assumption] : second_assumptions) {
        if (!first_assumptions.contains(symbol)) {
            first_assumptions.insert_or_assign(symbol, assumption);
        }
    }
    try {
        return !symbolic::maps::
                    dependence_deltas(first_elements, second_elements, order, first_assumptions, second_assumptions)
                        .empty;
    } catch (...) {
        return true;
    }
}

} // namespace

Accesses accesses_of(
    StructuredSDFG& sdfg, analysis::AnalysisManager& analysis_manager, structured_control_flow::ControlFlowNode& node
) {
    auto& users = analysis_manager.get<analysis::Users>();
    auto& layouts = analysis_manager.get<analysis::MemoryLayoutAnalysis>();
    analysis::UsersView view(users, node);
    Accesses uses;
    for (auto* user : view.uses()) {
        auto [entry, added] = uses.try_emplace(user->container());
        auto& use = entry->second;
        // A symbol, in a subset, condition, header or assignment, is one element: the whole container.
        bool element = dynamic_cast<data_flow::AccessNode*>(user->element()) != nullptr;
        bool aliasing = user->use() == analysis::VIEW || user->use() == analysis::MOVE;
        use.reads |= user->use() == analysis::READ || aliasing;
        use.writes |= user->use() == analysis::WRITE || aliasing;
        if (added) {
            use.boxes = std::vector<Box>{};
        }
        if (!element || aliasing) {
            use.boxes = std::nullopt;
        }
    }
    std::vector<structured_control_flow::Block*> blocks;
    collect_blocks(node, blocks);
    std::map<std::string, std::vector<const data_flow::Memlet*>> memlets;
    for (auto* block : blocks) {
        for (auto& memlet : block->dataflow().edges()) {
            for (auto* end : {&memlet.src(), &memlet.dst()}) {
                auto* access = dynamic_cast<const data_flow::AccessNode*>(end);
                if (access && !dynamic_cast<const data_flow::ConstantNode*>(end)) {
                    auto& list = memlets[access->data()];
                    if (list.empty() || list.back() != &memlet) {
                        list.push_back(&memlet);
                    }
                }
            }
            // A pointer a library node invalidates is written all over.
            auto* access = dynamic_cast<const data_flow::AccessNode*>(&memlet.src());
            auto* library = dynamic_cast<const data_flow::LibraryNode*>(&memlet.dst());
            if (access && library) {
                auto meta = library->pointer_access_type(memlet);
                if (meta && meta->invalidated_after()) {
                    auto& use = uses[access->data()];
                    use.writes = true;
                    use.boxes = std::nullopt;
                }
            }
        }
    }
    // Library nodes write through pointers they also read; their metadata tells.
    auto summaries = tiles::TileAnalysis::summarize(sdfg, node);
    static const std::vector<const data_flow::Memlet*> none;
    for (auto& [container, use] : uses) {
        if (auto found = summaries.find(container); found != summaries.end()) {
            use.reads |= found->second.reads;
            use.writes |= found->second.writes;
            if (found->second.aliased) {
                use.boxes = std::nullopt;
            }
        }
        if (use.boxes) {
            auto found = memlets.find(container);
            use.boxes = boxes(layouts, node, container, found == memlets.end() ? none : found->second);
        }
    }
    return uses;
}

bool movable(StructuredSDFG& sdfg, structured_control_flow::ControlFlowNode& node) {
    if (transfers_control(node)) {
        return false;
    }
    std::vector<structured_control_flow::Block*> blocks;
    collect_blocks(node, blocks);
    for (auto* block : blocks) {
        for (auto* library : block->dataflow().library_nodes()) {
            // A library node's side effects are accesses the dependences account for if it accesses memory only
            // through memlets, every pointer it takes described: an allocation, a free or a tile copy. One without
            // memlets, as a barrier, has effects of its own.
            if (!library->side_effect()) {
                continue;
            }
            auto& dataflow = block->dataflow();
            bool connected = dataflow.in_degree(*library) > 0 || dataflow.out_degree(*library) > 0;
            if (!connected || library->pointer_use_creates_side_effects(dataflow, sdfg)) {
                return false;
            }
        }
    }
    return true;
}

bool may_conflict(const Accesses& first, const Accesses& second, const symbolic::Assumptions& scope, Dependences kinds) {
    for (auto& [container, first_use] : first) {
        auto found = second.find(container);
        if (found == second.end()) {
            continue;
        }
        auto& second_use = found->second;
        bool related = (kinds.flow && first_use.writes && second_use.reads) ||
                       (kinds.anti && first_use.reads && second_use.writes) ||
                       (kinds.output && first_use.writes && second_use.writes);
        if (!related) {
            continue;
        }
        if (!first_use.boxes || !second_use.boxes) {
            return true;
        }
        for (auto& first_box : *first_use.boxes) {
            for (auto& second_box : *second_use.boxes) {
                if (may_overlap(first_box, second_box, scope)) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool may_conflict(
    StructuredSDFG& sdfg,
    analysis::AnalysisManager& analysis_manager,
    structured_control_flow::ControlFlowNode& first,
    structured_control_flow::ControlFlowNode& second,
    const symbolic::Assumptions& scope,
    Dependences kinds
) {
    return may_conflict(
        accesses_of(sdfg, analysis_manager, first), accesses_of(sdfg, analysis_manager, second), scope, kinds
    );
}

bool reads_what_it_writes(const Accesses& accesses) {
    for (auto& [container, use] : accesses) {
        if (use.reads && use.writes) {
            return true;
        }
    }
    return false;
}

} // namespace code_motion
} // namespace sdfg
