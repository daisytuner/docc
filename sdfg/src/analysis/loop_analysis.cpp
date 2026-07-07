#include "sdfg/analysis/loop_analysis.h"
#include <unordered_set>
#include <vector>

#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/reduce.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"

namespace sdfg {
namespace analysis {

LoopAnalysis::LoopAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg), loops_(), loop_tree_() {}

void LoopAnalysis::init_new_loop_info(
    LoopState& info,
    uint32_t id,
    uint32_t loop_level,
    structured_control_flow::ControlFlowNode* loop,
    structured_control_flow::Map* map,
    structured_control_flow::ControlFlowNode* while_loop
) {
    auto is_elementwise = (map != nullptr && map->is_contiguous());
    LocalLoopInfo::LoopType type;
    if (while_loop != nullptr) {
        type = LocalLoopInfo::LoopType::While;
    } else if (map != nullptr) {
        type = LocalLoopInfo::LoopType::Map;
    } else {
        type = LocalLoopInfo::LoopType::For;
    }

    info.local = {
        .loop_id = id,
        .loop_level = loop_level,
        .type = type,
        .is_elementwise = is_elementwise,
        .contains_non_perfectly_nested = false,
        .contains_side_effects = false
    };
}

void LoopAnalysis::
    run(structured_control_flow::ControlFlowNode& scope,
        structured_control_flow::ControlFlowNode* parent_loop,
        uint32_t loop_level) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&scope};
    bool non_perfectly_nested = false;
    bool side_effects = false;

    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        structured_control_flow::While* new_while = nullptr;
        structured_control_flow::Map* new_map = nullptr;
        structured_control_flow::For* new_for = nullptr;
        structured_control_flow::ControlFlowNode* new_loop = nullptr;
        // Loop detected
        if (auto while_stmt = dyn_cast<structured_control_flow::While*>(current)) {
            new_loop = while_stmt;
            new_while = while_stmt;
        } else if (auto map_stmt = dyn_cast<structured_control_flow::Map*>(current)) {
            new_map = map_stmt;
            new_loop = map_stmt;
        } else if (auto for_stmt = dyn_cast<structured_control_flow::For*>(current)) {
            new_for = for_stmt;
            new_loop = for_stmt;
        } else if (auto loop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(current)) {
            // Generic structured loop (e.g. Reduce) that is neither a Map nor a For.
            new_loop = loop_stmt;
        }

        if (new_loop != nullptr) {
            auto id = this->loops_.size();
            this->loops_.push_back(new_loop);
            this->loop_tree_[new_loop] = parent_loop;
            this->loop_children_[parent_loop].push_back(new_loop);
            this->loop_children_[new_loop]; // ensure it gets created
            auto& loop_info = loop_infos_[new_loop];
            init_new_loop_info(loop_info, id, loop_level, new_loop, new_map, new_while);
        }

        if (auto block = dyn_cast<structured_control_flow::Block*>(current)) {
            non_perfectly_nested = true;
            for (auto& node : block->dataflow().nodes()) {
                if (auto library_node = dynamic_cast<data_flow::LibraryNode*>(&node)) {
                    if (library_node->side_effect()) { // also look at pointer metadata (no capture to not infer
                                                       // side-effects)
                        side_effects = true;
                        break;
                    }
                }
            }
        } else if (auto sequence_stmt = dyn_cast<structured_control_flow::Sequence*>(current)) {
            auto seq_entries = sequence_stmt->size();
            if (current != &scope) { // the body-root of each loop is expected to be a sequence
                non_perfectly_nested = true;
            }
            for (size_t i = 0; i < seq_entries; i++) {
                auto entry = sequence_stmt->at(i);
                queue.push_back(&entry.first);
            }
        } else if (auto if_else_stmt = dyn_cast<structured_control_flow::IfElse*>(current)) {
            non_perfectly_nested = true;
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dyn_cast<structured_control_flow::While*>(current)) {
            this->run(while_stmt->root(), while_stmt, loop_level + 1);
        } else if (auto for_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->run(for_stmt->root(), for_stmt, loop_level + 1);
        } else if (dyn_cast<structured_control_flow::Break*>(current)) {
            non_perfectly_nested = true;
            continue;
        } else if (dyn_cast<structured_control_flow::Continue*>(current)) {
            non_perfectly_nested = true;
            continue;
        } else if (dyn_cast<structured_control_flow::Return*>(current)) {
            non_perfectly_nested = true;
            continue;
        } else {
            throw std::runtime_error("Unsupported control flow node type");
        }
    }

    if (parent_loop != nullptr) {
        auto& state = loop_infos_.at(parent_loop);
        state.local.contains_side_effects = side_effects;
        state.local.contains_non_perfectly_nested = non_perfectly_nested;
    }
}

structured_control_flow::Sequence* LoopAnalysis::get_loop_content_root(structured_control_flow::ControlFlowNode* loop) {
    structured_control_flow::Sequence* root = nullptr;
    if (auto while_stmt = dyn_cast<structured_control_flow::While*>(loop)) {
        root = &while_stmt->root();
    } else if (auto loop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
        root = &loop_stmt->root();
    } else {
        throw std::runtime_error("Node is not a loop");
    }
    return root;
}

LoopAnalysis::LoopState& LoopAnalysis::compute_loop_infos(structured_control_flow::ControlFlowNode* loop) {
    // Recursion
    auto& loop_children = this->loop_children_.at(loop);
    for (auto& child_loop : loop_children) {
        this->compute_loop_infos(child_loop);
    }

    auto& loop_state = this->loop_infos_.at(loop);
    auto new_state = this->aggregate_loop_info(loop);
    loop_state.nest = std::move(new_state.nest);
    loop_state.local.last_child_id = new_state.last_child_id;
    return loop_state;
}

LoopAnalysis::AggregatedResult LoopAnalysis::aggregate_loop_info(structured_control_flow::ControlFlowNode* loop) const {
    auto& loop_state = this->loop_infos_.at(loop);
    const LocalLoopInfo& local = loop_state.local;
    auto& loop_children = this->loop_children_.at(loop);

    // Start from the existing infos to preserve fields that do not depend on the subtree
    // (element_id, loop_level, loopnest_index) and reset the aggregated, subtree-derived fields.
    AggregatedResult result;
    auto& info = result.nest;
    info.element_id = loop->element_id();
    info.loop_level = local.loop_level;
    info.num_loops = 1;
    info.num_maps = local.type == LocalLoopInfo::LoopType::Map ? 1 : 0;
    info.num_whiles = local.type == LocalLoopInfo::LoopType::While ? 1 : 0;
    info.num_fors = local.type == LocalLoopInfo::LoopType::For ? 1 : 0;
    info.max_depth = 1;
    result.last_child_id = local.loop_id;

    bool is_perfectly_nested = !local.contains_non_perfectly_nested;
    bool is_perfectly_parallel = local.type == LocalLoopInfo::LoopType::Map;
    bool is_elementwise = local.is_elementwise;
    bool map_stack_member = local.type == LocalLoopInfo::LoopType::Map;
    bool has_side_effects = local.contains_side_effects;
    bool map_stack_children = map_stack_member && loop_children.size() <= 1;
    uint32_t map_stack_depth = 0;

    for (auto& child_loop : loop_children) {
        auto& sub_state = this->loop_infos_.at(child_loop);
        auto& sub_info = sub_state.nest;
        info.num_loops += sub_info.num_loops;
        info.num_maps += sub_info.num_maps;
        info.num_fors += sub_info.num_fors;
        info.num_whiles += sub_info.num_whiles;
        info.max_depth = std::max(info.max_depth, 1 + sub_info.max_depth);
        result.last_child_id = std::max(result.last_child_id, sub_state.local.last_child_id);

        has_side_effects |= sub_info.has_side_effects;
        is_perfectly_nested &= sub_info.is_perfectly_nested;
        is_perfectly_parallel &= sub_info.is_perfectly_parallel;
        is_elementwise &= sub_info.is_elementwise;
        if (map_stack_children) {
            map_stack_depth = sub_info.map_stack_depth; // only allowed if there is just a single, direct child
        }
    }

    info.is_perfectly_parallel = is_perfectly_parallel;
    auto child_count = loop_children.size();
    if (child_count > 1) {
        is_perfectly_nested = false;
    } else if (child_count < 1) {
        is_perfectly_nested = true;
    }
    info.is_perfectly_nested = is_perfectly_nested;
    info.is_elementwise = is_elementwise && is_perfectly_nested & is_perfectly_parallel;
    info.has_side_effects = has_side_effects;
    if (map_stack_member) {
        if (local.contains_non_perfectly_nested) { // needs to be perfectly nested to form a larger stack
            map_stack_depth = 0;
        }
        info.map_stack_depth = map_stack_depth + 1;
    }

    return result;
}

void LoopAnalysis::reindex_loop_nest_idx() {
    auto& root_loops = this->loop_children_.at(nullptr);

    for (size_t i = 0; i < root_loops.size(); ++i) {
        this->loop_infos_[root_loops.at(i)].nest.loopnest_index = i;
    }
}

void LoopAnalysis::run(AnalysisManager& analysis_manager) {
    this->loops_.clear();
    this->loop_tree_.clear();
    this->loop_infos_.clear();
    this->loop_children_.clear();
    this->loop_children_[nullptr]; // ensure it exists
    this->run(this->sdfg_.root(), nullptr, 0);

    // Set loopnest indices for outermost loops
    int loopnest_index = 0;
    auto& root_loops = this->loop_children_.at(nullptr);

    for (auto* root_loop : root_loops) {
        this->compute_loop_infos(root_loop);
        this->loop_infos_[root_loop].nest.loopnest_index = loopnest_index++;
    }
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopAnalysis::loops() const {
    return this->loops_;
} // copies by default...
const std::vector<structured_control_flow::ControlFlowNode*>& LoopAnalysis::loops_in_pre_order() const {
    return this->loops_;
}

LoopInfo LoopAnalysis::loop_info(structured_control_flow::ControlFlowNode* loop) const {
    return this->loop_infos_.at(loop).nest;
}

const LocalLoopInfo& LoopAnalysis::loop_info_local(structured_control_flow::ControlFlowNode* loop) const {
    return this->loop_infos_.at(loop).local;
}

structured_control_flow::ControlFlowNode* LoopAnalysis::find_loop_by_indvar(const std::string& indvar) {
    for (auto& loop : this->loops_) {
        if (auto loop_stmt = dyn_cast<structured_control_flow::StructuredLoop*>(loop)) {
            if (loop_stmt->indvar()->get_name() == indvar) {
                return loop;
            }
        }
    }
    return nullptr;
}

const std::unordered_map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*>&
LoopAnalysis::loop_tree() const {
    return this->loop_tree_;
}

structured_control_flow::ControlFlowNode* LoopAnalysis::parent_loop(structured_control_flow::ControlFlowNode* loop
) const {
    return this->loop_tree_.at(loop);
}

const std::vector<structured_control_flow::ControlFlowNode*>& LoopAnalysis::outermost_loops() const {
    return loop_children_.at(nullptr);
}

bool LoopAnalysis::is_outermost_loop(structured_control_flow::ControlFlowNode* loop) const {
    if (this->loop_tree_.find(loop) == this->loop_tree_.end()) {
        return false;
    }
    return this->loop_tree_.at(loop) == nullptr;
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopAnalysis::outermost_maps() const {
    std::vector<structured_control_flow::ControlFlowNode*> outermost_maps_;
    for (const auto& [loop, parent] : this->loop_tree_) {
        if (dyn_cast<structured_control_flow::Map*>(loop)) {
            auto ancestor = parent;
            while (true) {
                if (ancestor == nullptr) {
                    outermost_maps_.push_back(loop);
                    break;
                }
                if (dyn_cast<structured_control_flow::Map*>(ancestor)) {
                    break;
                }
                ancestor = this->loop_tree_.at(ancestor);
            }
        }
    }
    return outermost_maps_;
}

const std::vector<sdfg::structured_control_flow::ControlFlowNode*>& LoopAnalysis::
    children(sdfg::structured_control_flow::ControlFlowNode* node) const {
    // Find unique child
    return loop_children_.at(node);
}

std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> LoopAnalysis::
    loop_tree_paths(sdfg::structured_control_flow::ControlFlowNode* loop) const {
    return this->loop_tree_paths(loop, this->loop_tree_);
};

std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> LoopAnalysis::loop_tree_paths(
    sdfg::structured_control_flow::ControlFlowNode* loop,
    const std::unordered_map<
        sdfg::structured_control_flow::ControlFlowNode*,
        sdfg::structured_control_flow::ControlFlowNode*>& tree
) const {
    // Collect all paths in tree starting from loop recursively (DFS)
    std::list<std::vector<sdfg::structured_control_flow::ControlFlowNode*>> paths;
    auto& children = this->children(loop);
    if (children.empty()) {
        paths.push_back({loop});
        return paths;
    }

    for (auto& child : children) {
        auto p = this->loop_tree_paths(child, tree);
        for (auto& path : p) {
            path.insert(path.begin(), loop);
            paths.push_back(path);
        }
    }

    return paths;
};

std::unordered_set<sdfg::structured_control_flow::ControlFlowNode*> LoopAnalysis::
    descendants(sdfg::structured_control_flow::ControlFlowNode* loop) const {
    std::unordered_set<sdfg::structured_control_flow::ControlFlowNode*> desc;
    std::list<sdfg::structured_control_flow::ControlFlowNode*> queue = {loop};
    while (!queue.empty()) { // TODO use ordered list directly
        auto current = queue.front();
        queue.pop_front();
        auto& children = this->children(current);
        for (auto& child : children) {
            if (desc.find(child) == desc.end()) {
                desc.insert(child);
                queue.push_back(child);
            }
        }
    }
    return desc;
}

void LoopAnalysis::dump_to_file(std::filesystem::path file) const {
    nlohmann::json arr;
    for (auto& loop : this->loops_) {
        auto& info = loop_infos_.at(loop);

        nlohmann::json entry;
        loop_info_local_to_json(entry["info"], info.local);
        entry["nest"] = loop_info_to_json(info.nest);
        arr.push_back(entry);
    }


    std::filesystem::create_directories(file.parent_path());
    std::ofstream out(file, std::ofstream::out);
    if (!out.is_open()) {
        std::cerr << "Could not open file " << file << " for writing JSON output." << std::endl;
    }
    out << arr << std::endl;
    out.close();
}

void LoopAnalysis::copied_loop(
    structured_control_flow::ControlFlowNode* existing_loop,
    structured_control_flow::ControlFlowNode* new_parent_loop,
    structured_control_flow::ControlFlowNode* new_loop,
    bool start_not_end
) {
    // `existing_loop` is only the conceptual source of the copy. The caller has already materialized
    // `new_loop` (with whatever subtree it has) inside `new_parent_loop` in the SDFG, so we simply
    // (re)derive everything for the new subtree directly from the SDFG instead of cloning infos.
    (void) existing_loop;

    // Where the root of the new subtree should live in the pre-order `loops_` vector.
    auto insert_idx = child_insertion_index(new_parent_loop, start_not_end);

    uint32_t new_loop_level = new_parent_loop == nullptr ? 0 : loop_infos_.at(new_parent_loop).local.loop_level + 1;

    // run() rescans the scope and overwrites the parent's local side-effect / non-perfect-nesting
    // flags based on that scan. Adding a loop child does not legitimately change either flag (they
    // describe non-loop body content), so snapshot and restore them around the call.
    bool had_parent = new_parent_loop != nullptr;
    bool saved_side_effects = false;
    bool saved_non_perfect = false;
    if (had_parent) {
        auto& pl = loop_infos_.at(new_parent_loop).local;
        saved_side_effects = pl.contains_side_effects;
        saved_non_perfect = pl.contains_non_perfectly_nested;
    }

    // Analyze the new subtree. run() appends it (in pre-order) to the end of `loops_`, registers it
    // in loop_tree_/loop_children_ (as the last child of new_parent_loop) and fills in local infos.
    auto block_start = static_cast<uint32_t>(loops_.size());
    this->run(*new_loop, new_parent_loop, new_loop_level);

    if (had_parent) {
        auto& pl = loop_infos_.at(new_parent_loop).local;
        pl.contains_side_effects = saved_side_effects;
        pl.contains_non_perfectly_nested = saved_non_perfect;
    }

    // Bottom-up fill the nest infos (and last_child_id spans) of the freshly added subtree. At this
    // point the subtree still sits at the tail of `loops_` with contiguous, correct relative ids.
    this->compute_loop_infos(new_loop);

    // run() always appends the new root as the last child; move it to the front if requested.
    if (start_not_end) {
        auto& np_children = loop_children_.at(new_parent_loop);
        np_children.pop_back();
        np_children.insert(np_children.begin(), new_loop);
    }

    // Splice the new block out of the tail of `loops_` and into its pre-order position.
    std::vector<structured_control_flow::ControlFlowNode*> new_block(loops_.begin() + block_start, loops_.end());
    loops_.erase(loops_.begin() + block_start, loops_.end());
    loops_.insert(loops_.begin() + insert_idx, new_block.begin(), new_block.end());

    // Everything from the insertion point onward now has a stale loop_id (the block moved, the rest
    // shifted right). reindex preserves each node's subtree span, which is correct for all of them.
    reindex(loops_.begin() + insert_idx, loops_.end());

    // Propagate the new subtree's contribution (grown last_child_id, changed nest aggregates) up the
    // parent chain. The new subtree itself already holds up-to-date infos from compute_loop_infos.
    if (new_parent_loop != nullptr) {
        propagate_changed_nest_info(loops_.begin() + loop_infos_.at(new_parent_loop).local.loop_id);
    }

    reindex_loop_nest_idx();
}

uint32_t LoopAnalysis::child_insertion_index(structured_control_flow::ControlFlowNode* new_parent, bool start_not_end)
    const {
    const auto& children = loop_children_.at(new_parent);
    auto it = start_not_end ? children.begin() : children.end();
    if (it != children.end()) {
        // Insert right where the chosen child's subtree starts.
        return loop_infos_.at(*it).local.loop_id;
    }
    if (new_parent == nullptr) {
        // Appending after all root-level loops.
        return static_cast<uint32_t>(loops_.size());
    }
    // Insert right after the new parent's current subtree.
    return loop_infos_.at(new_parent).local.last_child_id + 1;
}

void LoopAnalysis::moved_loop(
    structured_control_flow::ControlFlowNode* existing_loop,
    structured_control_flow::StructuredLoop* new_parent,
    bool start_not_end
) {
    auto& moved_state = loop_infos_.at(existing_loop);
    auto old_parent = loop_tree_.at(existing_loop);

    // The moved subtree occupies the contiguous pre-order range [first_moved_idx, next_after_moved_idx).
    auto first_moved_idx = moved_state.local.loop_id;
    auto next_after_moved_idx = moved_state.local.last_child_id + 1;
    auto moved_span = next_after_moved_idx - first_moved_idx;

    // Translate the requested child position (front/back) into an index in the pre-order `loops_`.
    auto insert_idx = child_insertion_index(new_parent, start_not_end);
    auto& new_parent_children = loop_children_.at(new_parent);

    // Snapshot the moved block of nodes before we mutate `loops_`.
    std::vector<structured_control_flow::ControlFlowNode*>
        moved_block(loops_.begin() + first_moved_idx, loops_.begin() + next_after_moved_idx);

    // Update the level of every node in the moved subtree (its root's level becomes new_parent.level + 1).
    auto level_delta = static_cast<int64_t>(loop_infos_.at(new_parent).local.loop_level) + 1 -
                       static_cast<int64_t>(moved_state.local.loop_level);
    if (level_delta != 0) {
        for (auto* node : moved_block) {
            auto& local = loop_infos_.at(node).local;
            local.loop_level = static_cast<uint32_t>(static_cast<int64_t>(local.loop_level) + level_delta);
        }
    }

    // Detach from the old parent's child list and reparent the moved root.
    auto& old_parent_children = loop_children_.at(old_parent);
    old_parent_children.erase(
        std::remove(old_parent_children.begin(), old_parent_children.end(), existing_loop), old_parent_children.end()
    );
    loop_tree_[existing_loop] = new_parent;

    // Splice the moved block out of `loops_` and back in at the insertion point.
    // Erasing first shifts everything after the block left by `moved_span`; account for that when the
    // insertion point lies past the removed block.
    loops_.erase(loops_.begin() + first_moved_idx, loops_.begin() + next_after_moved_idx);
    if (insert_idx > next_after_moved_idx) {
        insert_idx -= moved_span;
    } else if (insert_idx > first_moved_idx) {
        // Insertion point was inside the moved block (only possible if new_parent is within the moved
        // subtree, which is not a valid move) -> clamp to the block's start.
        insert_idx = first_moved_idx;
    }
    loops_.insert(loops_.begin() + insert_idx, moved_block.begin(), moved_block.end());

    // Insert into the new parent's child list. Recompute the iterator since `start_not_end` may not be the
    // only supported mode in the future; for begin()/end() this is stable across the splice above.
    auto new_insert_it = start_not_end ? new_parent_children.begin() : new_parent_children.end();
    new_parent_children.insert(new_insert_it, existing_loop);

    // All loop_ids from the lower of the two touched regions onward are now stale: reindex the whole span.
    auto reindex_from = std::min(first_moved_idx, insert_idx);
    reindex(loops_.begin() + reindex_from, loops_.end());

    // Propagate nest-info changes up both affected parent chains. The deeper of the two parents must be
    // processed first so the shallower one aggregates already-updated children.
    structured_control_flow::ControlFlowNode* lo = old_parent;
    structured_control_flow::ControlFlowNode* hi = new_parent;
    if (lo != hi) {
        auto level_of = [&](structured_control_flow::ControlFlowNode* n) {
            return n == nullptr ? -1 : static_cast<int64_t>(loop_infos_.at(n).local.loop_level);
        };
        if (level_of(lo) < level_of(hi)) {
            std::swap(lo, hi);
        }
        if (lo != nullptr) {
            propagate_changed_nest_info(loops_.begin() + loop_infos_.at(lo).local.loop_id);
        }
        if (hi != nullptr) {
            propagate_changed_nest_info(loops_.begin() + loop_infos_.at(hi).local.loop_id);
        }
    } else if (new_parent != nullptr) {
        propagate_changed_nest_info(loops_.begin() + loop_infos_.at(new_parent).local.loop_id);
    }

    reindex_loop_nest_idx();
}

void LoopAnalysis::reindex(
    std::vector<structured_control_flow::ControlFlowNode*>::iterator start,
    std::vector<structured_control_flow::ControlFlowNode*>::iterator end
) {
    for (auto it = start; it != end; ++it) {
        auto& local_info = loop_infos_.at(*it).local;
        auto new_loop_id = static_cast<uint32_t>(std::distance(loops_.begin(), it));
        // The number of (transitive) descendants is unchanged; only the base index shifts.
        auto span = local_info.last_child_id - local_info.loop_id;
        local_info.loop_id = new_loop_id;
        local_info.last_child_id = new_loop_id + span;
    }
}

void LoopAnalysis::propagate_changed_nest_info(std::vector<structured_control_flow::ControlFlowNode*>::iterator top) {
    // Starting at `top`, recompute each loop's nest info from its (already up-to-date) children and local state.
    // Walk up the parent chain, stopping as soon as a loop's info is unchanged or the root (nullptr) is reached.
    auto loop_info_equal = [](const LoopInfo& a, const LoopInfo& b) {
        bool equal = true;
#define X(type, name, val) equal = equal && (a.name == b.name);
        LOOP_INFO_PROPERTIES
#undef X
        return equal;
    };

    structured_control_flow::ControlFlowNode* current = *top;
    while (current != nullptr) {
        auto new_info = this->aggregate_loop_info(current);
        auto& state = this->loop_infos_.at(current);
        if (state.local.last_child_id == new_info.last_child_id && loop_info_equal(state.nest, new_info.nest)) {
            break; // no change here means nothing changes further up
        }
        state.local.last_child_id = new_info.last_child_id;
        state.nest = std::move(new_info.nest);
        current = this->loop_tree_.at(current);
    }
}

void LoopAnalysis::removed_loop(structured_control_flow::ControlFlowNode* existing_loop) {
    auto& state = loop_infos_.at(existing_loop);
    auto removed_loop_idx = state.local.loop_id;
    auto next_not_removed_idx = state.local.last_child_id + 1;
    auto parent_of_removed = loop_tree_.at(existing_loop);
    auto first_removed = loops_.begin() + removed_loop_idx;
    auto next_not_removed = loops_.begin() + next_not_removed_idx;

    for (auto* elem : std::ranges::subrange(first_removed, next_not_removed) | std::views::reverse) {
        loop_infos_.erase(elem);
        loop_tree_.erase(elem);
        loop_children_.erase(elem);
    }

    auto& parent_children = loop_children_.at(parent_of_removed);
    parent_children
        .erase(std::remove(parent_children.begin(), parent_children.end(), existing_loop), parent_children.end());

    loops_.erase(first_removed, next_not_removed);

    reindex(first_removed, loops_.end());

    if (parent_of_removed != nullptr) {
        propagate_changed_nest_info(loops_.begin() + loop_infos_.at(parent_of_removed).local.loop_id);
    }

    reindex_loop_nest_idx();
}

void LoopAnalysis::
    added_local_contents(structured_control_flow::ControlFlowNode* loop, bool side_effects, bool non_perfectly_nested) {
    auto& state = loop_infos_.at(loop);
    state.local.contains_side_effects = side_effects;
    state.local.contains_non_perfectly_nested = non_perfectly_nested;

    propagate_changed_nest_info(loops_.begin() + state.local.loop_id);
}

void loop_info_local_to_json(nlohmann::json& j, const LocalLoopInfo& info) {
    j["loop_id"] = info.loop_id;
    j["last_child_id"] = info.last_child_id;
    j["loop_level"] = info.loop_level;
    j["type"] = info.type;
    j["is_elementwise"] = info.is_elementwise;
    j["contains_non_perfectly_nested"] = info.contains_non_perfectly_nested;
    j["contains_side_effects"] = info.contains_side_effects;
}

} // namespace analysis
} // namespace sdfg
