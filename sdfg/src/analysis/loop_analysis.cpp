#include "sdfg/analysis/loop_analysis.h"
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"

namespace sdfg {
namespace analysis {

LoopAnalysis::LoopAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg), loops_(), loop_tree_(DFSLoopComparator(&loops_)) {}

void LoopAnalysis::init_new_loop_info(
    std::pair<LocalLoopInfo, LoopInfo>& info,
    uint32_t loop_level,
    structured_control_flow::ControlFlowNode* loop,
    structured_control_flow::Map* map,
    structured_control_flow::ControlFlowNode* while_loop
) {
    auto is_elementwise = (map != nullptr && map->is_contiguous());

    info.first = {
        .loop_id = this->next_loop_id_++,
        .loop_level = loop_level,
        .is_map = map != nullptr,
        .is_elementwise = is_elementwise,
        .contains_non_perfectly_nested = false,
        .contains_side_effects = false
    };
    info.second.element_id = loop->element_id();
    info.second.is_elementwise = is_elementwise;
    info.second.has_side_effects = false;
    info.second.max_depth = 1;
    info.second.num_loops = 1;
    info.second.loop_level = loop_level;
    if (map != nullptr) {
        info.second.num_maps = 1;
        info.second.num_fors = 0;
        info.second.num_whiles = 0;
    } else if (while_loop != nullptr) {
        info.second.num_maps = 0;
        info.second.num_fors = 0;
        info.second.num_whiles = 1;
    } else {
        info.second.num_maps = 0;
        info.second.num_fors = 1;
        info.second.num_whiles = 0;
    }
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
        if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            new_loop = while_stmt;
            new_while = while_stmt;
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(current)) {
            new_map = map_stmt;
            new_loop = map_stmt;
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            new_for = for_stmt;
            new_loop = for_stmt;
        }

        if (new_loop != nullptr) {
            this->loops_.push_back(new_loop);
            this->loop_tree_[new_loop] = parent_loop;
            this->loop_children_[parent_loop].push_back(new_loop);
            this->loop_children_[new_loop]; // ensure it gets created
            auto& loop_info = loop_infos_[new_loop];
            init_new_loop_info(loop_info, loop_level, new_loop, new_map, new_while);
        }

        if (auto block = dynamic_cast<structured_control_flow::Block*>(current)) {
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
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            auto seq_entries = sequence_stmt->size();
            if (current != &scope) { // the body-root of each loop is expected to be a sequence
                non_perfectly_nested = true;
            }
            for (size_t i = 0; i < seq_entries; i++) {
                auto entry = sequence_stmt->at(i);
                if (i > 0 || !entry.second.empty()) {
                    non_perfectly_nested = true;
                }
                queue.push_back(&entry.first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            non_perfectly_nested = true;
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->run(while_stmt->root(), while_stmt, loop_level + 1);
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->run(for_stmt->root(), for_stmt, loop_level + 1);
        } else if (dynamic_cast<structured_control_flow::Break*>(current)) {
            non_perfectly_nested = true;
            continue;
        } else if (dynamic_cast<structured_control_flow::Continue*>(current)) {
            non_perfectly_nested = true;
            continue;
        } else if (dynamic_cast<structured_control_flow::Return*>(current)) {
            non_perfectly_nested = true;
            continue;
        } else {
            throw std::runtime_error("Unsupported control flow node type");
        }
    }

    if (parent_loop != nullptr) {
        auto child_count = loop_children_.at(parent_loop).size();
        auto& state = loop_infos_.at(parent_loop);
        state.first.last_child_id = this->next_loop_id_ - 1;
        state.first.contains_side_effects = side_effects;

        if (child_count > 1) {
            non_perfectly_nested = true;
        } else if (child_count < 1) {
            non_perfectly_nested = false;
        }
        if (non_perfectly_nested) {
            state.first.contains_non_perfectly_nested = non_perfectly_nested;
        }
    }
}

structured_control_flow::Sequence* LoopAnalysis::get_loop_content_root(structured_control_flow::ControlFlowNode* loop) {
    structured_control_flow::Sequence* root = nullptr;
    if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(loop)) {
        root = &while_stmt->root();
    } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
        root = &loop_stmt->root();
    } else {
        throw std::runtime_error("Node is not a loop");
    }
    return root;
}

LoopState& LoopAnalysis::compute_loop_infos(structured_control_flow::ControlFlowNode* loop) {
    // Recursion
    auto& loop_state = this->loop_infos_.at(loop);
    LoopInfo& info = loop_state.second;
    auto& loop_children = this->loop_children_.at(loop);

    bool is_perfectly_nested = !loop_state.first.contains_non_perfectly_nested;
    bool is_perfectly_parallel = loop_state.first.is_map;
    bool is_elementwise = loop_state.first.is_elementwise;
    bool map_stack_member = loop_state.first.is_map;
    bool has_side_effects = loop_state.first.contains_side_effects;
    bool map_stack_children = map_stack_member && loop_children.size() <= 1;
    uint32_t map_stack_depth = 0;

    for (auto& child_loop : loop_children) {
        this->compute_loop_infos(child_loop);

        auto& sub_info = this->loop_infos_.at(child_loop).second;
        info.num_loops += sub_info.num_loops;
        info.num_maps += sub_info.num_maps;
        info.num_fors += sub_info.num_fors;
        info.num_whiles += sub_info.num_whiles;
        info.max_depth = std::max(info.max_depth, 1 + sub_info.max_depth);

        has_side_effects |= sub_info.has_side_effects;
        is_perfectly_nested &= sub_info.is_perfectly_nested;
        is_perfectly_parallel &= sub_info.is_perfectly_parallel;
        is_elementwise &= sub_info.is_elementwise;
        if (map_stack_children) {
            map_stack_depth = sub_info.map_stack_depth; // only allowed if there is just a single, direct child
        }
    }

    info.is_perfectly_parallel = is_perfectly_parallel;
    info.is_perfectly_nested = is_perfectly_nested;
    info.is_elementwise &= is_elementwise && is_perfectly_nested & is_perfectly_parallel;
    info.has_side_effects = has_side_effects;
    if (map_stack_member) {
        if (!is_perfectly_nested) { // needs to be perfectly nested to form a larger stack
            map_stack_depth = 0;
        }
        info.map_stack_depth = map_stack_depth + 1;
    }

    return loop_state;
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
    auto root_it = this->loop_children_.find(nullptr);
    if (root_it != this->loop_children_.end()) {
        auto& root_loops = root_it->second;
        for (auto* root_loop : root_loops) {
            this->compute_loop_infos(root_loop);
            this->loop_infos_[root_loop].second.loopnest_index = loopnest_index++;
        }
    }
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopAnalysis::loops() const { return this->loops_; }

LoopInfo LoopAnalysis::loop_info(structured_control_flow::ControlFlowNode* loop) const {
    return this->loop_infos_.at(loop).second;
}

const LocalLoopInfo& LoopAnalysis::loop_info_local(structured_control_flow::ControlFlowNode* loop) const {
    return this->loop_infos_.at(loop).first;
}

structured_control_flow::ControlFlowNode* LoopAnalysis::find_loop_by_indvar(const std::string& indvar) {
    for (auto& loop : this->loops_) {
        if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(loop)) {
            if (loop_stmt->indvar()->get_name() == indvar) {
                return loop;
            }
        }
    }
    return nullptr;
}

const std::map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*, DFSLoopComparator>&
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
        if (dynamic_cast<structured_control_flow::Map*>(loop)) {
            auto ancestor = parent;
            while (true) {
                if (ancestor == nullptr) {
                    outermost_maps_.push_back(loop);
                    break;
                }
                if (dynamic_cast<structured_control_flow::Map*>(ancestor)) {
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
    const std::map<
        sdfg::structured_control_flow::ControlFlowNode*,
        sdfg::structured_control_flow::ControlFlowNode*,
        DFSLoopComparator>& tree
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
    while (!queue.empty()) {
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

        arr.push_back(loop_info_to_json(info.second));
    }


    std::filesystem::create_directories(file.parent_path());
    std::ofstream out(file, std::ofstream::out);
    if (!out.is_open()) {
        std::cerr << "Could not open file " << file << " for writing JSON output." << std::endl;
    }
    out << arr << std::endl;
    out.close();
}

} // namespace analysis
} // namespace sdfg
