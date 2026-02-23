#include "sdfg/analysis/loop_analysis.h"
#include <unordered_set>
#include <vector>

#include "sdfg/analysis/assumptions_analysis.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/symbolic/conjunctive_normal_form.h"
#include "sdfg/symbolic/series.h"

namespace sdfg {
namespace analysis {

LoopAnalysis::LoopAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg), loops_(), loop_tree_(DFSLoopComparator(&loops_)) {}

void LoopAnalysis::
    run(structured_control_flow::ControlFlowNode& scope, structured_control_flow::ControlFlowNode* parent_loop) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&scope};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        // Loop detected
        if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->loops_.push_back(while_stmt);
            this->loop_tree_[while_stmt] = parent_loop;
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->loops_.push_back(loop_stmt);
            this->loop_tree_[loop_stmt] = parent_loop;
        }

        if (dynamic_cast<structured_control_flow::Block*>(current)) {
            continue;
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            this->run(while_stmt->root(), while_stmt);
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            this->run(for_stmt->root(), for_stmt);
        } else if (dynamic_cast<structured_control_flow::Break*>(current)) {
            continue;
        } else if (dynamic_cast<structured_control_flow::Continue*>(current)) {
            continue;
        } else if (dynamic_cast<structured_control_flow::Return*>(current)) {
            continue;
        } else {
            throw std::runtime_error("Unsupported control flow node type");
        }
    }
}

void LoopAnalysis::run(AnalysisManager& analysis_manager) {
    this->loops_.clear();
    this->loop_tree_.clear();
    this->loop_infos_.clear();
    this->run(this->sdfg_.root(), nullptr);

    // Loop info for outermost loops
    int loopnest_index = 0;
    for (const auto& [loop, parent] : this->loop_tree_) {
        if (parent != nullptr) {
            continue;
        }

        LoopInfo info;
        info.loopnest_index = loopnest_index;
        loopnest_index++;

        info.element_id = loop->element_id();

        auto descendants = this->descendants(loop);
        descendants.insert(loop);

        // Structure of loop nest
        info.num_loops = descendants.size();
        info.max_depth = 0;
        for (const auto& path : this->loop_tree_paths(loop)) {
            info.max_depth = std::max(info.max_depth, path.size());
        }

        info.is_perfectly_nested = true;
        auto current = loop;
        while (true) {
            auto children = this->children(current);
            if (children.empty()) {
                break;
            }

            if (children.size() > 1) {
                info.is_perfectly_nested = false;
                break;
            }

            auto child = children[0];
            structured_control_flow::Sequence* root = nullptr;

            if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
                root = &while_stmt->root();
            } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
                root = &loop_stmt->root();
            }

            if (root == nullptr || root->size() != 1 || &root->at(0).first != child) {
                info.is_perfectly_nested = false;
                break;
            }

            current = child;
        }

        // Count types of loops
        info.num_maps = 0;
        info.num_fors = 0;
        info.num_whiles = 0;
        for (auto node : descendants) {
            if (dynamic_cast<structured_control_flow::Map*>(node)) {
                info.num_maps++;
            } else if (dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                info.num_fors++;
            } else if (dynamic_cast<structured_control_flow::While*>(node)) {
                info.num_whiles++;
            }
        }
        info.is_perfectly_parallel = (info.num_loops == info.num_maps);

        // Classifiy loop nest
        info.is_elementwise = false;
        if (info.is_perfectly_nested && info.is_perfectly_parallel) {
            bool all_contiguous = true;
            for (auto node : descendants) {
                if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(node)) {
                    bool is_contiguous =
                        symbolic::series::is_contiguous(loop_stmt->update(), loop_stmt->indvar(), symbolic::Assumptions());
                    if (!is_contiguous) {
                        all_contiguous = false;
                        break;
                    }
                } else {
                    all_contiguous = false;
                    break;
                }
            }
            info.is_elementwise = all_contiguous;
        }

        // Criterion: Loop must not have side-effecting body
        structured_control_flow::Sequence* root = nullptr;
        if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(loop)) {
            root = &while_stmt->root();
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::For*>(loop)) {
            root = &loop_stmt->root();
        }
        // Maps cannot have side effects by definition

        info.has_side_effects = false;
        if (root != nullptr) {
            std::list<const structured_control_flow::ControlFlowNode*> queue = {root};
            while (!queue.empty()) {
                auto current = queue.front();
                queue.pop_front();

                if (auto block = dynamic_cast<const structured_control_flow::Block*>(current)) {
                    for (auto& node : block->dataflow().nodes()) {
                        if (auto library_node = dynamic_cast<const data_flow::LibraryNode*>(&node)) {
                            if (library_node->side_effect()) {
                                info.has_side_effects = true;
                                break;
                            }
                        }
                    }
                } else if (auto seq = dynamic_cast<const structured_control_flow::Sequence*>(current)) {
                    for (size_t i = 0; i < seq->size(); i++) {
                        auto& child = seq->at(i).first;
                        queue.push_back(&child);
                    }
                } else if (auto ifelse = dynamic_cast<const structured_control_flow::IfElse*>(current)) {
                    for (size_t i = 0; i < ifelse->size(); i++) {
                        auto& branch = ifelse->at(i).first;
                        queue.push_back(&branch);
                    }
                } else if (auto loop = dynamic_cast<const structured_control_flow::For*>(current)) {
                    queue.push_back(&loop->root());
                } else if (auto while_stmt = dynamic_cast<const structured_control_flow::While*>(current)) {
                    queue.push_back(&while_stmt->root());
                } else if (auto loop = dynamic_cast<const structured_control_flow::Map*>(current)) {
                    continue;
                } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Break*>(current)) {
                    continue;
                } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Continue*>(current)) {
                    continue;
                } else if (auto for_stmt = dynamic_cast<const structured_control_flow::Return*>(current)) {
                    info.has_side_effects = true;
                    break;
                } else {
                    throw InvalidSDFGException("Unknown control flow node type in Loop Analysis.");
                }
            }
        }

        this->loop_infos_[loop] = info;
    }
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopAnalysis::loops() const { return this->loops_; }

LoopInfo LoopAnalysis::loop_info(structured_control_flow::ControlFlowNode* loop) const {
    if (this->loop_infos_.find(loop) == this->loop_infos_.end()) {
        return LoopInfo();
    }
    return this->loop_infos_.at(loop);
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

bool LoopAnalysis::is_monotonic(structured_control_flow::StructuredLoop* loop, AssumptionsAnalysis& assumptions_analysis) {
    auto assums = assumptions_analysis.get(*loop, true);

    return symbolic::series::is_monotonic(loop->update(), loop->indvar(), assums);
}

bool LoopAnalysis::is_monotonic(structured_control_flow::StructuredLoop* loop, const symbolic::Assumptions& assumptions) {
    return symbolic::series::is_monotonic(loop->update(), loop->indvar(), assumptions);
}

bool LoopAnalysis::is_contiguous(structured_control_flow::StructuredLoop* loop, AssumptionsAnalysis& assumptions_analysis) {
    auto assums = assumptions_analysis.get(*loop, true);

    return symbolic::series::is_contiguous(loop->update(), loop->indvar(), assums);
}

bool LoopAnalysis::is_contiguous(structured_control_flow::StructuredLoop* loop, const symbolic::Assumptions& assumptions) {
    return symbolic::series::is_contiguous(loop->update(), loop->indvar(), assumptions);
}

symbolic::Expression LoopAnalysis::
    canonical_bound(structured_control_flow::StructuredLoop* loop, AssumptionsAnalysis& assumptions_analysis) {
    auto assums = assumptions_analysis.get(*loop, true);
    return LoopAnalysis::canonical_bound(loop, assums);
}

symbolic::Expression LoopAnalysis::
    canonical_bound(structured_control_flow::StructuredLoop* loop, const symbolic::Assumptions& assumptions) {
    if (!LoopAnalysis::is_monotonic(loop, assumptions)) {
        return SymEngine::null;
    }

    symbolic::CNF cnf;
    try {
        cnf = symbolic::conjunctive_normal_form(loop->condition());
    } catch (const std::runtime_error& e) {
        return SymEngine::null;
    }

    bool has_complex_clauses = false;
    for (auto& clause : cnf) {
        if (clause.size() > 1) {
            has_complex_clauses = true;
            break;
        }
    }
    if (has_complex_clauses) {
        return SymEngine::null;
    }

    auto indvar = loop->indvar();
    symbolic::Expression bound = SymEngine::null;
    for (auto& clause : cnf) {
        for (auto& literal : clause) {
            if (SymEngine::is_a<SymEngine::StrictLessThan>(*literal)) {
                auto lt = SymEngine::rcp_dynamic_cast<const SymEngine::StrictLessThan>(literal);
                auto lhs = lt->get_args()[0];
                auto rhs = lt->get_args()[1];
                if (SymEngine::eq(*lhs, *indvar)) {
                    if (bound == SymEngine::null) {
                        bound = rhs;
                    } else {
                        bound = symbolic::min(bound, rhs);
                    }
                } else {
                    return SymEngine::null;
                }
            } else if (SymEngine::is_a<SymEngine::LessThan>(*literal)) {
                auto le = SymEngine::rcp_dynamic_cast<const SymEngine::LessThan>(literal);
                auto lhs = le->get_args()[0];
                auto rhs = le->get_args()[1];
                if (SymEngine::eq(*lhs, *indvar)) {
                    if (bound == SymEngine::null) {
                        bound = symbolic::add(rhs, symbolic::one());
                    } else {
                        bound = symbolic::min(bound, symbolic::add(rhs, symbolic::one()));
                    }
                } else {
                    return SymEngine::null;
                }
            } else {
                return SymEngine::null;
            }
        }
    }

    return bound;
}

symbolic::Integer LoopAnalysis::stride(structured_control_flow::StructuredLoop* loop) {
    auto expr = loop->update();
    auto indvar = loop->indvar();

    if (SymEngine::is_a<SymEngine::Add>(*expr)) {
        auto add_expr = SymEngine::rcp_static_cast<const SymEngine::Add>(expr);
        if (add_expr->get_args().size() != 2) {
            return SymEngine::null;
        }
        auto arg1 = add_expr->get_args()[0];
        auto arg2 = add_expr->get_args()[1];
        if (symbolic::eq(arg1, indvar)) {
            if (SymEngine::is_a<SymEngine::Integer>(*arg2)) {
                return SymEngine::rcp_static_cast<const SymEngine::Integer>(arg2);
            }
        }
        if (symbolic::eq(arg2, indvar)) {
            if (SymEngine::is_a<SymEngine::Integer>(*arg1)) {
                return SymEngine::rcp_static_cast<const SymEngine::Integer>(arg1);
            }
        }
    }
    return SymEngine::null;
}

const std::map<structured_control_flow::ControlFlowNode*, structured_control_flow::ControlFlowNode*, DFSLoopComparator>&
LoopAnalysis::loop_tree() const {
    return this->loop_tree_;
}

structured_control_flow::ControlFlowNode* LoopAnalysis::parent_loop(structured_control_flow::ControlFlowNode* loop
) const {
    return this->loop_tree_.at(loop);
}

const std::vector<structured_control_flow::ControlFlowNode*> LoopAnalysis::outermost_loops() const {
    std::vector<structured_control_flow::ControlFlowNode*> outermost_loops_;
    for (const auto& [loop, parent] : this->loop_tree_) {
        if (parent == nullptr) {
            outermost_loops_.push_back(loop);
        }
    }
    return outermost_loops_;
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

std::vector<sdfg::structured_control_flow::ControlFlowNode*> LoopAnalysis::
    children(sdfg::structured_control_flow::ControlFlowNode* node) const {
    // Find unique child
    return this->children(node, this->loop_tree_);
};

std::vector<sdfg::structured_control_flow::ControlFlowNode*> LoopAnalysis::children(
    sdfg::structured_control_flow::ControlFlowNode* node,
    const std::map<
        sdfg::structured_control_flow::ControlFlowNode*,
        sdfg::structured_control_flow::ControlFlowNode*,
        DFSLoopComparator>& tree
) const {
    // Find unique child
    std::vector<sdfg::structured_control_flow::ControlFlowNode*> c;
    for (auto& entry : tree) {
        if (entry.second == node) {
            c.push_back(entry.first);
        }
    }
    return c;
};

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
    auto children = this->children(loop, tree);
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
        auto children = this->children(current, this->loop_tree_);
        for (auto& child : children) {
            if (desc.find(child) == desc.end()) {
                desc.insert(child);
                queue.push_back(child);
            }
        }
    }
    return desc;
}

} // namespace analysis
} // namespace sdfg
