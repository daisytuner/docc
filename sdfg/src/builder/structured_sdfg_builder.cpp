#include "sdfg/builder/structured_sdfg_builder.h"

#include <cstddef>

#include "sdfg/data_flow/library_node.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/types/utils.h"

#define TRAVERSE_CUTOFF 30

using namespace sdfg::control_flow;
using namespace sdfg::structured_control_flow;

namespace sdfg {
namespace builder {

std::unordered_set<const control_flow::State*> StructuredSDFGBuilder::
    determine_loop_nodes(SDFG& sdfg, const control_flow::State& start, const control_flow::State& end) const {
    std::unordered_set<const control_flow::State*> nodes;
    std::unordered_set<const control_flow::State*> visited;
    std::list<const control_flow::State*> queue = {&start};
    while (!queue.empty()) {
        auto curr = queue.front();
        queue.pop_front();
        if (visited.find(curr) != visited.end()) {
            continue;
        }
        visited.insert(curr);

        nodes.insert(curr);
        if (curr == &end) {
            continue;
        }

        for (auto& iedge : sdfg.in_edges(*curr)) {
            queue.push_back(&iedge.src());
        }
    }

    // Iteratively expand nodes to reduce frontier size
    auto dom_tree = sdfg.dominator_tree();
    auto dominates = [&](const control_flow::State* a, const control_flow::State* b) {
        const control_flow::State* curr = b;
        while (curr != nullptr) {
            if (curr == a) return true;
            if (dom_tree.find(curr) == dom_tree.end()) break;
            curr = dom_tree.at(curr);
        }
        return false;
    };

    // Identify header exits
    std::unordered_set<const control_flow::State*> stop_nodes;
    for (auto& edge : sdfg.out_edges(end)) {
        if (nodes.find(&edge.dst()) == nodes.end()) {
            stop_nodes.insert(&edge.dst());
        }
    }

    // If no header exits, check latch exits (Do-While)
    if (stop_nodes.empty()) {
        for (auto& edge : sdfg.out_edges(start)) {
            if (nodes.find(&edge.dst()) == nodes.end()) {
                stop_nodes.insert(&edge.dst());
            }
        }
    }

    // If still no exits, check any natural loop exit (e.g. infinite loop with break)
    if (stop_nodes.empty()) {
        for (auto node : nodes) {
            for (auto& edge : sdfg.out_edges(*node)) {
                if (nodes.find(&edge.dst()) == nodes.end()) {
                    stop_nodes.insert(&edge.dst());
                }
            }
        }
    }

    while (true) {
        std::unordered_set<const control_flow::State*> frontier;
        for (auto node : nodes) {
            for (auto& edge : sdfg.out_edges(*node)) {
                if (nodes.find(&edge.dst()) == nodes.end()) {
                    frontier.insert(&edge.dst());
                }
            }
        }

        bool changed = false;
        for (auto f : frontier) {
            // If f is a stop node, do not include it
            if (stop_nodes.find(f) != stop_nodes.end()) {
                continue;
            }
            // If f is dominated by the header, it belongs to the loop body (extended)
            if (dominates(&end, f)) {
                nodes.insert(f);
                changed = true;
            }
        }

        if (!changed) break;
    }

    return nodes;
};

bool post_dominates(
    const State* pdom,
    const State* node,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree
) {
    if (pdom == node) {
        return true;
    }

    auto current = pdom_tree.at(node);
    while (current != nullptr) {
        if (current == pdom) {
            return true;
        }
        current = pdom_tree.at(current);
    }

    return false;
}


void StructuredSDFGBuilder::traverse(SDFG& sdfg) {
    // Start of SDFGS
    Sequence& root = *structured_sdfg_->root_;
    const State* start_state = &sdfg.start_state();

    auto pdom_tree = sdfg.post_dominator_tree();

    std::unordered_set<const InterstateEdge*> breaks;
    std::unordered_set<const InterstateEdge*> continues;
    for (auto& edge : sdfg.back_edges()) {
        continues.insert(edge);
    }

    this->current_traverse_loop_ = nullptr;
    std::unordered_set<const control_flow::State*> visited;
    this->structure_region(sdfg, root, start_state, nullptr, continues, breaks, pdom_tree, visited);
};

void StructuredSDFGBuilder::structure_region(
    SDFG& sdfg,
    Sequence& scope,
    const State* entry,
    const State* exit,
    const std::unordered_set<const InterstateEdge*>& continues,
    const std::unordered_set<const InterstateEdge*>& breaks,
    const std::unordered_map<const control_flow::State*, const control_flow::State*>& pdom_tree,
    std::unordered_set<const control_flow::State*>& visited,
    bool is_loop_body
) {
    const State* current = entry;
    while (current != exit) {
        if (current == nullptr) {
            break;
        }


        // Cutoff
        if (this->function().element_counter_ > sdfg.states().size() * TRAVERSE_CUTOFF) {
            throw UnstructuredControlFlowException();
        }

        if (visited.find(current) != visited.end()) {
            throw UnstructuredControlFlowException();
        }
        visited.insert(current);

        // Loop detection
        bool is_loop_header = false;
        if (!is_loop_body || current != entry) {
            for (auto& iedge : sdfg.in_edges(*current)) {
                if (continues.find(&iedge) != continues.end()) {
                    is_loop_header = true;
                    break;
                }
            }
        }

        if (is_loop_header) {
            // 1. Determine nodes of loop body
            std::unordered_set<const InterstateEdge*> loop_edges;
            for (auto& iedge : sdfg.in_edges(*current)) {
                if (continues.find(&iedge) != continues.end()) {
                    loop_edges.insert(&iedge);
                }
            }

            std::unordered_set<const control_flow::State*> body;
            for (auto back_edge : loop_edges) {
                auto loop_nodes = this->determine_loop_nodes(sdfg, back_edge->src(), back_edge->dst());
                body.insert(loop_nodes.begin(), loop_nodes.end());
            }

            // 2. Determine exit states and exit edges
            std::unordered_set<const control_flow::State*> exit_states;
            std::unordered_set<const control_flow::InterstateEdge*> exit_edges;
            for (auto node : body) {
                for (auto& edge : sdfg.out_edges(*node)) {
                    if (body.find(&edge.dst()) == body.end()) {
                        if (continues.find(&edge) != continues.end()) {
                            continue;
                        }
                        exit_edges.insert(&edge);
                        exit_states.insert(&edge.dst());
                    }
                }
            }

            if (exit_states.size() > 1) {
                std::unordered_set<const control_flow::State*> non_return_exits;
                for (auto s : exit_states) {
                    if (dynamic_cast<const control_flow::ReturnState*>(s)) {
                        continue;
                    }
                    if (sdfg.out_degree(*s) > 0) {
                        non_return_exits.insert(s);
                    }
                }
                if (non_return_exits.size() == 1) {
                    exit_states = non_return_exits;
                }
            }

            if (exit_states.size() != 1) {
                throw UnstructuredControlFlowException();
            }
            const control_flow::State* exit_state = *exit_states.begin();

            for (auto& edge : breaks) {
                exit_edges.insert(edge);
            }

            // Collect debug information
            DebugInfo dbg_info = current->debug_info();
            for (auto& edge : sdfg.in_edges(*current)) {
                dbg_info = DebugInfo::merge(dbg_info, edge.debug_info());
            }
            for (auto node : body) {
                dbg_info = DebugInfo::merge(dbg_info, node->debug_info());
            }
            for (auto edge : exit_edges) {
                dbg_info = DebugInfo::merge(dbg_info, edge->debug_info());
            }

            // 3. Add while loop
            While& loop = this->add_while(scope, {}, dbg_info);
            auto last_loop_ = this->current_traverse_loop_;
            this->current_traverse_loop_ = &loop;

            std::unordered_set<const control_flow::State*> loop_visited(visited);
            loop_visited.erase(current);

            this->structure_region(
                sdfg, loop.root(), current, exit_state, continues, exit_edges, pdom_tree, loop_visited, true
            );
            this->current_traverse_loop_ = last_loop_;

            current = exit_state;
            continue;
        }

        auto out_edges = sdfg.out_edges(*current);
        auto out_degree = sdfg.out_degree(*current);

        // Case 1: Sink node
        if (out_degree == 0) {
            if (!std::ranges::empty(current->dataflow().nodes())) {
                this->add_block(scope, current->dataflow(), {}, current->debug_info());
            }

            auto return_state = dynamic_cast<const control_flow::ReturnState*>(current);
            assert(return_state != nullptr);
            if (return_state->is_data()) {
                this->add_return(scope, return_state->data(), {}, return_state->debug_info());
            } else if (return_state->is_constant()) {
                this->add_constant_return(
                    scope, return_state->data(), return_state->type(), {}, return_state->debug_info()
                );
            } else {
                assert(false && "Unknown return state type");
            }

            break;
        }

        // Case 2: Transition
        if (out_degree == 1) {
            auto& oedge = *out_edges.begin();
            if (!oedge.is_unconditional()) {
                throw UnstructuredControlFlowException();
            }

            if (!std::ranges::empty(current->dataflow().nodes()) || !oedge.assignments().empty()) {
                this->add_block(scope, current->dataflow(), oedge.assignments(), current->debug_info());
            }

            if (continues.find(&oedge) != continues.end()) {
                if (this->current_traverse_loop_ == nullptr) {
                    throw UnstructuredControlFlowException();
                }
                this->add_continue(scope, {}, oedge.debug_info());
                break;
            } else if (breaks.find(&oedge) != breaks.end()) {
                if (this->current_traverse_loop_ == nullptr) {
                    throw UnstructuredControlFlowException();
                }
                this->add_break(scope, {}, oedge.debug_info());
                break;
            } else {
                current = &oedge.dst();
            }
            continue;
        }

        // Case 3: Branches
        if (out_degree > 1) {
            if (!std::ranges::empty(current->dataflow().nodes())) {
                this->add_block(scope, current->dataflow(), {}, current->debug_info());
            }

            // Determine Merge Point
            const State* merge = nullptr;
            if (pdom_tree.find(current) != pdom_tree.end()) {
                merge = pdom_tree.at(current);
            }


            // If merge is beyond exit, clamp to exit
            if (exit != nullptr && merge != nullptr) {
                if (post_dominates(merge, exit, pdom_tree)) {
                    merge = exit;
                }
            }

            if (merge != nullptr && visited.find(merge) != visited.end()) {
                merge = exit;
            }

            if (merge == nullptr && exit != nullptr) {
                merge = exit;
            }

            auto& if_else = this->add_if_else(scope, {}, current->debug_info());
            for (auto& out_edge : out_edges) {
                auto& branch = this->add_case(if_else, out_edge.condition(), out_edge.debug_info());
                if (!out_edge.assignments().empty()) {
                    this->add_block(branch, out_edge.assignments(), out_edge.debug_info());
                }
                if (continues.find(&out_edge) != continues.end()) {
                    if (this->current_traverse_loop_ == nullptr) {
                        throw UnstructuredControlFlowException();
                    }
                    this->add_continue(branch, {}, out_edge.debug_info());
                } else if (breaks.find(&out_edge) != breaks.end()) {
                    if (this->current_traverse_loop_ == nullptr) {
                        throw UnstructuredControlFlowException();
                    }
                    this->add_break(branch, {}, out_edge.debug_info());
                } else {
                    std::unordered_set<const control_flow::State*> branch_visited(visited);
                    this->structure_region(
                        sdfg, branch, &out_edge.dst(), merge, continues, breaks, pdom_tree, branch_visited
                    );
                }
            }

            current = merge;
            continue;
        }
    }
}

Function& StructuredSDFGBuilder::function() const { return static_cast<Function&>(*this->structured_sdfg_); };

StructuredSDFGBuilder::StructuredSDFGBuilder(StructuredSDFG& sdfg)
    : FunctionBuilder(), structured_sdfg_(&sdfg, owned(false)) {};

StructuredSDFGBuilder::StructuredSDFGBuilder(std::unique_ptr<StructuredSDFG>& sdfg)
    : FunctionBuilder(), structured_sdfg_(sdfg.release(), owned(true)) {};

StructuredSDFGBuilder::StructuredSDFGBuilder(const std::string& name, FunctionType type)
    : FunctionBuilder(), structured_sdfg_(new StructuredSDFG(name, type), owned(true)) {};

StructuredSDFGBuilder::StructuredSDFGBuilder(const std::string& name, FunctionType type, const types::IType& return_type)
    : FunctionBuilder(), structured_sdfg_(new StructuredSDFG(name, type, return_type), owned(true)) {};

StructuredSDFGBuilder::StructuredSDFGBuilder(SDFG& sdfg)
    : FunctionBuilder(),
      structured_sdfg_(new StructuredSDFG(sdfg.name(), sdfg.type(), sdfg.return_type()), owned(true)) {
    for (auto& entry : sdfg.structures_) {
        this->structured_sdfg_->structures_.insert({entry.first, entry.second->clone()});
    }

    for (auto& entry : sdfg.containers_) {
        this->structured_sdfg_->containers_.insert({entry.first, entry.second->clone()});
    }

    for (auto& arg : sdfg.arguments_) {
        this->structured_sdfg_->arguments_.push_back(arg);
    }

    for (auto& ext : sdfg.externals_) {
        this->structured_sdfg_->externals_.push_back(ext);
        this->structured_sdfg_->externals_linkage_types_[ext] = sdfg.linkage_type(ext);
    }

    for (auto& entry : sdfg.assumptions_) {
        this->structured_sdfg_->assumptions_.insert({entry.first, entry.second});
    }

    for (auto& entry : sdfg.metadata_) {
        this->structured_sdfg_->metadata_[entry.first] = entry.second;
    }

    this->traverse(sdfg);
};

StructuredSDFG& StructuredSDFGBuilder::subject() const { return *this->structured_sdfg_; };

std::unique_ptr<StructuredSDFG> StructuredSDFGBuilder::move() {
#ifndef NDEBUG
    this->structured_sdfg_->validate();
#endif

    if (!structured_sdfg_.get_deleter().should_delete_) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Cannot move a non-owned SDFG");
    }

    return std::move(std::unique_ptr<StructuredSDFG>(structured_sdfg_.release()));
};

void StructuredSDFGBuilder::rename_container(const std::string& old_name, const std::string& new_name) const {
    FunctionBuilder::rename_container(old_name, new_name);

    this->structured_sdfg_->root_->replace(symbolic::symbol(old_name), symbolic::symbol(new_name));
};

Element* StructuredSDFGBuilder::find_element_by_id(const size_t& element_id) const {
    auto& sdfg = this->subject();
    std::list<Element*> queue = {&sdfg.root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (current->element_id() == element_id) {
            return current;
        }

        if (auto block_stmt = dynamic_cast<structured_control_flow::Block*>(current)) {
            auto& dataflow = block_stmt->dataflow();
            for (auto& node : dataflow.nodes()) {
                queue.push_back(&node);
            }
            for (auto& edge : dataflow.edges()) {
                queue.push_back(&edge);
            }
        } else if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                queue.push_back(&sequence_stmt->at(i).first);
                queue.push_back(&sequence_stmt->at(i).second);
            }
        } else if (dynamic_cast<structured_control_flow::Return*>(current)) {
            // Do nothing
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto for_stmt = dynamic_cast<structured_control_flow::For*>(current)) {
            queue.push_back(&for_stmt->root());
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&while_stmt->root());
        } else if (dynamic_cast<structured_control_flow::Continue*>(current)) {
            // Do nothing
        } else if (dynamic_cast<structured_control_flow::Break*>(current)) {
            // Do nothing
        } else if (auto map_stmt = dynamic_cast<structured_control_flow::Map*>(current)) {
            queue.push_back(&map_stmt->root());
        }
    }

    return nullptr;
};

Sequence& StructuredSDFGBuilder::
    add_sequence(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Sequence&>(*parent.children_.back().get());
};

Sequence& StructuredSDFGBuilder::add_sequence_before(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index, std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Sequence&>(*parent.children_.at(index).get());
};

Sequence& StructuredSDFGBuilder::add_sequence_after(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1,
        std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Sequence&>(*parent.children_.at(index + 1).get());
};

std::pair<Sequence&, Transition&> StructuredSDFGBuilder::
    add_sequence_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index, std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Sequence&>(new_entry.first);

    return {new_block, new_entry.second};
};

void StructuredSDFGBuilder::remove_child(Sequence& parent, size_t index) {
    parent.children_.erase(parent.children_.begin() + index);
    parent.transitions_.erase(parent.transitions_.begin() + index);
};

void StructuredSDFGBuilder::remove_children(Sequence& parent) {
    parent.children_.clear();
    parent.transitions_.clear();
};

void StructuredSDFGBuilder::move_child(Sequence& source, size_t source_index, Sequence& target) {
    size_t target_index = target.size();
    if (&source == &target) {
        target_index--;
    }
    this->move_child(source, source_index, target, target_index);
};

void StructuredSDFGBuilder::move_child(Sequence& source, size_t source_index, Sequence& target, size_t target_index) {
    auto node_ptr = std::move(source.children_.at(source_index));
    auto trans_ptr = std::move(source.transitions_.at(source_index));
    source.children_.erase(source.children_.begin() + source_index);
    source.transitions_.erase(source.transitions_.begin() + source_index);

    trans_ptr->parent_ = &target;
    target.children_.insert(target.children_.begin() + target_index, std::move(node_ptr));
    target.transitions_.insert(target.transitions_.begin() + target_index, std::move(trans_ptr));
};

void StructuredSDFGBuilder::move_children(Sequence& source, Sequence& target) {
    this->move_children(source, target, target.size());
};

void StructuredSDFGBuilder::move_children(Sequence& source, Sequence& target, size_t target_index) {
    target.children_.insert(
        target.children_.begin() + target_index,
        std::make_move_iterator(source.children_.begin()),
        std::make_move_iterator(source.children_.end())
    );
    target.transitions_.insert(
        target.transitions_.begin() + target_index,
        std::make_move_iterator(source.transitions_.begin()),
        std::make_move_iterator(source.transitions_.end())
    );
    for (auto& trans : target.transitions_) {
        trans->parent_ = &target;
    }
    source.children_.clear();
    source.transitions_.clear();
};

Sequence& StructuredSDFGBuilder::hoist_root() {
    auto current_root = std::move(this->structured_sdfg_->root_);

    this->structured_sdfg_->root_ =
        std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), current_root->debug_info()));

    this->structured_sdfg_->root_->children_.push_back(std::move(current_root));
    this->structured_sdfg_->root_->transitions_.push_back(std::unique_ptr<Transition>(
        new Transition(this->new_element_id(), current_root->debug_info(), *this->structured_sdfg_->root_)
    ));
    return *this->structured_sdfg_->root_;
};

Block& StructuredSDFGBuilder::
    add_block(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.back());
    (*new_block.dataflow_).parent_ = &new_block;

    return new_block;
};

Block& StructuredSDFGBuilder::add_block(
    Sequence& parent,
    const data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_.push_back(std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.back());
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_before(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index));
    (*new_block.dataflow_).parent_ = &new_block;

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_before(
    Sequence& parent,
    ControlFlowNode& child,
    data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index));
    (*new_block.dataflow_).parent_ = &new_block;
    this->add_dataflow(data_flow_graph, new_block);

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_after(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index + 1));
    (*new_block.dataflow_).parent_ = &new_block;

    return new_block;
};

Block& StructuredSDFGBuilder::add_block_after(
    Sequence& parent,
    ControlFlowNode& child,
    data_flow::DataFlowGraph& data_flow_graph,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    auto& new_block = static_cast<structured_control_flow::Block&>(*parent.children_.at(index + 1));
    (*new_block.dataflow_).parent_ = &new_block;
    this->add_dataflow(data_flow_graph, new_block);

    return new_block;
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::
    add_block_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = static_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_before(
    Sequence& parent, ControlFlowNode& child, data_flow::DataFlowGraph& data_flow_graph, const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::
    add_block_after(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    return {new_block, new_entry.second};
};

std::pair<Block&, Transition&> StructuredSDFGBuilder::add_block_after(
    Sequence& parent, ControlFlowNode& child, data_flow::DataFlowGraph& data_flow_graph, const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<Block>(new Block(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index + 1);
    auto& new_block = dynamic_cast<structured_control_flow::Block&>(new_entry.first);
    (*new_block.dataflow_).parent_ = &new_block;

    this->add_dataflow(data_flow_graph, new_block);

    return {new_block, new_entry.second};
};

IfElse& StructuredSDFGBuilder::
    add_if_else(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<IfElse&>(*parent.children_.back().get());
};

IfElse& StructuredSDFGBuilder::add_if_else_before(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<IfElse&>(*parent.children_.at(index));
};

IfElse& StructuredSDFGBuilder::add_if_else_after(
    Sequence& parent,
    ControlFlowNode& child,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1, std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info))
    );

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<IfElse&>(*parent.children_.at(index + 1));
};

std::pair<IfElse&, Transition&> StructuredSDFGBuilder::
    add_if_else_before(Sequence& parent, ControlFlowNode& child, const DebugInfo& debug_info) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_
        .insert(parent.children_.begin() + index, std::unique_ptr<IfElse>(new IfElse(this->new_element_id(), debug_info)));

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent))
    );

    auto new_entry = parent.at(index);
    auto& new_block = dynamic_cast<structured_control_flow::IfElse&>(new_entry.first);

    return {new_block, new_entry.second};
};

Sequence& StructuredSDFGBuilder::add_case(IfElse& scope, const sdfg::symbolic::Condition cond, const DebugInfo& debug_info) {
    scope.cases_.push_back(std::unique_ptr<Sequence>(new Sequence(this->new_element_id(), debug_info)));

    scope.conditions_.push_back(cond);
    return *scope.cases_.back();
};

void StructuredSDFGBuilder::remove_case(IfElse& scope, size_t index, const DebugInfo& debug_info) {
    scope.cases_.erase(scope.cases_.begin() + index);
    scope.conditions_.erase(scope.conditions_.begin() + index);
};

While& StructuredSDFGBuilder::
    add_while(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<While>(new While(this->new_element_id(), debug_info)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<While&>(*parent.children_.back().get());
};

For& StructuredSDFGBuilder::add_for(
    Sequence& parent,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_
        .push_back(std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar, init, update, condition)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<For&>(*parent.children_.back().get());
};

For& StructuredSDFGBuilder::add_for_before(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar, init, update, condition))
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<For&>(*parent.children_.at(index).get());
};

For& StructuredSDFGBuilder::add_for_after(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1,
        std::unique_ptr<For>(new For(this->new_element_id(), debug_info, indvar, init, update, condition))
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<For&>(*parent.children_.at(index + 1).get());
};

Map& StructuredSDFGBuilder::add_map(
    Sequence& parent,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update,
    const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_
        .push_back(std::unique_ptr<
                   Map>(new Map(this->new_element_id(), debug_info, indvar, init, update, condition, schedule_type)));

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Map&>(*parent.children_.back().get());
};

Map& StructuredSDFGBuilder::add_map_before(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update,
    const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index,
        std::unique_ptr<Map>(new Map(this->new_element_id(), debug_info, indvar, init, update, condition, schedule_type)
        )
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Map&>(*parent.children_.at(index).get());
};

Map& StructuredSDFGBuilder::add_map_after(
    Sequence& parent,
    ControlFlowNode& child,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update,
    const ScheduleType& schedule_type,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    int index = parent.index(child);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    parent.children_.insert(
        parent.children_.begin() + index + 1,
        std::unique_ptr<Map>(new Map(this->new_element_id(), debug_info, indvar, init, update, condition, schedule_type)
        )
    );

    // Increment element id for body node
    this->new_element_id();

    parent.transitions_.insert(
        parent.transitions_.begin() + index + 1,
        std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
    );

    return static_cast<Map&>(*parent.children_.at(index + 1).get());
};

Continue& StructuredSDFGBuilder::
    add_continue(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Continue>(new Continue(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Continue&>(*parent.children_.back().get());
};

Break& StructuredSDFGBuilder::
    add_break(Sequence& parent, const sdfg::control_flow::Assignments& assignments, const DebugInfo& debug_info) {
    parent.children_.push_back(std::unique_ptr<Break>(new Break(this->new_element_id(), debug_info)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Break&>(*parent.children_.back().get());
};

Return& StructuredSDFGBuilder::add_return(
    Sequence& parent,
    const std::string& data,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_.push_back(std::unique_ptr<Return>(new Return(this->new_element_id(), debug_info, data)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Return&>(*parent.children_.back().get());
};

Return& StructuredSDFGBuilder::add_constant_return(
    Sequence& parent,
    const std::string& data,
    const types::IType& type,
    const sdfg::control_flow::Assignments& assignments,
    const DebugInfo& debug_info
) {
    parent.children_.push_back(std::unique_ptr<Return>(new Return(this->new_element_id(), debug_info, data, type)));

    parent.transitions_
        .push_back(std::unique_ptr<Transition>(new Transition(this->new_element_id(), debug_info, parent, assignments))
        );

    return static_cast<Return&>(*parent.children_.back().get());
};

For& StructuredSDFGBuilder::convert_while(
    Sequence& parent,
    While& loop,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update
) {
    int index = parent.index(loop);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    auto iter = parent.children_.begin() + index;
    auto& new_iter = *parent.children_.insert(
        iter + 1,
        std::unique_ptr<For>(new For(this->new_element_id(), loop.debug_info(), indvar, init, update, condition))
    );

    // Increment element id for body node
    this->new_element_id();

    auto& for_loop = dynamic_cast<For&>(*new_iter);
    this->move_children(loop.root(), for_loop.root());

    // Remove while loop
    parent.children_.erase(parent.children_.begin() + index);

    return for_loop;
};

Map& StructuredSDFGBuilder::convert_for(Sequence& parent, For& loop) {
    int index = parent.index(loop);
    if (index == -1) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Child not found");
    }

    auto iter = parent.children_.begin() + index;
    auto& new_iter = *parent.children_.insert(
        iter + 1,
        std::unique_ptr<Map>(new Map(
            this->new_element_id(),
            loop.debug_info(),
            loop.indvar(),
            loop.init(),
            loop.update(),
            loop.condition(),
            ScheduleType_Sequential::create()
        ))
    );

    // Increment element id for body node
    this->new_element_id();

    auto& map = dynamic_cast<Map&>(*new_iter);
    this->move_children(loop.root(), map.root());

    // Remove for loop
    parent.children_.erase(parent.children_.begin() + index);

    return map;
};

void StructuredSDFGBuilder::update_if_else_condition(IfElse& if_else, size_t index, const symbolic::Condition condition) {
    if (index >= if_else.conditions_.size()) {
        throw InvalidSDFGException("StructuredSDFGBuilder: Index out of range");
    }
    if_else.conditions_.at(index) = condition;
};

void StructuredSDFGBuilder::update_loop(
    StructuredLoop& loop,
    const symbolic::Symbol indvar,
    const symbolic::Condition condition,
    const symbolic::Expression init,
    const symbolic::Expression update
) {
    loop.indvar_ = indvar;
    loop.condition_ = condition;
    loop.init_ = init;
    loop.update_ = update;
};

void StructuredSDFGBuilder::update_schedule_type(Map& map, const ScheduleType& schedule_type) {
    map.schedule_type_ = schedule_type;
}

Sequence& StructuredSDFGBuilder::parent(const ControlFlowNode& node) {
    std::list<structured_control_flow::ControlFlowNode*> queue = {&this->structured_sdfg_->root()};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();

        if (auto sequence_stmt = dynamic_cast<structured_control_flow::Sequence*>(current)) {
            for (size_t i = 0; i < sequence_stmt->size(); i++) {
                if (&sequence_stmt->at(i).first == &node) {
                    return *sequence_stmt;
                }
                queue.push_back(&sequence_stmt->at(i).first);
            }
        } else if (auto if_else_stmt = dynamic_cast<structured_control_flow::IfElse*>(current)) {
            for (size_t i = 0; i < if_else_stmt->size(); i++) {
                queue.push_back(&if_else_stmt->at(i).first);
            }
        } else if (auto while_stmt = dynamic_cast<structured_control_flow::While*>(current)) {
            queue.push_back(&while_stmt->root());
        } else if (auto loop_stmt = dynamic_cast<structured_control_flow::StructuredLoop*>(current)) {
            queue.push_back(&loop_stmt->root());
        }
    }

    return this->structured_sdfg_->root();
};

/***** Section: Dataflow Graph *****/

data_flow::AccessNode& StructuredSDFGBuilder::
    add_access(structured_control_flow::Block& block, const std::string& data, const DebugInfo& debug_info) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::AccessNode>(
             new data_flow::AccessNode(this->new_element_id(), debug_info, vertex, block.dataflow(), data)
         )}
    );

    return dynamic_cast<data_flow::AccessNode&>(*(res.first->second));
};

data_flow::ConstantNode& StructuredSDFGBuilder::add_constant(
    structured_control_flow::Block& block, const std::string& data, const types::IType& type, const DebugInfo& debug_info
) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::ConstantNode>(
             new data_flow::ConstantNode(this->new_element_id(), debug_info, vertex, block.dataflow(), data, type)
         )}
    );

    return dynamic_cast<data_flow::ConstantNode&>(*(res.first->second));
};


data_flow::Tasklet& StructuredSDFGBuilder::add_tasklet(
    structured_control_flow::Block& block,
    const data_flow::TaskletCode code,
    const std::string& output,
    const std::vector<std::string>& inputs,
    const DebugInfo& debug_info
) {
    auto vertex = boost::add_vertex(block.dataflow_->graph_);
    auto res = block.dataflow_->nodes_.insert(
        {vertex,
         std::unique_ptr<data_flow::Tasklet>(
             new data_flow::Tasklet(this->new_element_id(), debug_info, vertex, block.dataflow(), code, output, inputs)
         )}
    );

    return dynamic_cast<data_flow::Tasklet&>(*(res.first->second));
};

data_flow::Memlet& StructuredSDFGBuilder::add_memlet(
    structured_control_flow::Block& block,
    data_flow::DataFlowNode& src,
    const std::string& src_conn,
    data_flow::DataFlowNode& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    auto edge = boost::add_edge(src.vertex_, dst.vertex_, block.dataflow_->graph_);
    auto res = block.dataflow_->edges_.insert(
        {edge.first,
         std::unique_ptr<data_flow::Memlet>(new data_flow::Memlet(
             this->new_element_id(), debug_info, edge.first, block.dataflow(), src, src_conn, dst, dst_conn, subset, base_type
         ))}
    );

    auto& memlet = dynamic_cast<data_flow::Memlet&>(*(res.first->second));
#ifndef NDEBUG
    memlet.validate(*this->structured_sdfg_);
#endif

    return memlet;
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::Tasklet& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, "void", dst, dst_conn, subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::Tasklet& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, src_conn, dst, "void", subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::Tasklet& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const DebugInfo& debug_info
) {
    const types::IType* src_type = nullptr;
    if (auto cnode = dynamic_cast<data_flow::ConstantNode*>(&src)) {
        src_type = &cnode->type();
    } else {
        src_type = &this->structured_sdfg_->type(src.data());
    }
    auto base_type = types::infer_type(*this->structured_sdfg_, *src_type, subset);
    if (base_type->type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException("Computational memlet must have a scalar type");
    }
    return this->add_memlet(block, src, "void", dst, dst_conn, subset, *src_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::Tasklet& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const DebugInfo& debug_info
) {
    auto& dst_type = this->structured_sdfg_->type(dst.data());
    auto base_type = types::infer_type(*this->structured_sdfg_, dst_type, subset);
    if (base_type->type_id() != types::TypeID::Scalar) {
        throw InvalidSDFGException("Computational memlet must have a scalar type");
    }
    return this->add_memlet(block, src, src_conn, dst, "void", subset, dst_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::LibraryNode& dst,
    const std::string& dst_conn,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, "void", dst, dst_conn, subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_computational_memlet(
    structured_control_flow::Block& block,
    data_flow::LibraryNode& src,
    const std::string& src_conn,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, src_conn, dst, "void", subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_reference_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::AccessNode& dst,
    const data_flow::Subset& subset,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    return this->add_memlet(block, src, "void", dst, "ref", subset, base_type, debug_info);
};

data_flow::Memlet& StructuredSDFGBuilder::add_dereference_memlet(
    structured_control_flow::Block& block,
    data_flow::AccessNode& src,
    data_flow::AccessNode& dst,
    bool derefs_src,
    const types::IType& base_type,
    const DebugInfo& debug_info
) {
    if (derefs_src) {
        return this->add_memlet(block, src, "void", dst, "deref", {symbolic::zero()}, base_type, debug_info);
    } else {
        return this->add_memlet(block, src, "deref", dst, "void", {symbolic::zero()}, base_type, debug_info);
    }
};

void StructuredSDFGBuilder::remove_memlet(structured_control_flow::Block& block, const data_flow::Memlet& edge) {
    auto& graph = block.dataflow();
    auto e = edge.edge();
    boost::remove_edge(e, graph.graph_);
    graph.edges_.erase(e);
};

void StructuredSDFGBuilder::remove_node(structured_control_flow::Block& block, const data_flow::DataFlowNode& node) {
    auto& graph = block.dataflow();
    auto v = node.vertex();
    boost::remove_vertex(v, graph.graph_);
    graph.nodes_.erase(v);
};

void StructuredSDFGBuilder::clear_node(structured_control_flow::Block& block, const data_flow::CodeNode& node) {
    auto& graph = block.dataflow();

    std::unordered_set<const data_flow::DataFlowNode*> to_delete = {&node};

    // Delete incoming
    std::list<const data_flow::Memlet*> iedges;
    for (auto& iedge : graph.in_edges(node)) {
        iedges.push_back(&iedge);
    }
    for (auto iedge : iedges) {
        auto& src = iedge->src();
        to_delete.insert(&src);

        auto edge = iedge->edge();
        graph.edges_.erase(edge);
        boost::remove_edge(edge, graph.graph_);
    }

    // Delete outgoing
    std::list<const data_flow::Memlet*> oedges;
    for (auto& oedge : graph.out_edges(node)) {
        oedges.push_back(&oedge);
    }
    for (auto oedge : oedges) {
        auto& dst = oedge->dst();
        to_delete.insert(&dst);

        auto edge = oedge->edge();
        graph.edges_.erase(edge);
        boost::remove_edge(edge, graph.graph_);
    }

    // Delete nodes
    for (auto obsolete_node : to_delete) {
        if (graph.in_degree(*obsolete_node) == 0 && graph.out_degree(*obsolete_node) == 0) {
            auto vertex = obsolete_node->vertex();
            graph.nodes_.erase(vertex);
            boost::remove_vertex(vertex, graph.graph_);
        }
    }
};

void StructuredSDFGBuilder::clear_node(structured_control_flow::Block& block, const data_flow::AccessNode& node) {
    auto& graph = block.dataflow();

    std::list<const data_flow::Memlet*> tmp;
    std::list<const data_flow::DataFlowNode*> queue = {&node};
    while (!queue.empty()) {
        auto current = queue.front();
        queue.pop_front();
        if (current != &node) {
            if (dynamic_cast<const data_flow::AccessNode*>(current)) {
                if (graph.in_degree(*current) > 0 || graph.out_degree(*current) > 0) {
                    continue;
                }
            }
        }

        tmp.clear();
        for (auto& iedge : graph.in_edges(*current)) {
            tmp.push_back(&iedge);
        }
        for (auto iedge : tmp) {
            auto& src = iedge->src();
            queue.push_back(&src);

            auto edge = iedge->edge();
            graph.edges_.erase(edge);
            boost::remove_edge(edge, graph.graph_);
        }

        if (current != &node || graph.out_degree(*current) == 0) {
            auto vertex = current->vertex();
            graph.nodes_.erase(vertex);
            boost::remove_vertex(vertex, graph.graph_);
        }
    }
};

void StructuredSDFGBuilder::add_dataflow(const data_flow::DataFlowGraph& from, Block& to) {
    auto& to_dataflow = to.dataflow();

    std::unordered_map<graph::Vertex, graph::Vertex> node_mapping;
    for (auto& entry : from.nodes_) {
        auto vertex = boost::add_vertex(to_dataflow.graph_);
        to_dataflow.nodes_.insert({vertex, entry.second->clone(this->new_element_id(), vertex, to_dataflow)});
        node_mapping.insert({entry.first, vertex});
    }

    for (auto& entry : from.edges_) {
        auto src = node_mapping[entry.second->src().vertex()];
        auto dst = node_mapping[entry.second->dst().vertex()];

        auto edge = boost::add_edge(src, dst, to_dataflow.graph_);

        to_dataflow.edges_.insert(
            {edge.first,
             entry.second->clone(
                 this->new_element_id(), edge.first, to_dataflow, *to_dataflow.nodes_[src], *to_dataflow.nodes_[dst]
             )}
        );
    }
};

void StructuredSDFGBuilder::merge_siblings(data_flow::AccessNode& source_node) {
    auto& user_graph = source_node.get_parent();
    auto* block = dynamic_cast<structured_control_flow::Block*>(user_graph.get_parent());
    if (!block) {
        throw InvalidSDFGException("Parent of user graph must be a block!");
    }

    // Merge access nodes if they access the same container on a code node
    for (auto& oedge : user_graph.out_edges(source_node)) {
        if (auto* code_node = dynamic_cast<data_flow::CodeNode*>(&oedge.dst())) {
            std::unordered_set<data_flow::Memlet*> iedges;
            for (auto& iedge : user_graph.in_edges(*code_node)) {
                iedges.insert(&iedge);
            }
            for (auto* iedge : iedges) {
                if (dynamic_cast<data_flow::ConstantNode*>(&iedge->src())) {
                    continue;
                }
                auto* access_node = static_cast<data_flow::AccessNode*>(&iedge->src());
                if (access_node == &source_node || access_node->data() != source_node.data()) {
                    continue;
                }
                this->add_memlet(
                    *block,
                    source_node,
                    iedge->src_conn(),
                    *code_node,
                    iedge->dst_conn(),
                    iedge->subset(),
                    iedge->base_type(),
                    iedge->debug_info()
                );
                this->remove_memlet(*block, *iedge);
                source_node.set_debug_info(DebugInfo::merge(source_node.debug_info(), access_node->debug_info()));
            }
        }
    }

    // Also merge "output" access nodes if they access the same container on a library node
    for (auto& iedge : user_graph.in_edges(source_node)) {
        if (auto* libnode = dynamic_cast<data_flow::LibraryNode*>(&iedge.src())) {
            std::unordered_set<data_flow::Memlet*> oedges;
            for (auto& oedge : user_graph.out_edges(*libnode)) {
                oedges.insert(&oedge);
            }
            for (auto* oedge : oedges) {
                auto* access_node = static_cast<data_flow::AccessNode*>(&oedge->dst());
                if (access_node == &source_node || access_node->data() != source_node.data()) {
                    continue;
                }
                this->add_memlet(
                    *block,
                    *libnode,
                    oedge->src_conn(),
                    source_node,
                    oedge->dst_conn(),
                    oedge->subset(),
                    oedge->base_type(),
                    oedge->debug_info()
                );
                this->remove_memlet(*block, *oedge);
                source_node.set_debug_info(DebugInfo::merge(source_node.debug_info(), access_node->debug_info()));
            }
        }
    }
}

} // namespace builder
} // namespace sdfg
