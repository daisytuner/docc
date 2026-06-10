#include "sdfg/analysis/escape_analysis.h"

#include <unordered_set>

#include "sdfg/analysis/scope_analysis.h"
#include "sdfg/analysis/users.h"
#include "sdfg/data_flow/library_nodes/stdlib/malloc.h"
#include "sdfg/structured_control_flow/block.h"
#include "sdfg/structured_control_flow/for.h"
#include "sdfg/structured_control_flow/map.h"
#include "sdfg/structured_control_flow/return.h"
#include "sdfg/structured_control_flow/sequence.h"
#include "sdfg/structured_control_flow/structured_loop.h"
#include "sdfg/structured_control_flow/while.h"
#include "sdfg/visitor/structured_sdfg_visitor.h"

namespace sdfg {
namespace analysis {

namespace {

bool is_loop(structured_control_flow::ControlFlowNode* node) {
    return dynamic_cast<structured_control_flow::StructuredLoop*>(node) != nullptr ||
           dynamic_cast<structured_control_flow::While*>(node) != nullptr;
}

structured_control_flow::ControlFlowNode* find_containing_loop(structured_control_flow::Block* block) {
    auto ancestor = block->get_parent();
    while (ancestor) {
        if (is_loop(ancestor)) {
            return ancestor;
        } else {
            ancestor = ancestor->get_parent();
        }
    }
    return nullptr;
}

class MallocFinder : public visitor::ActualStructuredSDFGVisitor {
public:
    std::unordered_map<std::string, stdlib::MallocNode*> malloc_containers;
    std::unordered_map<std::string, structured_control_flow::Block*> malloc_blocks;

    bool visit(structured_control_flow::Block& block) override {
        auto& dataflow = block.dataflow();
        for (auto* lib_node : dataflow.library_nodes()) {
            if (lib_node->code() == stdlib::LibraryNodeType_Malloc) {
                auto* malloc_node = dynamic_cast<stdlib::MallocNode*>(lib_node);
                if (malloc_node) {
                    // Find the output access node to get the container name
                    for (auto& oedge : dataflow.out_edges(*malloc_node)) {
                        if (auto* access = dynamic_cast<data_flow::AccessNode*>(&oedge.dst())) {
                            malloc_containers[access->data()] = malloc_node;
                            malloc_blocks[access->data()] = &block;
                        }
                    }
                }
            }
        }
        return true;
    }
};

class ReturnEscapeChecker : public visitor::ActualStructuredSDFGVisitor {
public:
    std::unordered_set<std::string> returned_containers;

    bool visit(structured_control_flow::Return& ret) override {
        if (ret.is_data()) {
            returned_containers.insert(ret.data());
        }
        return true;
    }
};

} // anonymous namespace

EscapeAnalysis::EscapeAnalysis(StructuredSDFG& sdfg) : Analysis(sdfg) {}

void EscapeAnalysis::run(analysis::AnalysisManager& analysis_manager) {
    auto& sdfg = this->sdfg_;
    auto& users = analysis_manager.get<Users>();

    // Step 1: Find all malloc allocations
    MallocFinder malloc_finder;
    malloc_finder.dispatch(sdfg.root());

    for (auto& [container, malloc_node] : malloc_finder.malloc_containers) {
        malloc_containers_.insert(container);
    }
    for (auto& [container, block] : malloc_finder.malloc_blocks) {
        malloc_blocks_[container] = block;
    }

    // Step 2: Find all containers that escape through return
    ReturnEscapeChecker return_checker;
    return_checker.dispatch(sdfg.root());

    // Step 3: For each malloc allocation, determine if it escapes
    for (const auto& container : malloc_containers_) {
        bool container_escapes = false;

        // Check 1: Returned from function
        if (return_checker.returned_containers.count(container) > 0) {
            container_escapes = true;
        }

        // Check 2: Stored into a non-transient (argument/global) container
        // This happens when the pointer is VIEWed or MOVEd into another container
        if (!container_escapes) {
            auto views = users.views(container);
            for (auto* view : views) {
                container_escapes = true;
                break;
            }
        }

        // Check 3: If the container is not transient, it's an argument and escapes
        if (!container_escapes && !sdfg.is_transient(container)) {
            container_escapes = true;
        }

        escapes_[container] = container_escapes;

        // Step 4: For non-escaping allocations, find the last use
        if (!container_escapes) {
            auto* malloc_block = malloc_blocks_[container];
            auto* containing_loop = find_containing_loop(malloc_block);

            User* last = nullptr;

            if (containing_loop) {
                // Malloc is inside a loop - use UsersView to find last use within the loop body
                UsersView loop_users(users, *containing_loop);
                auto loop_uses = loop_users.uses(container);

                // Find the last use within this loop iteration
                for (auto* use : loop_uses) {
                    auto uses_after = loop_users.all_uses_after(*use);
                    bool is_last = true;
                    for (auto* after : uses_after) {
                        if (after->container() == container) {
                            is_last = false;
                            break;
                        }
                    }
                    if (is_last) {
                        last = use;
                        break;
                    }
                }
            } else {
                // Malloc is not inside a loop - check if uses are inside loops
                auto all_uses = users.uses(container);

                // Find the outermost loop containing any use (to determine proper last use)
                structured_control_flow::ControlFlowNode* outermost_use_loop = nullptr;
                for (auto* use : all_uses) {
                    auto* use_element = use->element();
                    structured_control_flow::Block* use_block = nullptr;
                    if (auto* access = dynamic_cast<data_flow::AccessNode*>(use_element)) {
                        use_block = dynamic_cast<structured_control_flow::Block*>(access->get_parent().get_parent());
                    } else if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(use_element)) {
                        use_block = dynamic_cast<structured_control_flow::Block*>(lib->get_parent().get_parent());
                    }
                    if (use_block) {
                        auto* use_loop = find_containing_loop(use_block);
                        if (use_loop && !outermost_use_loop) {
                            outermost_use_loop = use_loop;
                        }
                    }
                }

                if (outermost_use_loop) {
                    // Look for uses after the loop
                    for (auto* use : all_uses) {
                        auto* use_element = use->element();
                        structured_control_flow::Block* use_block = nullptr;
                        if (auto* access = dynamic_cast<data_flow::AccessNode*>(use_element)) {
                            use_block = dynamic_cast<structured_control_flow::Block*>(access->get_parent().get_parent()
                            );
                        } else if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(use_element)) {
                            use_block = dynamic_cast<structured_control_flow::Block*>(lib->get_parent().get_parent());
                        }
                        if (use_block) {
                            auto* use_loop = find_containing_loop(use_block);
                            // If this use is not inside the loop, it could be the last use
                            if (!use_loop) {
                                auto uses_after = users.all_uses_after(*use);
                                bool is_last = true;
                                for (auto* after : uses_after) {
                                    if (after->container() == container) {
                                        // Check if 'after' is also outside the loop
                                        auto* after_element = after->element();
                                        structured_control_flow::Block* after_block = nullptr;
                                        if (auto* acc = dynamic_cast<data_flow::AccessNode*>(after_element)) {
                                            after_block =
                                                dynamic_cast<structured_control_flow::Block*>(acc->get_parent()
                                                                                                  .get_parent());
                                        } else if (auto* lib = dynamic_cast<data_flow::LibraryNode*>(after_element)) {
                                            after_block =
                                                dynamic_cast<structured_control_flow::Block*>(lib->get_parent()
                                                                                                  .get_parent());
                                        }
                                        if (after_block) {
                                            auto* after_loop = find_containing_loop(after_block);
                                            if (!after_loop) {
                                                is_last = false;
                                                break;
                                            }
                                        }
                                    }
                                }
                                if (is_last) {
                                    last = use;
                                    break;
                                }
                            }
                        }
                    }

                    if (!last) {
                        UsersView loop_users(users, *outermost_use_loop);
                        auto loop_uses = loop_users.uses(container);
                        for (auto* use : loop_uses) {
                            auto uses_after = loop_users.all_uses_after(*use);
                            bool is_last = true;
                            for (auto* after : uses_after) {
                                if (after->container() == container) {
                                    is_last = false;
                                    break;
                                }
                            }
                            if (is_last) {
                                last = use;
                                break;
                            }
                        }
                    }
                } else {
                    // No loops involved - simple case
                    for (auto* use : all_uses) {
                        auto uses_after = users.all_uses_after(*use);
                        bool is_last = true;
                        for (auto* after : uses_after) {
                            if (after->container() == container) {
                                is_last = false;
                                break;
                            }
                        }
                        if (is_last) {
                            last = use;
                            break;
                        }
                    }
                }
            }

            last_uses_[container] = last;
        }
    }
}

bool EscapeAnalysis::is_malloc_allocation(const std::string& container) const {
    return malloc_containers_.count(container) > 0;
}

bool EscapeAnalysis::escapes(const std::string& container) const {
    auto it = escapes_.find(container);
    if (it == escapes_.end()) {
        return false; // Not a malloc allocation
    }
    return it->second;
}

User* EscapeAnalysis::last_use(const std::string& container) const {
    auto it = last_uses_.find(container);
    if (it == last_uses_.end()) {
        return nullptr;
    }
    return it->second;
}

structured_control_flow::Block* EscapeAnalysis::malloc_block(const std::string& container) const {
    auto it = malloc_blocks_.find(container);
    if (it == malloc_blocks_.end()) {
        return nullptr;
    }
    return it->second;
}

std::unordered_set<std::string> EscapeAnalysis::non_escaping_allocations() const {
    std::unordered_set<std::string> result;
    for (const auto& container : malloc_containers_) {
        if (!escapes(container)) {
            result.insert(container);
        }
    }
    return result;
}

} // namespace analysis
} // namespace sdfg
